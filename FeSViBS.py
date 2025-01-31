import os 
import numpy as np
import models 
import random
from dataset import skinCancer, bloodmnisit, isic2019, other
from utils import weight_dec_global, weight_vec
import argparse 
import torch as torch
from torch import nn




def fesvibs(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.fesvibs_arg: 
        method_flag = 'FeSViBS'
    else:
        method_flag = 'SViBS'

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'

    if args.DP:
        std = np.sqrt(2 * np.math.log(1.25/args.delta)) / args.epsilon 
        mean=0
        dir_name = f"{args.model_name}_{args.lr}lr_{args.dataset_name}_{args.num_clients}Clients_{args.initial_block}to{args.final_block}Blocks_{args.batch_size}Batch__{args.epsilon,args.delta}DP_{method_flag}"
    else:
        mean = 0
        std = 0
        dir_name = f"{args.model_name}_{args.lr}lr_{args.dataset_name}_{args.num_clients}Clients_{args.initial_block}to{args.final_block}Blocks_{args.batch_size}Batch_{method_flag}"

    save_dir = f'{dir_name}' 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    

    print(f"Logging to: {dir_name}")

    print('Getting the Dataset and Dataloader!')
    if args.dataset_name == 'HAM': 
        num_classes = 7
        _, _, traindataset, testdataset = skinCancer(input_size= args.input_size, batch_size = args.batch_size, base_dir= args.base_dir, num_workers=args.num_workers)
        num_channels = 3

    elif args.dataset_name == 'bloodmnist':
        num_classes = 8
        _, _, traindataset, testdataset = bloodmnisit(input_size= args.input_size, batch_size = args.batch_size, download= True, num_workers=args.num_workers)
        num_channels = 3

    elif args.dataset_name == 'isic2019': 
        num_classes = 8
        DATALOADERS, _, _, _, _, test_loader = isic2019(input_size= args.input_size, batch_size = args.batch_size, root_dir=args.root_dir, csv_file_path=args.csv_file_path, num_workers=args.num_workers)
        num_channels = 3

    elif args.dataset_name == 'other': 
        num_classes = args.num_classes
        DATALOADERS, _, _, _, _, test_loader = other(input_size= args.input_size, batch_size = args.batch_size, root_dir=args.root_dir, csv_file_path=args.csv_file_path, num_workers=args.num_workers, num_clients = args.num_clients)
        num_channels = 3

    criterion = nn.CrossEntropyLoss()

    fesvibs_network = models.FeSVBiS(
            ViT_name= args.model_name, num_classes= num_classes,
            num_clients = args.num_clients, in_channels = num_channels,
            ViT_pretrained= args.pretrained,
            initial_block= args.initial_block, final_block= args.final_block,
            resnet_dropout= args.resnet_dropout, DP=args.DP, mean= mean, std= std
            ).to(device)
    
    Split = models.SplitFeSViBS(
        num_clients=args.num_clients, device = device, network = fesvibs_network, 
        criterion = criterion, base_dir=save_dir, 
        initial_block= args.initial_block, final_block= args.final_block,
        )
    

    if args.dataset_name != 'isic2019' and args.dataset_name != 'other':
        print('Distribute Images Among Clients')
        Split.distribute_images(dataset_name=args.dataset_name, train_data= traindataset,test_data= testdataset ,batch_size = args.batch_size)  
    else: 
        Split.CLIENTS_DATALOADERS = DATALOADERS
        Split.testloader = test_loader

    Split.set_optimizer(args.opt_name, lr = args.lr)
    Split.init_logs()

    print('Start Training! \n')

    for r in range(args.Epochs):
        print(f"Round {r+1} / {args.Epochs}")
        agg_weights = None
        for client_i in range(args.num_clients):
            weight_dict = Split.train_round(client_i)
            if client_i == 0: 
                agg_weights = weight_dict
            else: 
                agg_weights['blocks'] +=  weight_dict['blocks']
                agg_weights['cls'] +=  weight_dict['cls']
                agg_weights['pos_embed'] +=  weight_dict['pos_embed']
                
        agg_weights['blocks'] /= args.num_clients
        agg_weights['cls'] /= args.num_clients
        agg_weights['pos_embed'] /= args.num_clients  

        
        Split.network.vit.blocks = weight_dec_global(
            Split.network.vit.blocks,
            agg_weights['blocks'].to(device)
            )
        
        Split.network.vit.cls_token.data = agg_weights['cls'].to(device) + 0.0
        Split.network.vit.pos_embed.data = agg_weights['pos_embed'].to(device) + 0.0

        if args.fesvibs_arg and ((r+1) % args.local_round == 0 and r!= 0):
                print('========================== \t \t Federation \t \t ==========================')
                tails_weights = []
                head_weights = []
                for head, tail in zip(Split.network.resnet50_clients, Split.network.mlp_clients_tail):
                    head_weights.append(weight_vec(head).detach().cpu())
                    tails_weights.append(weight_vec(tail).detach().cpu())
                
                mean_avg_tail = torch.mean(torch.stack(tails_weights), axis = 0)
                mean_avg_head = torch.mean(torch.stack(head_weights), axis = 0)

                for i in range(args.num_clients):
                    Split.network.mlp_clients_tail[i] = weight_dec_global(Split.network.mlp_clients_tail[i], 
                                                                        mean_avg_tail.to(device))
                    Split.network.resnet50_clients[i] = weight_dec_global(Split.network.resnet50_clients[i], 
                                                                        mean_avg_head.to(device))
       
        for client_i in range(args.num_clients):
            Split.eval_round(client_i)
            
        print('---------')

        if (r+1) % args.save_every_epochs == 0 and r != 0: 
            Split.save_pickles(save_dir)
        print('============================================')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Centralized Experiments')
    parser.add_argument('--dataset_name', type=str, choices=['HAM', 'bloodmnist', 'isic2019', 'other'], help='Dataset Name')
    parser.add_argument('--input_size',  type=int, default= 224, help='Input size --> (input_size, input_size), default : 224')
    parser.add_argument('--local_round',  type=int, default= 2, help='Local round before federation in FeSViBS, default : 2')
    parser.add_argument('--num_workers',  type=int, default= 8, help='Number of workers for dataloaders, default : 8')
    parser.add_argument('--initial_block',  type=int, default= 1, help='Initial Block, default : 1')
    parser.add_argument('--final_block',  type=int, default= 6, help='Final Block, default : 6')
    parser.add_argument('--num_clients',  type=int, default= 6, help='Number of Clients, default : 6')
    parser.add_argument('--model_name', type=str, default= 'vit_base_r50_s16_224', help='Model name from timm library, default: vit_base_r50_s16_224')
    parser.add_argument('--pretrained', type=bool, default= False, help='Pretrained weights flag, default: False')
    parser.add_argument('--fesvibs_arg', type=bool, default= False, help='Flag to indicate whether SViBS or FeSViBS, default: False')
    parser.add_argument('--batch_size',  type=int, default= 32, help='Batch size, default : 32')
    parser.add_argument('--Epochs',  type=int, default= 200, help='Number of Epochs, default : 200')
    parser.add_argument('--opt_name', type=str, choices=['Adam'], default = 'Adam', help='Optimizer name, only ADAM optimizer is available')
    parser.add_argument('--lr',  type=float, default= 1e-4, help='Learning rate, default : 1e-4')
    parser.add_argument('--save_every_epochs',  type=int, default= 10, help='Save metrics every this number of epochs, default: 10')
    parser.add_argument('--seed',  type=int, default= 105, help='Seed, default: 105')
    parser.add_argument('--base_dir', type=str, default= None, help='')
    parser.add_argument('--root_dir', type=str, default= None, help='')
    parser.add_argument('--csv_file_path', type=str, default=None, help='')
    parser.add_argument('--DP', type=bool, default= False, help='Differential Privacy , default: False')
    parser.add_argument('--epsilon',  type=float, default= 0, help='Epsilon Value for differential privacy')
    parser.add_argument('--delta',  type=float, default= 0.00001, help='Delta Value for differential privacy')
    parser.add_argument('--resnet_dropout',  type=float, default= 0.5, help='ResNet Dropout, Default: 0.5')
    parser.add_argument('--num_classes',  type=int, default= 2, help='Number of classes for other dataset, default: 2')
    args = parser.parse_args()

    fesvibs(args)