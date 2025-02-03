import timm 
import torch
import dataset
import os 
import random
import numpy as np 
from curses.ascii import FF
from models import CentralizedFashion
from torch import nn
import argparse

from dataset import skinCancer, bloodmnisit, isic2019, other


def centralized(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print('Creating Loggings Directory!')
    save_dir = f'{args.model_name}_{args.lr}lr_{args.dataset_name}_{args.Epochs}rounds_Centralized'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Getting the Dataset and Dataloader!')
    if args.dataset_name == 'HAM': 
        num_classes = 7
        train_loader, test_loader,_,_ = skinCancer(input_size= args.input_size, batch_size = args.batch_size, base_dir= args.base_dir, num_workers=args.num_workers)
        num_channels = 3

    elif args.dataset_name == 'bloodmnist':
        num_classes = 8
        train_loader, test_loader,_,_ = bloodmnisit(input_size= args.input_size, batch_size = args.batch_size, download= True, num_workers=args.num_workers)
        num_channels = 3

    elif args.dataset_name == 'isic2019': 
        num_classes = 8
        _, _, train_loader, _, _, test_loader = isic2019(input_size= args.input_size, batch_size = args.batch_size, root_dir=args.root_dir, csv_file_path=args.csv_file_path, num_workers=args.num_workers)
        num_channels = 3

    elif args.dataset_name == 'other': 
        num_classes = args.num_classes
        _, _, train_loader, _, _, test_loader = other(input_size= args.input_size, batch_size = args.batch_size, root_dir=args.root_dir, csv_file_path=args.csv_file_path, num_workers=args.num_workers)
        num_channels = 3

    print('Getting the model from timm library!')
    model = timm.create_model(
        model_name= args.model_name, pretrained= args.pretrained,
        num_classes = num_classes, in_chans=num_channels
        ).to(device)


    criterion = torch.nn.CrossEntropyLoss()

    centralized_network = CentralizedFashion(
        device= device, network=model, criterion= criterion,
        base_dir=save_dir
        )

    #Instantiate metrics and set optimizer
    centralized_network.init_logs()
    centralized_network.set_optimizer(name=args.opt_name, lr = args.lr)

    print(f'Train Centralized Fashion:\n model: {args.model_name}\n dataset: {args.dataset_name}\n LR: {args.lr}\n Number of Epochs: {args.Epochs}\n Loggings: {save_dir}\n')
    print('Start Training! \n')

    #Training and Evaluation Loop
    for r in range(args.Epochs):
        print(f"Round {r+1} / {args.Epochs}")
        centralized_network.train_round(train_loader)
        centralized_network.eval_round(test_loader)
        print('---------')
        if (r+1) % args.save_every_epochs == 0 and r != 0: 
            centralized_network.save_pickles(save_dir)
        print('============================================')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Centralized Experiments')

    parser.add_argument('--dataset_name', type=str, choices=['HAM', 'bloodmnist', 'isic2019', 'other'], help='Dataset Name')
    parser.add_argument('--input_size',  type=int, default= 224, help='Input size --> (input_size, input_size), default : 224')
    parser.add_argument('--num_workers',  type=int, default= 8, help='Number of workers for dataloaders, default : 8')
    parser.add_argument('--model_name', type=str, default= 'vit_base_r50_s16_224', help='Model name from timm library, default: vit_base_r50_s16_224')
    parser.add_argument('--pretrained', type=bool, default= False, help='Pretrained weights flag, default: False')
    parser.add_argument('--batch_size',  type=int, default= 32, help='Batch size, default : 32')
    parser.add_argument('--Epochs',  type=int, default= 200, help='Number of Epochs, default : 200')
    parser.add_argument('--opt_name', type=str, choices=['Adam'], default = 'Adam', help='Optimizer name, only ADAM optimizer is available')
    parser.add_argument('--lr',  type=float, default= 1e-4, help='Learning rate, default : 1e-4')
    parser.add_argument('--save_every_epochs',  type=int, default= 10, help='Save metrics every this number of epochs, default: 10')
    parser.add_argument('--seed',  type=int, default= 105, help='Seed, default: 105')
    parser.add_argument('--base_dir', type=str, default= None, help='')
    parser.add_argument('--root_dir', type=str, default= None, help='')
    parser.add_argument('--csv_file_path', type=str, default=None, help='')
    parser.add_argument('--num_classes',  type=int, default= 2, help='Number of classes for other dataset, default: 2')

    args = parser.parse_args()

    centralized(args)