from tqdm import tqdm
import pickle as pkl
import os 
import timm 
import copy 
import numpy as np 

import torch.nn as nn 
import torch 
from sklearn.metrics import balanced_accuracy_score

from dataset import blood_noniid, bloodmnisit, distribute_data
from utils import weight_vec

import torch.nn.functional as F
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue
import random

transformations = {
    0: {'brightness': 1.4, 'contrast': 1.2, 'hue': 0.08}, 
    1: {'brightness': 1.2, 'contrast': 1.1, 'hue': 0.06}, 
    2: {'brightness': 1.0, 'contrast': 1.0, 'hue': 0.04}, 
    3: {'brightness': 0.8, 'contrast': 0.8, 'hue': 0.02}, 
    4: {'brightness': 0.7, 'contrast': 0.7, 'hue': 0.01},  
    5: {'brightness': 0.6, 'contrast': 0.6, 'hue': 0.0},  
}

def  transition_batch_images(batch_tensor, current_transformation, target_transformations):
    """
    Transforms a batch of images from a single current transformation to per-image target transformations.
    
    Args:
        batch_tensor (torch.Tensor): Input batch of images, shape (N, C, H, W).
        current_transformation (int): Scalar index of the current transformation (applies to all images).
        target_transformations (torch.Tensor): Tensor of shape (N,) with target transformations for each image.
    
    Returns:
        torch.Tensor: Transformed batch of images, shape (N, C, H, W).
    """
    # Get parameters for the current transformation
    current_params = transformations[current_transformation]
    
    # Initialize a list to store transformed images
    x_new = []
    y_new = []
    
    # Process each image in the batch
    batch_size = batch_tensor.shape[0]
    for idx, (img, target_transformation) in enumerate(zip(batch_tensor, target_transformations)):
        if idx % 2 == 0:
            # Get target parameters for the current image
            target_params = transformations[int(target_transformation.item())]
            
            # Calculate relative adjustments
            relative_brightness = target_params['brightness'] / current_params['brightness']
            relative_contrast = target_params['contrast'] / current_params['contrast']
            relative_hue = target_params['hue'] - current_params['hue']
            
            # Apply relative adjustments
            img = adjust_brightness(img, relative_brightness)
            img = adjust_contrast(img, relative_contrast)
            img = adjust_hue(img, relative_hue)
            y_new.append(0)  # Label 0: Similar
        else:
            while True:
                rand_idx = random.randint(0, batch_size - 1)
                if rand_idx != idx:  # Ensure it's a different image
                    break
            img = batch_tensor[rand_idx]
            y_new.append(1)  # Label 1: Dissimilar
        x_new.append(img)
    
    # Stack the transformed images back into a batch
    x_new = torch.stack(x_new)
    y_new = torch.tensor(y_new, dtype=torch.float32)
    
    return x_new, y_new

class CentralizedFashion(): 
    def __init__(self, device, network, criterion, base_dir):
        """
            Class for Centralized Paradigm.    
            args:
                device: cuda vs cpu
                network: ViT model
                criterion: loss function to be used
                base_dir: where to save metrics as pickles
            return: 
                None 
        """
        self.device = device
        self.network = network
        self.criterion = criterion
        self.base_dir = base_dir

    def set_optimizer(self, name, lr):
        """
        name: Optimizer name, e.g. Adam 
        lr: learning rate 

        """
        if name == 'Adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr)

    def init_logs(self):
            """
            A method to initialize dictionaries for the metrics
            return : None 
            args: None
            """
            self.losses  = {'train':[], 'test':[]}
            self.balanced_accs = {'train':[], 'test':[]}

    def train_round(self, train_loader):
        """
        Training loop. 

        """
        running_loss = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        for imgs, labels in tqdm(train_loader): 
            self.optimizer.zero_grad()
            imgs, labels = imgs.to(self.device),labels.to(self.device)
            output = self.network(imgs)
            labels = labels.reshape(labels.shape[0])
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() 
            _, predicted = torch.max(output, 1)
            whole_probs.append(torch.nn.Softmax(dim = -1)(output).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu())    
        self.metrics(whole_labels, whole_preds, running_loss, len(train_loader), whole_probs, train = True)
        
    def eval_round(self, test_loader):
        """
        Evaluation loop. 

        client_i: Client index.
                
        """
        running_loss = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader): 
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                output = self.network(imgs)
                labels = labels.reshape(labels.shape[0])
                loss = self.criterion(output, labels)
                running_loss += loss.item() 
                _, predicted = torch.max(output, 1)
                whole_probs.append(torch.nn.Softmax(dim = -1)(output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())    
            self.metrics(whole_labels, whole_preds, running_loss, len(test_loader), whole_probs, train= False)
    
    def metrics(self, whole_labels, whole_preds, running_loss, len_loader, whole_probs, train):
        """
        Save metrics as pickle files and the model as .pt file.
        
        """
        whole_labels = torch.cat(whole_labels)
        whole_preds = torch.cat(whole_preds)
        loss_epoch = running_loss/len_loader
        balanced_acc = balanced_accuracy_score(whole_labels.detach().cpu(),whole_preds.detach().cpu())
        if train == True:
            eval_name = 'train'
        else:
            eval_name = 'test'

        self.losses[eval_name].append(loss_epoch)
        self.balanced_accs[eval_name].append(balanced_acc)
        
        print(f"{eval_name}:")
        print(f"{eval_name}_loss :{loss_epoch:.3f}")
        print(f"{eval_name}_balanced_acc :{balanced_acc:.3f}")


    def save_pickles(self, base_dir, local= None, client_id=None): 
        if local and client_id: 
            with open(os.path.join(base_dir,f'loss_epoch_Client{client_id}'), 'wb') as handle:
                pkl.dump(self.losses, handle)
            with open(os.path.join(base_dir,f'balanced_accs{client_id}'), 'wb') as handle:
                pkl.dump(self.balanced_accs, handle)
        else: 
            with open(os.path.join(base_dir,'loss_epoch'), 'wb') as handle:
                pkl.dump(self.losses, handle)
            with open(os.path.join(base_dir,f'balanced_accs'), 'wb') as handle:
                pkl.dump(self.balanced_accs, handle)

class SLViT(nn.Module): 
    def __init__(
        self, ViT_name, num_classes , num_clients=6, 
        in_channels=3, ViT_pretrained = False,
        diff_privacy = False, mean = 0, std = 1
        ) -> None:

        super().__init__()

        self.vit = timm.create_model(
            model_name = ViT_name,
            pretrained = ViT_pretrained,
            num_classes = num_classes,
            in_chans = in_channels
        )   
        client_tail = MLP_cls_classes(num_classes= num_classes)
        self.mlp_clients_tail =  nn.ModuleList([copy.deepcopy(client_tail)for i in range(num_clients)])
        self.resnet50_clients = nn.ModuleList([copy.deepcopy(self.vit.patch_embed) for i in range(num_clients)]) 
        
        self.diff_privacy = diff_privacy
        self.mean = mean 
        self.std = std

    def forward(self, x, client_idx):
        x = self.resnet50_clients[client_idx](x)
        if self.diff_privacy == True:
            noise = torch.randn(size= x.shape).cuda() * self.std + self.mean
            x = x + noise
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for block_num in range(12):
            x = self.vit.blocks[block_num](x)
        x = self.vit.norm(x)
        cls = self.vit.pre_logits(x)[:,0,:]
        x = self.mlp_clients_tail[client_idx](cls)
        return x, cls

class MLP_cls_classes(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.identity = nn.Identity()
        self.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.identity(x)
        x = self.fc(x)
        return x 

class SplitNetwork():
    def __init__(
        self, num_clients, device, network, 
        criterion, base_dir,
        ):
        """
        args:
            num_clients
            device: cuda vs cpu
            network: ViT model
            criterion: loss function to be used
            base_dir: where to save pickles/model files
        """
        
        self.device = device
        self.num_clients = num_clients
        self.criterion = criterion
        self.network = network
        self.base_dir = base_dir

    def init_logs(self):
        """
        This method initializes dictionaries for the metrics

        """
        self.losses  = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.reg_losses  = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}
        self.balanced_accs = {'train':[[] for i in range(self.num_clients)], 'test':[[] for i in range(self.num_clients)]}

    def set_optimizer(self, name, lr):
        """
        name: Optimizer name, e.g. Adam 
        lr: learning rate 

        """
        if name == 'Adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr)
    
    def distribute_images(self, dataset_name ,train_data, test_data, batch_size):
        """
        This method splits the dataset among clients.
        train_data: train dataset 
        test_data: test dataset 
        batch_size: batch size

        """
        if dataset_name == 'HAM':
            self.CLIENTS_DATALOADERS = distribute_data(self.num_clients, train_data, batch_size)
            self.testloader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, num_workers= 8)
            
        elif dataset_name == 'bloodmnist':
            _, self.testloader, train_dataset, _ = bloodmnisit(batch_size= batch_size)
            _, self.CLIENTS_DATALOADERS, _ = blood_noniid(self.num_clients, train_dataset, batch_size =batch_size)  

    def train_round(self, client_i):
        """
        Training loop. 

        client_i: Client index.

        """
        running_loss_client_i = 0
        mel_running_loss = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        copy_network = copy.deepcopy(self.network)
        weight_dic = {'blocks':None, 'cls':None, 'pos_embed':None}
        self.network.train()
        for data in tqdm(self.CLIENTS_DATALOADERS[client_i]): 
            self.optimizer.zero_grad()
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.reshape(labels.shape[0])
            tail_output = self.network(imgs, client_i)
            loss = self.criterion(tail_output[0], labels)
            loss.backward()
            self.optimizer.step()
            running_loss_client_i+= loss.item() 
            _, predicted = torch.max(tail_output[0], 1)
            whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output[0]).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu()) 
        self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.CLIENTS_DATALOADERS[client_i]), whole_probs, train = True)
        
        # if self.avg_body:
        weight_dic['blocks'] = weight_vec(self.network.vit.blocks).detach().cpu()
        weight_dic['cls'] = self.network.vit.cls_token.detach().cpu()
        weight_dic['pos_embed'] = self.network.vit.pos_embed.detach().cpu()

        self.network.vit.blocks = copy.deepcopy(copy_network.vit.blocks)
        self.network.vit.cls_token = copy.deepcopy(copy_network.vit.cls_token)
        self.network.vit.pos_embed =  copy.deepcopy(copy_network.vit.pos_embed)
        return weight_dic
            
    def eval_round(self, client_i):
        """
        Evaluation loop. 

        client_i: Client index.
                
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(self.testloader): 
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                tail_output = self.network(imgs, client_i)[0]
                labels = labels.reshape(labels.shape[0])
                loss = self.criterion(tail_output, labels)
                running_loss_client_i+= loss.item() 
                _, predicted = torch.max(tail_output, 1)
                whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())    
            self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.testloader), whole_probs, train= False)

    def metrics(self, client_i, whole_labels, whole_preds, running_loss_client_i, running_reg_loss_client_i, len_loader, whole_probs, train):
        """
        Save metrics as pickle files and the model as .pt file.
        
        """
        whole_labels = torch.cat(whole_labels)
        whole_preds = torch.cat(whole_preds)
        loss_epoch = running_loss_client_i/len_loader
        reg_loss_epoch = running_reg_loss_client_i/len_loader
        balanced_acc = balanced_accuracy_score(whole_labels.detach().cpu(), whole_preds.detach().cpu())
       
        if train == True:
            eval_name = 'train'
        else:
            eval_name = 'test'

        self.losses[eval_name][client_i].append(loss_epoch)
        self.reg_losses[eval_name][client_i].append(reg_loss_epoch)
        self.balanced_accs[eval_name][client_i].append(balanced_acc)

        print(f"client{client_i}_{eval_name}:")
        print(f" Loss {eval_name}:{loss_epoch:.3f}")
        print(f" Reg loss {eval_name}:{reg_loss_epoch:.3f}")
        print(f"balanced accuracy {eval_name}:{balanced_acc:.3f}")

    def save_pickles(self, base_dir): 
        with open(os.path.join(base_dir,'loss_epoch'), 'wb') as handle:
            pkl.dump(self.losses, handle)
        with open(os.path.join(base_dir,'reg_loss_epoch'), 'wb') as handle:
            pkl.dump(self.reg_losses, handle)
        with open(os.path.join(base_dir,'balanced_accs'), 'wb') as handle:
            pkl.dump(self.balanced_accs, handle)

class FeSVBiS(nn.Module): 
    def __init__(
        self, ViT_name, num_classes, alpha, device,
        num_clients=6, in_channels=3, ViT_pretrained=False, 
        initial_block=1, final_block=6, resnet_dropout = None, DP = False, mean = None, std = None
        ) -> None:
        super().__init__()

        self.initial_block = initial_block
        self.final_block = final_block

        self.vit = timm.create_model(
            model_name = ViT_name,
            pretrained = ViT_pretrained,
            num_classes = num_classes,
            in_chans = in_channels
        )   

        self.resnet50 = self.vit.patch_embed
        self.resnet50_clients = nn.ModuleList([copy.deepcopy(self.resnet50) for i in range(num_clients)])
        self.common_network = ResidualBlock(drop_out=resnet_dropout)
        client_tail = MLP_cls_classes(num_classes= num_classes)
        self.mlp_clients_tail =  nn.ModuleList([copy.deepcopy(client_tail) for i in range(num_clients)])
        self.DP = DP
        self.mean = mean
        self.std = std
        self.device = device
        self.alpha = alpha
        self.num_clients = num_clients
        self.criterion = ContrastiveLoss(margin=1.0)

    def forward(self, x, chosen_block, client_idx):
        x_orig = self.resnet50_clients[client_idx](x)

        if self.alpha != 0:
            batch_size = x.shape[0]
            d = client_idx
            d_new = torch.randint(0, self.num_clients, (batch_size, )).to(self.device)
            x_new, y_new = transition_batch_images(
                x, d, d_new
            )

            z = torch.mean(F.relu(x_orig), dim=1)
            z_new = torch.mean(F.relu(self.resnet50_clients[client_idx](x_new)), dim=1)
            reg = self.alpha * self.criterion(z_new, z, y_new.to(self.device))
        else:
            reg = 0
        x = x_orig

        if self.DP: 
            noise = torch.randn(size= x.shape).to(self.device) * self.std + self.mean
            x = x + noise
        for block_num in range(chosen_block):
            x = self.vit.blocks[block_num](x)
        x = self.common_network(x)
        x = self.mlp_clients_tail[client_idx](x)
        return x, reg


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, stride = 1, downsample = None, drop_out= None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.pool = nn.AvgPool2d(14, stride=1)
        self.dropout = nn.Dropout2d(p=drop_out)
        self.drop_out = drop_out

    def forward(self, x):
        if len(x.shape) == 3: 
            x = torch.permute(x,(0,-1,1))
            x = x.reshape(x.shape[0], x.shape[1] , 14, 14)
        residual = x
        out = self.conv1(x)
        if self.drop_out is not None: 
            out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.pool(out)
        return out.reshape(-1,768)

class SplitFeSViBS(SplitNetwork):
    def __init__(
        self, num_clients, device, 
        network, criterion, base_dir, 
        initial_block, final_block,
        ):

        self.initial_block = initial_block
        self.final_block   = final_block    
        self.num_clients = num_clients
        self.device = device
        self.network = network
        self.criterion = criterion
        self.base_dir = base_dir
        self.train_chosen_blocks = [0] * num_clients

    def set_optimizer_mel(self, name, lr):
        if name == 'Adam':
            self.optimizer_mel = [torch.optim.Adam(self.mel_body[i].parameters(), lr = lr) for i in range(self.num_clients)]

    def train_round(self, client_i):
        """
        Training loop. 

        client_i: Client index.

        """
        running_loss_client_i = 0
        running_reg_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        self.chosen_block = np.random.randint(low = self.initial_block, high= self.final_block+1) 
        self.train_chosen_blocks[client_i] =  self.chosen_block
        copy_network = copy.deepcopy(self.network)
        weight_dic = {}
        weight_dic['blocks'] = None
        weight_dic['cls'] = None
        weight_dic['pos_embed'] = None
        weight_dic['resnet'] = None
        print(f"Chosen Block:{self.chosen_block} for client {client_i}")
        self.network.train()
        for data in tqdm(self.CLIENTS_DATALOADERS[client_i]): 
            self.optimizer.zero_grad()
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.reshape(labels.shape[0])
            tail_output, reg_loss = self.network(x=imgs, chosen_block=self.chosen_block, client_idx = client_i)
            loss = self.criterion(tail_output, labels)
            (loss + reg_loss).backward()
            self.optimizer.step()
            running_loss_client_i+= loss.item() 
            if self.network.alpha != 0:
                running_reg_loss_client_i+= reg_loss.item() 
            _, predicted = torch.max(tail_output, 1)
            whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu()) 
        self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, running_reg_loss_client_i, len(self.CLIENTS_DATALOADERS[client_i]), whole_probs, train = True)
        
        weight_dic['blocks'] = weight_vec(self.network.vit.blocks).detach().cpu()
        weight_dic['cls'] = self.network.vit.cls_token.detach().cpu()
        weight_dic['pos_embed'] = self.network.vit.pos_embed.detach().cpu()
        
        self.network.vit.blocks = copy.deepcopy(copy_network.vit.blocks)
        self.network.vit.cls_token = copy.deepcopy(copy_network.vit.cls_token)
        self.network.vit.pos_embed =  copy.deepcopy(copy_network.vit.pos_embed)
        return weight_dic
    
    
    def fed_eval_round(self, client_i):
        """
        Federated evaluation loop. 

        client_i: Client index.
                
        """
        running_loss_client_i = 0
        running_reg_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        num_b = self.train_chosen_blocks[client_i]
        print(f"Chosen block for testing: {num_b}")
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(self.testloaders[client_i]): 
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                labels = labels.reshape(labels.shape[0])
                tail_output, reg_loss = self.network(x=imgs, chosen_block=num_b, client_idx = client_i)
                loss = self.criterion(tail_output, labels)
                running_loss_client_i+= loss.item() 
                if self.network.alpha != 0:
                    running_reg_loss_client_i+= reg_loss.item() 
                _, predicted = torch.max(tail_output, 1)
                whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())    
            self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, running_reg_loss_client_i, len(self.testloaders[client_i]), whole_probs, train= False)

    def eval_round(self, client_i):
        """
        Evaluation loop. 

        client_i: Client index.
                
        """
        running_loss_client_i = 0
        running_reg_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        num_b = self.train_chosen_blocks[client_i]
        print(f"Chosen block for testing: {num_b}")
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(self.testloader): 
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                labels = labels.reshape(labels.shape[0])
                tail_output, reg_loss = self.network(x=imgs, chosen_block=num_b, client_idx = client_i)
                loss = self.criterion(tail_output, labels)
                running_loss_client_i+= loss.item() 
                if self.network.alpha != 0:
                    running_reg_loss_client_i+= reg_loss.item() 
                _, predicted = torch.max(tail_output, 1)
                whole_probs.append(torch.nn.Softmax(dim = -1)(tail_output).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())    
            self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, running_reg_loss_client_i, len(self.testloaders[client_i]), whole_probs, train= False)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute the Euclidean distance between embeddings
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # print(f"output1.shape: {output1.shape}")
        # print(f"euclidean_distance.shape: {euclidean_distance.shape}")
        # print(f"label.shape: {label.shape}")
        # Contrastive loss formula
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +  # Similar pairs
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # Dissimilar pairs
        )
        return loss