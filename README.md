# FeSViBS
Source code for MICCAI 2023 paper entitled: 'FeSViBS: Federated Split Learning of Vision Transformer with Block Sampling'


<hr/>

![Method](Figures/method.PNG)

## Abstract
Data scarcity is a significant obstacle hindering the learning of powerful machine learning models in critical healthcare applications. Data-sharing mechanisms among multiple entities (e.g., hospitals) can accelerate model training and yield more accurate predictions. Recently, approaches such as Federated Learning (FL) and Split Learning (SL) have facilitated collaboration without the need to exchange private data. In this work, we propose a framework for medical imaging classification tasks called \textbf{Fe}derated \textbf{S}plit learning of \textbf{Vi}sion transformer with \textbf{B}lock \textbf{S}ampling \textbf{(FeSViBS)}. The FeSViBS framework builds upon the existing federated split vision transformer and introduces a \emph{block sampling} module, which leverages intermediate features extracted by the Vision Transformer (ViT) at the server. This is achieved by sampling features (patch tokens) from an intermediate transformer block and distilling their information content into a pseudo class token before passing them back to the client. These pseudo class tokens serve as an effective feature augmentation strategy and enhances the generalizability of the learned model. We demonstrate the utility of our proposed method compared to other SL and FL approaches on three publicly available medical imaging datasets: HAM1000, BloodMNIST, and Fed-ISIC2019, under both IID and non-IID settings.

## Install Dependinces
Install all dependincies by running the following command: 

```
pip install -r requirements.txt

```

## Datasets

We conduct all experiments on **three** datasets: 

1. HAM10000 -- 
2. Blood cells (BloodMNIST) -- MedMnist library 
3. Federated version of ISIC2019 dataset -- Flamby library 


## Running Centralized Training/Testing
In order to run  **Centralized Training** run the following command: 

```
python centralized.py  --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019]  --csv_file_path [Path to train_test_split csv for ISIC2019] --Epochs [Number of Epochs]

```


## Running Local Training/Testing for Each Client
In order to run  **Local Training/Testing** run the following command: 

```
python local.py  --local_arg True --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019]  --csv_file_path [Path to train_test_split csv for ISIC2019] --num_clients [Number of clients] --Epochs [Number of Epochs]

```

## Running Vanilla Split Learning with Vision Transformers (SLViT)
In order to run  **SLViT without** Differential Privacy (DP) run the following command: 

```
python SLViT.py --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019]  --csv_file_path [Path to train_test_split csv for ISIC2019] --num_clients [Number of clients] --Epochs [Number of Epochs]

```

**SLViT with** Differential Privacy (DP) run the following command:

```
python SLViT.py --DP True --epsilon [epsilon value] --delta [delta value] --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019] --csv_file_path [Path to train_test_split csv for ISIC2019] --num_clients [Number of clients] --Epochs [Number of Epochs] 

```

## Running Split Vision Transformer with Block Sampling (SViBS):
In order to run  **SViBS** run the following command: 

```
python FeSViBS.py --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019]  --csv_file_path [Path to train_test_split csv for ISIC2019] --num_clients [Number of clients] --Epochs [Number of Epochs] --initial_block 1 --final_block 6 

```

## Running Federated Split Vision Transformer with Block Sampling (FeSViBS):
In order to run  **FeSViBS without** Differential Privacy (DP) run the following command: 

```
python FeSViBS.py --fesvibs_arg True --local_round [number of local rounds before federation] --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019]  --csv_file_path [Path to train_test_split csv for ISIC2019] --num_clients [Number of clients] --Epochs [Number of Epochs] --initial_block 1 --final_block 6 

```

In order to run  **FeSViBS with** Differential Privacy (DP) run the following command: 

```
python FeSViBS.py --fesvibs_arg True --DP True --epsilon [epsilon value] --delta [delta value] --local_round [number of local rounds before federation] --dataset_name [choose the dataset name] --opt_name [default is Adam] --lr [learning rate] --seed [seed number] --base_dir [path data folder for HAM] --save_every_epochs [Save pickle files] --root_dir [Path to ISIC_2019_Training_Input_preprocessed for ISIC2019]  --csv_file_path [Path to train_test_split csv for ISIC2019] --num_clients [Number of clients] --Epochs [Number of Epochs] --initial_block 1 --final_block 6 

```