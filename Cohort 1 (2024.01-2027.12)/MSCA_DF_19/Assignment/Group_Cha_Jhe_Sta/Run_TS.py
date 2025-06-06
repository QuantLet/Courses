CUDA_LAUNCH_BLOCKING = 1.
import pandas as pd
import seaborn as sns
sns.set_theme('notebook', 'whitegrid')
import matplotlib.pyplot as plt
# Relevant imports 
import pandas as pd
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import Dataset, DataLoader
import lightning as L
import argparse
from NBEATS import NBEATS_TS
import os
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from dotenv import load_dotenv
load_dotenv()
try :
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
except :
    pass

from utils.get_data_binance import getbinance
from utils.get_data_bitstamp import getbitstamp
from utils.get_data_full import get_data

from typing import Union, Iterable
from torch.utils.data import IterableDataset


Coinlist = ['ADA', 'ETH', 'LINK', 'LTC', 'XRP', 'BNB', 'DOGE', 'TRX']
df_all = pd.read_parquet('data/df_all.parquet')
data = df_all.values.reshape(-1, 6,3)
data = torch.tensor(data)
data = torch.permute(data, (2,1,0))


class Data_portfolio(Dataset):
    def __init__(self, flag, input_len, split):
        super().__init__()
        self.flag = flag
        self.input_len = input_len
        self.split = split
        self.__read_data__()

    def __read_data__(self):
        df_all = pd.read_parquet('./data/df_all.parquet').iloc[:,:]
        data = df_all.values.reshape(-1, 8,3)
        data = torch.tensor(data)
        data = torch.permute(data, (2,1,0))
        self.split_test = self.split + 1/2*(1-self.split)
        self.split = int(self.split * data.shape[-1])
        
        self.test_split = int(self.split_test*data.shape[-1])
        if self.flag == 'train':
            self.data = data[:,:,:self.split]
        elif self.flag == 'val':
            self.data = data[:,:,self.split:self.test_split]
        elif self.flag == 'test':
            self.data = data[:,:,self.test_split:]
    def __getitem__(self, index):  
        seq_x =self.data[:,:,index:index+self.input_len]
        norm = self.data[0,None, :, index+self.input_len-1, None].repeat((3,1,self.input_len))
        seq_x = seq_x
        seq_y = self.data[0,:,index+self.input_len] / self.data[0,:,index+self.input_len-1]
        return seq_x, seq_y
    
    def __len__(self):
        return self.data.shape[-1]-self.input_len-1

class DataModule_noval(L.LightningDataModule):
    def __init__(self, input_len, split, batch_size, size, start_date):
        self.split = split
        self.batch_size = batch_size
        self.input_len = input_len
        self.size = size
        self.start_date = start_date
        super().__init__()
        # self.save_hyperparameters()
    def setup(self, stage : str):
        if stage == 'fit' :
            self.dataset_train = Data_portfolio( flag = 'train', input_len=self.input_len, split = self.split)
            self.dataset_val = Data_portfolio( flag = 'val', input_len=self.input_len, split = self.split)
        if stage == 'test' :
            self.dataset_test = Data_portfolio(flag = 'test', input_len=self.input_len, split = self.split)
        
    def train_dataloader(self):
        return DataLoader(
        self.dataset_train,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = 3,
        drop_last = True)
    
    def val_dataloader(self):
        return DataLoader(
        self.dataset_val,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = 3,
        drop_last = True)
        
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last = True)

class LModel(L.LightningModule):
    def __init__(self, seq_len, batch_size):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        # self.model = Model(seq_len, batch_size)
        self.model = NBEATS_TS(h=2,input_size= seq_len)
        # var_init(self.model)
        self.c = 0.0025
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        
        loss = torch.mean(torch.abs((1+outputs)-batch_y))
        self.log('train_loss', loss, on_epoch = True)
        
        return loss 

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)

        loss = torch.mean(torch.abs((1+outputs)-batch_y))
        self.log('val_loss', loss)
        
        
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        mae = torch.mean(torch.abs((1+outputs)-batch_y))
        smape = torch.mean(2*torch.abs((1+outputs)-batch_y)/(torch.abs(outputs)+torch.abs(batch_y)))
        self.log('test_mae', mae)
        self.log('test_smape', smape)

    def forward(self, x):
        return self.model(x)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, maximize = False)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main file arguments')
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--size', type=int, default=10000)
    parser.add_argument('--start_date', type=str)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=500)


    args = parser.parse_args()
    size = args.size
    start_date = args.start_date
    seq_len = args.seq_len
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    logger =  MLFlowLogger(experiment_name = 'RL_proj', run_name = f'TS_run_naive', tracking_uri = os.getenv('MLFLOW_TRACKING_URI'), artifact_location = os.getenv('ARTIFACT_URI'))
    trainer = L.Trainer(devices = 1, accelerator = 'gpu',  max_epochs = n_epochs, log_every_n_steps=n_epochs//10, logger = logger, 
                        callbacks=[EarlyStopping('val_loss', patience=15, mode = 'min'), ModelCheckpoint(monitor = 'val_loss', save_top_k =1, mode = 'min')], 
                        enable_progress_bar=True)

    Lmodel = LModel(seq_len=seq_len, batch_size=batch_size)
    data = DataModule_noval(input_len = seq_len, split= 0.7, batch_size = batch_size, size=size, start_date=start_date)

    trainer.fit(Lmodel, data)
    trainer.test(Lmodel, data, ckpt_path = 'best')