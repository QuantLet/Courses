CUDA_LAUNCH_BLOCKING = 1.
import pandas as pd
import seaborn as sns
sns.set_theme('notebook', 'whitegrid')
import matplotlib.pyplot as plt
# Relevant imports 
import pandas as pd
import numpy as np
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import Dataset, DataLoader
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import lightning as L
import mlflow
import argparse
from NBEATS import NBEATS
from model import CNN
from torch import nn
import os
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from dotenv import load_dotenv
load_dotenv()
try : 
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
except:
    pass

from typing import Union, Iterable
from torch.utils.data import IterableDataset


Coinlist = ['ADA', 'ETH', 'LINK', 'LTC', 'XRP', 'BNB', 'DOGE', 'TRX']
df_all = pd.read_parquet('data/df_all.parquet')
data = df_all.values.reshape(-1, 8,3)
data = torch.tensor(data)
data = torch.permute(data, (2,1,0))

def var_init(model, std=0.5):
    for name, param in model.named_parameters():
        param.data.normal_(mean=0.0, std=0.5)

class Data_portfolio(Dataset):
    def __init__(self, flag, input_len, split, samples, start):
        super().__init__()
        self.flag = flag
        self.input_len = input_len
        self.split = split
        self.samples = samples
        self.start = start
        self.__read_data__()

    def __read_data__(self):
        df_all = pd.read_parquet('./data/df_all.parquet').iloc[self.start:self.start+self.samples,:]
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
        seq_y = self.data[0,:,index+self.input_len]/self.data[0,:,index+self.input_len-1]
        seq_y = torch.cat((torch.tensor([1]), seq_y), dim = 0)
        return seq_x, seq_y
    
    def __len__(self):
        return self.data.shape[-1]-self.input_len-1
    
class Dataset(IterableDataset):
    def __init__(self, buffer: 'Buffer', source: str) -> None:
        self.buffer = buffer
        self.source = source

    def __iter__(self) -> Iterable:
        return self

    def __next__(self):
        return self.buffer.next_batch(self.source)
    def __len__(self):
        return 1000


class Buffer(nn.Module):
    def __init__(self,
                 coin_features: torch.tensor,
                 batch_size=50,
                 window_size=50,
                 test_portion=0.15,
                 validation_portion=0.1,
                 sample_bias=0.0001,
                 portion_reversed=False,
                 device="cpu"
                 ):
        super(Buffer, self).__init__()
        
        assert coin_features.ndim == 3
        coin_num = coin_features.shape[1]
        period_num = coin_features.shape[2]

        coin_features = torch.tensor(coin_features, device=device)

        # portfolio vector memory
        pvm = torch.full([period_num, coin_num], 1.0 / coin_num, device=device)
        self.register_buffer("_coin_features", coin_features, True)
        self.register_buffer("_pvm", pvm, True)

        self._batch_size = batch_size
        self._window_size = window_size
        self._sample_bias = sample_bias
        self._portion_reversed = portion_reversed
        self._train_idx, self._test_idx, self._val_idx = \
            self._divide_data(period_num, window_size, test_portion,
                              validation_portion, portion_reversed)

        # the count of appended experiences
        self._new_exp_count = 0

    @property
    def train_num(self):
        return len(self._train_idx)

    @property
    def test_num(self):
        return len(self._test_idx)

    @property
    def val_num(self):
        return len(self._val_idx)

    def get_train_set(self):
        return self._pack_samples(self._train_idx)

    def get_test_set(self):
        return self._pack_samples(self._test_idx)

    def get_val_set(self):
        return self._pack_samples(self._val_idx)

    def get_train_dataset(self):
        return Dataset(self, "train")

    def get_test_dataset(self):
        return Dataset(self, "test")

    def get_val_dataset(self):
        return Dataset(self, "val")

    def next_batch(self, source="train"):

        if source == "train":
            start_idx = self._train_idx[0]
            end_idx = self._train_idx[-1]
        elif source == "test":
            start_idx = self._test_idx[0]
            end_idx = self._test_idx[-1]
        elif source == "val":
            start_idx = self._val_idx[0]
            end_idx = self._val_idx[-1]

        batch_start = self._sample_geometric(
            start_idx, end_idx-self._batch_size, self._sample_bias
        )
        batch_idx = list(range(batch_start, batch_start + self._batch_size))
        batch = self._pack_samples(batch_idx)
        return batch

    def _pack_samples(self, index):
        index = np.array(index)
        last_w = self._pvm[index - 1, :]

        def setw(w):
            assert torch.is_tensor(w)
            self._pvm[index, :] = w.to(self._pvm.device).detach()

        batch = torch.stack([
            self._coin_features[:, :, idx:idx + self._window_size + 1]
            for idx in index
        ])
        # features, [batch, feature, coin, time]
        X = batch[:, :, :, :-1]
        norm = X[:, 0:1, :, -1:].repeat((1,3,1,self._window_size))
        X = X/norm
        # price relative vector of the last period, [batch, norm_feature, coin]
        y = batch[:, :, :, -1] / batch[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    @staticmethod
    def _sample_geometric(start, end, bias):
        """
        Generate a index within [start, end) with geometric probability.

        Args:
            bias: A value in (0, 1).
        """
        sample = np.random.geometric(bias)
        while end - sample < start :
            sample = np.random.geometric(bias)
        result = end-sample    
        # result = np.random.randint(start, end)
        return result

    @staticmethod
    def _divide_data(period_num,
                     window_size,
                     test_portion,
                     val_portion,
                     portion_reversed):
        """
        Divide training data into three portions, train, test and validation.

        Args:
            period_num: Number of price records in the time dimension.
            window_size: Sliding window size of history price records
            visible to the agent.
            test_portion/val_portion: Percent of these two portions.
            portion_reversed: Whether reverse the order of portions.

        Returns:
            Three np.ndarray type index arrays, train, test, validation.
        """
        train_portion = 1 - test_portion - val_portion
        indices = np.arange(period_num)

        if portion_reversed:
            split_point = np.array(
                [val_portion, val_portion + test_portion]
            )
            split_idx = (split_point * period_num).astype(int)
            val_idx, test_idx, train_idx = np.split(indices, split_idx)
        else:
            split_point = np.array(
                [train_portion, train_portion + test_portion]
            )
            split_idx = (split_point * period_num).astype(int)
            train_idx, test_idx, val_idx = np.split(indices, split_idx)

        # truncate records in the last time window, otherwise we may
        # sample insufficient samples when reaching the last window.
        train_idx = train_idx[:-(window_size + 1)]
        test_idx = test_idx[:-(window_size + 1)]
        val_idx = val_idx[:-(window_size + 1)]

        return train_idx, test_idx, val_idx


class DataModule_noval(L.LightningDataModule):
    def __init__(self, input_len, split, batch_size, samples, start_index):
        self.split = split
        self.batch_size = batch_size
        self.input_len = input_len
        self.samples = samples
        self.start = start_index
        super().__init__()
        # self.save_hyperparameters()
    def setup(self, stage : str):
        if stage == 'fit' :
            self.dataset_train = Data_portfolio( flag = 'train', input_len=self.input_len, split = self.split, samples = self.samples, start = self.start)
            self.dataset_val = Data_portfolio( flag = 'val', input_len=self.input_len, split = self.split, samples = self.samples, start = self.start)
        if stage == 'test' :
            self.dataset_test = Data_portfolio(flag = 'test', input_len=self.input_len, split = self.split, samples = self.samples, start = self.start)
        
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

def var_init(model, std=0.5):
        for name, param in model.named_parameters():
            param.data.normal_(mean=0.0, std=0.2)

class LModel(L.LightningModule):
    def __init__(self, seq_len, batch_size, model, training, loss):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.loss = loss
        # self.model = Model(seq_len, batch_size)
        if model == 'NBEATS':
            self.model = NBEATS(h=2,input_size= seq_len)
            # var_init(self.model)
        elif model == 'CNN':
            self.model = CNN(seq_len=seq_len, batch_size=batch_size, training=training)
            var_init(self.model)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        # ### Adding trade fee
        # w_prime = (batch_y * outputs)[:-1]
        # w_prime = w_prime/torch.sum(w_prime, dim = 1, keepdim=True)
        # mu = 1 - torch.sum(torch.abs(outputs[1:] - w_prime), dim=1) * self.c
        # mu = torch.cat([torch.tensor([1.0]).to('cuda:0'), mu])
        # loss = torch.mean(torch.log(mu*torch.sum(outputs*batch_y, dim = 1)))

        loss = torch.mean(torch.log(torch.sum(outputs*batch_y, dim = 1)))
        APV = torch.exp(torch.sum(torch.log(torch.sum(outputs*batch_y, dim = 1))))
        # APV = torch.exp(torch.sum(torch.log(mu*torch.sum(outputs*batch_y, dim = 1))))
        pv =  torch.sum(outputs * batch_y, dim = 1)
        sharpe_ratio = (torch.mean(pv)-1)/torch.std(pv)
        self.log('train_loss', loss, on_epoch = True)
        self.log('train_sharpe', sharpe_ratio)
        if self.loss == 'return' :
            return loss 
        else :
            return sharpe_ratio

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)

        loss = torch.mean(torch.log(torch.sum(outputs*batch_y, dim = 1)))
        APV = torch.exp(torch.sum(torch.log(torch.sum(outputs*batch_y, dim = 1))))
        # APV = torch.exp(torch.sum(torch.log(mu*torch.sum(outputs*batch_y, dim = 1))))
        pv =  torch.sum(outputs * batch_y, dim = 1)
        sharpe_ratio = (torch.mean(pv)-1)/torch.std(pv)
        if self.loss == 'return':
            self.log('val_loss', loss)
            self.log('val_sharpe', sharpe_ratio)
        else : 
            self.log('val_loss', sharpe_ratio)
            self.log('val_return', loss)

        self.log('val_APV', APV)
        
        
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        ### Adding trade fee
        # w_prime = (batch_y * outputs)[:-1]
        # w_prime = w_prime/torch.sum(w_prime, dim = 1, keepdim=True)
        # mu = 1 - torch.sum(torch.abs(outputs[1:] - w_prime), dim=1) * self.c
        # mu = torch.cat([torch.tensor([1.0]).to('cuda:0'), mu])

        # reward = torch.mean(torch.log(mu*torch.sum(outputs*batch_y, dim = 1)))
        reward = torch.mean(torch.log(torch.sum(outputs*batch_y, dim = 1)))
        APV = torch.exp(torch.sum(torch.log(torch.sum(outputs*batch_y, dim = 1))))
        # APV = torch.exp(torch.sum(torch.log(mu*torch.sum(outputs*batch_y, dim = 1))))
        pv =  torch.sum(outputs * batch_y, dim = 1)
        sharpe_ratio = (torch.mean(pv)-1)/torch.std(pv)
        self.log('test_reward', reward)
        self.log('test_APV', APV)
        self.log('test_sharpe', sharpe_ratio)
    def forward(self, x):
        return self.model(x)
    
    
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, maximize = True)
        return optimizer

class TraderTrainer(L.LightningModule):
    def __init__(self, data,  seq_len, batch_size, c, bias, model, training, loss):
        super().__init__()
        self.buffer = Buffer(coin_features = data,batch_size= batch_size,window_size=seq_len,test_portion=0.15,validation_portion=0.15, device='cuda:0', sample_bias= bias)
        if model == 'NBEATS':
            self.model = NBEATS(h=2,input_size= seq_len)
            self.var_init(self.model)
        elif model == 'CNN':
            self.model = CNN(seq_len, batch_size, training)
            self.var_init(self.model)

        self.loss = loss
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.c = c
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(dataset=self.buffer.get_train_dataset(),
                          collate_fn=lambda x: x)

    def training_step(self, batch, _batch_idx):
        batch = batch[0]
        y = batch['y']
        new_w = self.model(batch["X"], batch["last_w"])
        batch["setw"](new_w[:, 1:])
        y = torch.cat([torch.ones([y.shape[0], 1], device=y.device),
                      y[:, 0, :]], dim=1)
        w_prime = y * new_w
        w_prime = w_prime / torch.sum(w_prime, dim=1).unsqueeze(-1)
        w_t = w_prime[:-1]
        w_t1 = new_w[1:]
        mu = 1 - torch.sum(torch.abs(w_t1 - w_t), dim=1) * self.c
        pv_vector = torch.sum(new_w * y, dim=1)*(torch.cat([torch.ones([1], device=mu.device), mu], dim=0))
        log_mean = torch.mean(torch.log(pv_vector))
        self.log('loss', log_mean, on_epoch = True)
        sharpe_ratio = (torch.mean(pv_vector)-1)/torch.std(pv_vector)
        if self.loss == 'return':
            return log_mean
        else:
            return sharpe_ratio
    
    def val_dataloader(self):
        return DataLoader(dataset=self.buffer.get_val_dataset(),
                          collate_fn=lambda x: x)

    def validation_step(self,batch, batch_idx):
        batch = batch[0]
        y = batch['y']
        new_w = self.model(batch["X"], batch["last_w"])
       
        batch["setw"](new_w[:, 1:])
        y = torch.cat([torch.ones([y.shape[0], 1], device=y.device),
                      y[:, 0, :]], dim=1)
        w_prime = y * new_w
        w_prime = w_prime / torch.sum(w_prime, dim=1).unsqueeze(-1)
        c = 0.0025
        w_t = w_prime[:-1]
        w_t1 = new_w[1:]
        mu = 1 - torch.sum(torch.abs(w_t1 - w_t), dim=1) * self.c
        pv_vector = torch.sum(new_w * y, dim=1)*(torch.cat([torch.ones([1], device=mu.device), mu], dim=0))
        log_mean = torch.mean(torch.log(pv_vector))
        pf_val = torch.prod(pv_vector)
        sharpe_ratio = (torch.mean(pv_vector)-1)/torch.std(pv_vector)
        self.log('val_APV', pf_val)
        
        if self.loss == 'return':
            self.log('val_loss', log_mean)
            self.log('val_sharpe', sharpe_ratio)
        else : 
            self.log('val_loss', sharpe_ratio)
            self.log('val_return', log_mean)

    def test_dataloader(self):
        return DataLoader(dataset=self.buffer.get_test_dataset(),
                          collate_fn=lambda x: x)
    
    def test_step(self,batch, batch_idx):
        batch = batch[0]
        y = batch['y']
        new_w = self.model(batch["X"], batch["last_w"])
       
        batch["setw"](new_w[:, 1:])
        y = torch.cat([torch.ones([y.shape[0], 1], device=y.device),
                      y[:, 0, :]], dim=1)
        w_prime = y * new_w
        w_prime = w_prime / torch.sum(w_prime, dim=1).unsqueeze(-1)
        w_t = w_prime[:-1]
        w_t1 = new_w[1:]
        mu = 1 - torch.sum(torch.abs(w_t1 - w_t), dim=1) * self.c
        pv_vector = torch.sum(new_w * y, dim=1)*(torch.cat([torch.ones([1], device=mu.device), mu], dim=0))
        log_mean = torch.mean(torch.log(pv_vector))
        pf_val = torch.prod(pv_vector)
        sharpe_ratio = (torch.mean(pv_vector)-1)/torch.std(pv_vector)
        dict_metrics = {
            'test_sharpe_ratio' : sharpe_ratio,
            'test_APV': pf_val,
        }
        self.log('test_loss', log_mean)
        self.log_dict(dict_metrics)
    
    
    def forward(self, x, last_w):
        return self.model(x, last_w)
    
    def var_init(model, std=0.5):
        for name, param in model.named_parameters():
            param.data.normal_(mean=0.0, std=0.2)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, maximize=True)
        return optimizer
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main file arguments')
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--size', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--training', type = str, choices=['buffer' ,'mapping'], default='mapping')
    parser.add_argument('--c', type =float,  default=0.0025)
    parser.add_argument('--samples', type =int,  default=15000)
    parser.add_argument('--bias', type =float,  default=0.0003)
    parser.add_argument('--model', type =str, choices = ['NBEATS', 'CNN'], default='NBEATS')
    parser.add_argument('--logger', type =int,  default=1)
    parser.add_argument('--loss', type =str,  default='return')
    parser.add_argument('--start_index', type =int,  default=0)


    close = pd.read_csv('20250528 Request Data/data_close.csv')
    args = parser.parse_args()
    size = args.size
    seq_len = args.seq_len
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    if args.logger == 1 :
        logger =  MLFlowLogger(experiment_name = 'RL_proj', run_name = f'RL_run_{args.training}_{args.model}', tracking_uri = os.getenv('MLFLOW_TRACKING_URI'), artifact_location = os.getenv('ARTIFACT_URI'))
        trainer = L.Trainer(devices = 1, accelerator = 'gpu',  max_epochs = n_epochs, log_every_n_steps=n_epochs//10, logger = logger, 
                        callbacks=[EarlyStopping('val_loss', patience=15, mode = 'max'), ModelCheckpoint(monitor = 'val_loss', save_top_k =1, mode = 'max')], 
                        enable_progress_bar=True)
        
    else:
        trainer = L.Trainer(devices = 1, accelerator = 'gpu',  max_epochs = n_epochs, log_every_n_steps=n_epochs//10,
                        callbacks=[EarlyStopping('val_loss', patience=10, mode = 'max'), ModelCheckpoint(dirpath = f'{pd.to_datetime(close.Time).dt.strftime("%Y-%m-%d")[args.start_index]}_RL_run_{args.training}_{args.model}',filename='best_model', save_top_k=1, mode='max')], 
                        enable_progress_bar=True)
    if args.training == 'mapping' : 
        Lmodel = LModel(seq_len=seq_len, batch_size=batch_size, model = args.model, training = args.training, loss = args.loss)
        data = DataModule_noval(input_len = seq_len, split= 0.7, batch_size = batch_size, samples=args.samples, start_index=args.start_index)
        trainer.fit(Lmodel, data)
        trainer.test(Lmodel, data, ckpt_path = 'best')

    elif args.training == 'buffer' :
        df_all = pd.read_parquet('./data/df_all.parquet').iloc[args.start_index:args.start_index + args.samples,:]
        data = df_all.values.reshape(-1, 8,3)
        data = torch.tensor(data)
        data = torch.permute(data, (2,1,0))

        Lmodel = TraderTrainer(data, seq_len=args.seq_len, batch_size=args.batch_size, c = args.c, bias= args.bias, model = args.model, training = args.training, loss = args.loss)
        trainer.fit(Lmodel)
        trainer.test(Lmodel, ckpt_path = 'best')
