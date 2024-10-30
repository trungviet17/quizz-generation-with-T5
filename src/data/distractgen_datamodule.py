import lightning.pytorch as pl 
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer 
import pandas as pd 
from datasets import load_dataset
from tqdm import tqdm 
from src.data.components.distractgen_dataset import DistractDataset
import os

class DistractDatamodule(pl.LightningDataModule): 

    def __init__(self, train_dir: str, val_dir: str, test_dir: str, tokenizer: T5Tokenizer,  
                 max_len_inp: int = 512, max_len_out: int = 96, batch_size: int = 8, num_workers: int = 4):

        super(DistractDatamodule, self).__init__() 
        self.save_hyperparameters(logger=False)


    def prepare_data(self):
        """Download data if needed"""
        
        os.makedirs(os.path.dirname(self.hparams.train_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.hparams.val_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.hparams.test_dir), exist_ok=True)

        dataset = load_dataset("race", "middle")
        train_df = self.create_dataset(dataset['train'])
        val_df = self.create_dataset(dataset['validation'])
        test_df = self.create_dataset(dataset['test'])

        train_df.to_csv(self.hparams.train_dir, index=False)
        val_df.to_csv(self.hparams.val_dir, index=False)
        test_df.to_csv(self.hparams.test_dir, index=False) 

    

    def setup(self, stage: str = None):
        train_df = pd.read_csv(self.hparams.train_dir)
        val_df = pd.read_csv(self.hparams.val_dir)
        test_df = pd.read_csv(self.hparams.test_dir)

        if stage == 'fit' or stage is None:
            self.train_dataset = DistractDataset(train_df, self.hparams.tokenizer, self.hparams.max_len_inp, self.hparams.max_len_out)
            self.valid_dataset = DistractDataset(val_df, self.hparams.tokenizer, self.hparams.max_len_inp, self.hparams.max_len_out)

        if stage == 'test' or stage is None:
            self.test_dataset = DistractDataset(test_df, self.hparams.tokenizer, self.hparams.max_len_inp, self.hparams.max_len_out)
 


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                            batch_size = self.hparams.batch_size, 
                            shuffle = True, 
                            num_workers = self.hparams.num_workers)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                            batch_size = self.hparams.batch_size, 
                            shuffle = True, 
                            num_workers = self.hparams.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                            batch_size = self.hparams.batch_size, 
                            shuffle = True, 
                            num_workers = self.hparams.num_workers)


    def create_dataset(self, data_split): 
        data_row = []
        for i in tqdm(range(len(data_split))): 
            curr_context = data_split[i]['article']
            curr_question = data_split[i]['question']
            
            ans_idx = ord(data_split[i]['answer']) - ord('A')
            all_options = data_split[i]['options']
            curr_correct = all_options[ans_idx]
            all_options.pop(ans_idx)
            
            curr_incorrect1 = all_options[0]
            curr_incorrect2 = all_options[1]
            curr_incorrect3 = all_options[2]
            
            data_row.append({
                "context": curr_context, 
                "question": curr_question, 
                "correct": curr_correct, 
                "incorrect1": curr_incorrect1, 
                "incorrect2": curr_incorrect2, 
                "incorrect3": curr_incorrect3
            })
            
        return pd.DataFrame(data_row)
