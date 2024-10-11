import lightning.pytorch as pl 
from torch.utils.data import DataLoader
from components.questgen_dataset import SQuADquestgen
from datasets import load_dataset
import pandas as pd 


class QuestgenDatamodule(pl.LightningDataModule): 

    def __init__(self, 
                train_dataset : SQuADquestgen, 
                valid_dataset : SQuADquestgen, 
                hpagrams): 
        super().init()
        self.hparams = hpagrams
        self.train_dataset = train_dataset
        self.valid_dataset= valid_dataset
        self.save_hyperparameters(hpagrams)


 
    def prepare_data(self) -> None:
        """Download du lieu neu can """
        train_set = load_dataset("squad", split = "train")
        valid_set = load_dataset("squad", split = "valid")

        def trans_to_df(df: pd.DataFrame, dataset): 

            for idx, row  in enumerate(dataset): 
                context = row['context']
                question = row['question']
                answer = row['answers']['text'][0]

                df.loc[idx] = [context] + [question] + [answer]
        
            return df 
        
        train_df = trans_to_df(
            pd.DataFrame(columns= ['context', "question", "answer"]), train_set
        )

        valid_df = trans_to_df(
            pd.DataFrame(columns= ['context', "question", "answer"]), valid_set
        )

        save_path = "./data"
        train_df.to_csv(save_path + "/train", index = False)
        valid_df.to_csv(save_path + "/valid", index = False)
    

    def setup(self): 
        pass 


    def train_dataloader(self) -> DataLoader: 

        return DataLoader(self.train_dataset,
                          batch_size = self.hparams.batch_size, 
                          shuffle= True, 
                          num_workers= self.hparams.num_workers)
    

    def valid_dataloader(self) -> DataLoader: 

        return DataLoader(
            self.valid_dataset,
            batch_size = self.hpagrams.batch_size, 
            shuffle = True, 
            num_workers = self.hpagrams.num_workers
        )
    

if __name__== "__main__": 


    def test(): 
        pass 