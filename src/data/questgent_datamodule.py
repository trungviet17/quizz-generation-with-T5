import lightning.pytorch as pl 
from torch.utils.data import DataLoader
from components.questgen_dataset import SQuADquestgen
from datasets import load_dataset


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

        



        return super().prepare_data()
    

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