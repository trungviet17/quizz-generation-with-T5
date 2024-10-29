import lightning.pytorch as pl 
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer
from datasets import load_dataset
import pandas as pd 
from src.data.components.questgen_dataset import SQuADquestgen

from omegaconf import DictConfig, OmegaConf
import hydra
import pyrootutils
from tqdm import tqdm


class QuestgenDatamodule(pl.LightningDataModule): 

    def __init__(self, train_dir : str, val_dir: str, tokenizer : T5Tokenizer,train_test_split: float = 0.2,
                max_len_inp: int = 512,max_len_out: int = 96, batch_size: int = 8, num_workers : int = 4, masking_chance: int = 0.2): 
        super(QuestgenDatamodule, self).__init__()
        self.save_hyperparameters(logger=False )



    def prepare_data(self) -> None:
        """Download du lieu neu can """
        pass 
    

    def setup(self, stage: str = None) -> None:
        train_df = pd.read_csv(self.hparams.train_dir)
        val_df = pd.read_csv(self.hparams.val_dir)

        train_size = int(len(train_df) *( 1 - self.hparams.train_test_split)) 
        test_df = train_df[train_size:]
        train_df = train_df[:train_size]
        if stage == 'fit' or stage is None:
            self.train_dataset = SQuADquestgen(train_df, self.hparams.tokenizer, self.hparams.max_len_inp, self.hparams.max_len_out)
            self.valid_dataset = SQuADquestgen(val_df, self.hparams.tokenizer, self.hparams.max_len_inp, self.hparams.max_len_out)
        if stage == 'test' or stage is None:
            self.test_dataset = SQuADquestgen(test_df, self.hparams.tokenizer, self.hparams.max_len_inp, self.hparams.max_len_out)
      
     


    def train_dataloader(self) -> DataLoader: 

        return DataLoader(self.train_dataset,
                          batch_size = self.hparams.batch_size, 
                          shuffle= True, 
                          num_workers= self.hparams.num_workers)
    

    def val_dataloader(self) -> DataLoader: 

        return DataLoader(
                self.valid_dataset,
                batch_size = self.hparams.batch_size, 
                shuffle = True, 
                num_workers = self.hparams.num_workers
            )
    
    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size = self.hparams.batch_size, 
                shuffle = True, 
                num_workers = self.hparams.num_workers
            )



if __name__== "__main__": 

    import warnings
    warnings.filterwarnings("ignore")


    pyrootutils.setup_root(__file__, indicator = ".project-root", pythonpath = True)
    path = pyrootutils.find_root(search_from=__file__, indicator = '.project-root')

    config_path = str(path/ 'configs' / 'data')
    output_path = str(path / 'output')
    OmegaConf.register_new_resolver("root_path", lambda: str(path))



    @hydra.main(version_base="1.3", config_path=config_path, config_name="squad")
    def test_datamodule(cofig: DictConfig):
        print("TEST DATAMODULE: ")
        tokenizer = T5Tokenizer.from_pretrained(cofig.tokenizer.pretrained_model_name_or_path)

        datamodule = QuestgenDatamodule(
            train_dir = str(path / cofig.train_dir),
            val_dir = str(path / cofig.val_dir),
            tokenizer = tokenizer,
            train_test_split = cofig.train_test_split,
            max_len_inp = cofig.max_len_inp,
            max_len_out = cofig.max_len_out
        )

        print("PASS INIT")

        datamodule.prepare_data()
        datamodule.setup() 
        
        print("PASS SETUP")

        train_loader = datamodule.train_dataloader()
        simple_batch = next(iter(train_loader))
        input_ = simple_batch[0]
        output = simple_batch[1]

        print(f"Shape of Input is {input_.shape}")
        print(f"Shape of output is {output.shape}")
        
        print(f"DATAMODULE PASSED")

    test_datamodule()