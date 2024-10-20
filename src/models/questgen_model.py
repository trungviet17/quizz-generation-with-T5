import lightning.pytorch as pl 
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim

import warnings 
import hydra 
import pyrootutils


class T5Finetuner(pl.LightningModule): 
    
    def __init__(self,model: T5ForConditionalGeneration, 
                 optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler ):
        """
        Khởi tạo att cần thiết để xây dựng mô hình bao bồm model, tokenizer, ... 
        """
        super(T5Finetuner, self).__init__()
        self.save_hyperparameters(logger=False)
    
    
    def forward(self, input_ids, attention_mask = None, labels = None): 
        """
        feed forward 
        """
        output = self.hparams.model(
            input_ids = input_ids, 
            attention_mask  = attention_mask, 
            labels = labels
        )
        
        return output.loss, output.logits
    

    def training_step(self, batch, batch_idx): 
        """
            training step 
        """
        loss, output = self(
            input_ids = batch["inp_ids"], 
            attention_mask = batch['inp_mask'], 
            labels= batch['labels']
        )

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
        
    
    def validation_step(self, batch, batch_idx): 
        """
        validation step 
        """
        loss, output = self(
            input_ids = batch["inp_ids"], 
            attention_mask = batch['inp_mask'], 
            labels= batch['labels']
        )

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        
        loss, output = self(
            input_ids = batch["inp_ids"], 
            attention_mask = batch['inp_mask'], 
            labels= batch['labels']
        )

        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
    

    def configure_optimizers(self): 
        """
        optimizer setup 
        """
        optimizer = self.hparams.optimizer(params = self.parameters())
        scheduler = self.hparams.scheduler(optimizer = optimizer)
        return optimizer,scheduler
    


if __name__ == '__main__': 

    warnings.filterwarnings("ignore")
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

    config_path = str(path / "configs" / "model")

    @hydra.main(version_base="1.3", config_path= config_path, config_name = "t5finetunner")
    def test_model(config: DictConfig):   

        model = hydra.utils.instantiate(config)

        print(model)
    
    test_model()