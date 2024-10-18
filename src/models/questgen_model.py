import lightning.pytorch as pl 
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim


class T5Finetuner(pl.LightningModule): 
    
    def __init__(self,model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, 
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
        return AdamW(self.parameters(), lr = self.learning_rate)
    


if __name__ == '__main__': 

    def test_model(): 
        pass 