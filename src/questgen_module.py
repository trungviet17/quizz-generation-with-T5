import lightning.pytorch as pl 
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW



class T5Finetuner(pl.LightningModule): 
    
    def __init__(self, hparams): 
        """
        Khởi tạo att cần thiết để xây dựng mô hình bao bồm model, tokenizer, ... 
        """
        super(T5Finetuner, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrain("t5-base")
        self.tokenizer = T5Tokenizer.from_pretrain("t5-base")
        self.hpagrams = hparams
        self.save_hyperparameters(hparams)
    
    
    def forward(self, input_ids, attention_mask = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None): 
        """
        feed forward 
        """
        output = self.model(
            input_ids = input_ids, 
            attention_mask  = attention_mask, 
            decoder_input_ids = decoder_input_ids, 
            decoder_attention_mask = decoder_attention_mask,
            labels = labels
        )
        
        return output 

    
    def training_step(self, batch, batch_idx): 
        """
            training step 
        """
        output = self.forward(
            input_ids = batch["inp_ids"], 
            attention_mask = batch['inp_mask'], 
            decoder_input_ids = batch['tar_ids'], 
            decoder_attention_mask = batch['tar_mask'],
            labels= batch['labels']
        )
        
        loss = output[0]
        self.log("train_loss",loss)
        return loss
        
    
    def validation_step(self, batch, batch_idx): 
        """
        validation step 
        """
        output = self.forward(
            input_ids = batch["inp_ids"], 
            attention_mask = batch['inp_mask'], 
            decoder_input_ids = batch['tar_ids'], 
            decoder_attention_mask = batch['tar_mask'],
            labels= batch['labels']
        )
        
        loss = output[0]
        self.log("val_loss",loss)
        return loss
        
    
    
    def configure_optimizers(self): 
        """
        optimizer setup 
        """
        return AdamW(self.parameters(), lr = 0.001, eps = 1e-8)
    


if __name__ == '__main__': 

    def test(): 
        pass 