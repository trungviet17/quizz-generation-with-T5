import lightning.pytorch as pl 
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
import torch 
import warnings 
import hydra 
import pyrootutils


class T5Finetuner(pl.LightningModule): 
    
    def __init__(self,model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, 
                 optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler ):
        """
        Khởi tạo att cần thiết để xây dựng mô hình bao bồm model, tokenizer, ... 
        Sử dụng hparams cho chỉ sử dụng để lưu tham số 
        """
        super(T5Finetuner, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens("<sep>")
        self.tokenizer.add_tokens("<mask>")
        self.model.resize_token_embeddings(len(tokenizer))
        
        peft_config = LoraConfig(
            task_type = TaskType.SEQ_2_SEQ_LM, 
            inference_mode= False, 
            r = 8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # bleu score 
        self.bleu_4_score = BLEUScore(n_gram = 4)
        self.bleu_5_score = BLEUScore(n_gram = 5)

        # rouge score
        self.rouge_score = ROUGEScore(rouge_keys= ("rouge1", "rouge2", "rougeL"), use_stemmer=False)

        self.save_hyperparameters(logger=False, ignore = ['model', 'tokenizer'])
    
    def frozen_model(self): 
        """
        freeze model 
        """
        for param in self.model.parameters(): 
            param.requires_grad = False

        for param in list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()): 
            param.requires_grad = True


    def compute_bleu(self, output_ids, target_ids): 
        """
        compute bleu 
        """
        target_ids = torch.where(target_ids == -100, self.tokenizer.pad_token_id, target_ids)
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        target_text = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        return self.bleu_4_score(output_text, target_text), self.bleu_5_score(output_text, target_text)


    def compute_rouge(self, output_ids, target_ids):
        """
        compute rouge 
        """
        target_ids = torch.where(target_ids == -100, self.tokenizer.pad_token_id, target_ids)
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        target_text = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        return self.rouge_score(output_text, target_text)

            
    def forward(self, input_ids, attention_mask = None, labels = None): 
        """
        feed forward 
        """
        output = self.model(
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
            input_ids = batch["input_ids"], 
            attention_mask = batch['attention_mask'], 
            labels= batch['label']
        )

        output_ids = self.model.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        bleu_4, _ = self.compute_bleu(output_ids, batch['label'])

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_bleu_4', bleu_4, prog_bar=True, logger=True)
    
        return loss
        
    
    def validation_step(self, batch, batch_idx): 
        """
        validation step 
        """
        loss, output = self(
            input_ids = batch["input_ids"], 
            attention_mask = batch['attention_mask'], 
            labels= batch['label']
        )

        output_ids = self.model.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        bleu_4, bleu_5 = self.compute_bleu(output_ids, batch['label'])
        rouge = self.compute_rouge(output_ids, batch['label'])

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('train_bleu_4', bleu_4, prog_bar=True, logger=True)
        self.log('train_bleu_5', bleu_5, prog_bar=True, logger=True)
        self.log('train_rouge1', rouge['rouge1_fmeasure'], prog_bar=True, logger=True)
        self.log('train_rouge2', rouge['rouge2_fmeasure'], prog_bar=True, logger=True)
        self.log('train_rougeL', rouge['rougeL_fmeasure'], prog_bar=True, logger=True)
    

        return loss
    
    def test_step(self, batch, batch_idx):
        
        loss, output = self(
            input_ids = batch["input_ids"], 
            attention_mask = batch['attention_mask'], 
            labels= batch['label']
        )
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
    

    def configure_optimizers(self): 
        """
        optimizer setup 
        """
        optimizer = self.hparams.optimizer(params = self.parameters())
        lr_scheduler = self.hparams.scheduler(optimizer = optimizer)
        print(type(lr_scheduler))
        return {
          "optimizer" : optimizer, 
          "lr_scheduler" : lr_scheduler, 
          "monitor": "val_loss"
        }


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