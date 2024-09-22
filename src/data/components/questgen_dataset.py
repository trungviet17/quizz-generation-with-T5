from torch.utils.data import Dataset
from transformers import  T5Tokenizer
import  tqdm
import copy 
import pandas as pd 


class SQuADquestgen(Dataset): 
    
    def __init__(self, tokenizer: T5Tokenizer,  file_path: str, max_len_inp: int = 512, max_len_out: int = 96): 
        """
        Khởi tạo các thông tin liên quan tới bộ dữ liệu 
        Input: đường dẫn tới file dữ liệu , chiều dài tối đa của input đầu và và đầu ra 
        Ouput:  
        """
        self. path = file_path 
        self.max_len_inp = max_len_inp 
        self.max_len_out = max_len_out 
        self.tokenizer = tokenizer
        
        self.quest_cols, self.ans_cols, self.context_cols = "question", "answers", "context"
        self.data = pd.read_csv(self.path)
        
        self.input = []
        self.target = []
        self.skipped = 0 
        
        self._processing()
        
    
    def __len__(self): 
        # kich thuoc bo du lieu
        return len(self.input)
    
    def __getitem__(self, idx: int):
        
        # get ids token -> return tensor
        inp_ids= self.inputs[idx]["input_ids"].squeeze()
        tar_ids = self.target[idx]["input_ids"].squeeze()
        
        inp_mask = self.inputs[idx]['attendtion_mask'].squeeze()
        tar_mask = self.target[idx]['attendtion_mask'].squeeze()
        
        labels = copy.deepcopy(tar_ids)
        labels[labels == 0] = -100
        
        inp_decode = self.tokenizer.decode(inp_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True) 
        tar_decode = self.tokenizer.decode(tar_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True) 
        return {
            "inp_ids": inp_ids,  "tar_ids": tar_ids, 
            "inp_mask": inp_mask, "tar_mask": tar_mask, 
            "labels": labels, "input": inp_decode, "target": tar_decode
        }
    
    def _processing(self): 
        """
        Hàm tiền xử lý dữ liệu, chuẩn hóa dữ liệu theo từng hàng, từ dữ liệu đầu vào chuyển hóa thành các token  
        """
        for idx in tqdm(range(len(self.data))): 
            context, question, answer = self.data.loc[idx, self.context_cols], self.data.loc[idx, self.quest_cols], self.data.loc[idx, self.ans_cols]
            
            inp = f"context: {context} answer : {answer} </s>"
            target = f"question: {question} </s>"
            
            # kiem tra chieu dai cua input ]
            input_encoding_test = self.tokenizer.encode_plus(inp, truncation = False, return_tensors = 'pt')
            
            len_input_encoding = len(input_encoding_test["input_ids"][0])
            
            if len_input_encoding > self.max_len_out:
                self.skipped += 1 
                continue 
                
            # get inp token 
            inp_encoding = self.tokenizer.batch_encode_plus(
                [inp], max_length = self.max_len_inp, pad_to_max_length = True, return_tensors = "pt"
            
            )
            
            # get target token 
            target_encoding = self.tokenizer.batch_encode_plus(
                [target], max_length = self.max_len_out, pad_to_max_length = True, return_tensors = "pt"
            )
            
            self.input.append(inp_encoding)
            self.target.append(target_encoding)


if __name__ == "__main__": 

    def test(): 
        pass 


