from torch.utils.data import Dataset
from transformers import  T5Tokenizer
from tqdm import tqdm
import copy 
import pandas as pd 
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra 
import pyrootutils


class SQuADquestgen(Dataset): 
    
    def __init__(self, data_frame: pd.DataFrame,  tokenizer: T5Tokenizer, max_len_inp: int = 512, 
                 max_len_out: int = 96, sep_token: str = "<sep>", masking_chance: float = 0.2): 
        """
        Khởi tạo các thông tin liên quan tới bộ dữ liệu 
        Input: đường dẫn tới file dữ liệu , chiều dài tối đa của input đầu và và đầu ra 
        Ouput:  
        """
    
        self.max_len_inp = max_len_inp 
        self.max_len_out = max_len_out 
        self.tokenizer = tokenizer

        # update special token 
        self.tokenizer.add_tokens(sep_token) 

        self.SEP_TOKEN = sep_token


        self.data = data_frame 
        self._processing()
        
    
    def __len__(self): 
        # kich thuoc bo du lieu
        return len(self.data)
    
    def __getitem__(self, idx: int):
        
        # get ids token -> return tensor
        data_row = self.data.iloc[idx]

        answer = data_row['answer'] 
        input_encoding = self.tokenizer(
            '{}{}{}'.format(answer, self.SEP_TOKEN, data_row['context']),
            max_length=self.max_len_inp,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        output_encoding = self.tokenizer(
            '{}{}'.format(data_row['question'], self.SEP_TOKEN),
            max_length=self.max_len_out,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        label = output_encoding['input_ids']
        label[label == 0] = -100

        return dict(
            answer = data_row['answer'], 
            context = data_row['context'],
            question = data_row['question'], 
            input_ids = input_encoding['input_ids'].flatten(),
            attention_mask = input_encoding['attention_mask'].flatten(),
            label = label.flatten()
        )

    
    def _processing(self): 
        """
        Hàm tiền xử lý dữ liệu, chuẩn hóa dữ liệu theo từng hàng, từ dữ liệu đầu vào chuyển hóa thành các token  
        """
        context_name = 'context_para'
        drop_content = ['context_sent', 'answer_start', 'answer_end']

        self.data = self.data.dropna()
        self.data.rename(columns = {context_name: 'context', 'answer_text': 'answer'}, inplace = True)
        self.data.drop(columns= drop_content, inplace = True)


if __name__ == "__main__": 
    
    import warnings 
    warnings.filterwarnings("ignore")

    pyrootutils.setup_root(__file__, indicator = ".project-root", pythonpath = True)
    path = pyrootutils.find_root(search_from=__file__, indicator = '.project-root')

    config_path = str(path/ 'configs' / 'data')
    OmegaConf.register_new_resolver("root_path", lambda: str(path))
    # test dataset module 
    @hydra.main(version_base="1.3", config_path=config_path, config_name='squad_dataset')
    def test_dataset(config: DictConfig):  

        print("TEST DATASET")
        dataset: SQuADquestgen = hydra.utils.instantiate(config)

        print(f"Dataset size: {len(dataset)}")

        print(dataset[0])
        

        print("DATASET TESTING PASSED")

        
    test_dataset()  
 

