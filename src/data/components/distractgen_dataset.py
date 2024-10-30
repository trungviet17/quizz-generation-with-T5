from torch.utils.data import Dataset 
from transformers import T5Tokenizer
import pandas as pd 


class DistractDataset(Dataset): 

    def __init__(self, df: pd.DataFrame, tokenizer: T5Tokenizer, max_len_inp: int = 512, max_len_out: int = 96, 
                 sep_token: str = "<sep>"): 
        
        self.tokenizer = tokenizer
        self.max_len_inp = max_len_inp
        self.max_len_out = max_len_out
        self.SEP_TOKEN = sep_token
        self.tokenizer.add_tokens(sep_token)
        self.data = df 


    def __len__(self): 
        return len(self.data)


    def __getitem__(self, idx: int): 
        data_row = self.data.iloc[idx]

        input_encoding = self.tokenizer(
            '{} {} {} {} {}'.format(data_row['correct'], self.SEP_TOKEN, data_row['question'], self.SEP_TOKEN, data_row['context']),
            max_length= self.max_len_inp,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )


        output_encoding = self.tokenizer(
            '{} {} {} {} {}'.format(data_row['incorrect1'], self.SEP_TOKEN, data_row['incorrect2'], self.SEP_TOKEN, data_row['incorrect3']),
            max_length=self.max_len_out,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = output_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            answer_text = data_row['correct'],
            context = data_row['context'],
            question = data_row['question'],
            incorrect1 = data_row['incorrect1'],
            incorrect2 = data_row['incorrect2'],
            incorrect3 = data_row['incorrect3'],
            input_ids = input_encoding['input_ids'].flatten(),
            attention_mask = input_encoding['attention_mask'].flatten(),
            label=labels.flatten()
        )


if __name__ == '__main__': 
    pass 