"""S1 verl dataset using EXACTLY the S3 expert tokenizer/mask convention."""
import json
from pathlib import Path

from torch.utils.data import Dataset

from verifiable_opd.posttraining.text_encode import encode_expert


class TextExpertSFTDataset(Dataset):
    def __init__(self,parquet_files,tokenizer,config,processor=None,max_samples=-1):
        if processor is not None:
            raise ValueError('Text-only SFT does not accept multimodal processors')
        paths=[parquet_files] if isinstance(parquet_files,str) else list(parquet_files)
        if len(paths)!=1 or not str(paths[0]).endswith('.jsonl'):
            raise ValueError('Use the one frozen expert_train.jsonl file')
        if config.pad_mode!='no_padding':
            raise ValueError('S1 adapter expects engine nested no_padding input')
        self.rows=[json.loads(s) for s in Path(paths[0]).read_text().split('\n') if s.strip()]
        if max_samples>0:self.rows=self.rows[:max_samples]
        self.encoded=[]
        for row in self.rows:
            if row['split']!='train' or row['kind']!='expert':
                raise ValueError('Only frozen train expert supervision permitted')
            encoded=encode_expert(tokenizer,row['messages'],row['expert_response'],2048,4096)
            if len(encoded.prompt_ids)+len(encoded.response_ids)>config.max_length:
                raise ValueError('Complete expert sequence exceeds S1 context; never silently truncate')
            self.encoded.append(encoded)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        import torch
        encoded=self.encoded[index]
        ids=encoded.prompt_ids+encoded.response_ids
        return {'input_ids':torch.tensor(ids,dtype=torch.long),
                'position_ids':torch.arange(len(ids),dtype=torch.long),
                'loss_mask':torch.tensor([0]*len(encoded.prompt_ids)+[1]*len(encoded.response_ids),dtype=torch.long)}
