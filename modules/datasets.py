import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.examples1={}
        temp=0
        z=0
        metadata = pd.read_csv('/data/zhuq3/mimic-cxr-2.0.0-metadata.csv',index_col=0)
        metadata.index.name = 'dicom_id'
        metadata = metadata[['StudyDate']]
        file1=open("mimic-context"+self.split+".json","a")
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            self.examples[i]['time']=metadata.loc[self.examples[i]["id"]]["StudyDate"]
        df=pd.DataFrame(self.examples)
        df.sort_values(['subject_id', 'time'], inplace=True)
        l=df.to_dict("records")
        temp=0
        z=0
        self.examples=l
        for i in range(len(self.examples)):

             if self.examples[i]["subject_id"]==temp:
                continue
             temp=self.examples[i]["subject_id"]
             m=1
             for j in range(len(self.examples)):
               if(self.examples[j]["subject_id"]==temp):
                  if(self.examples[i]["study_id"]!=self.examples[j]["study_id"]):
                    if(self.examples[j]["study_id"]!=self.examples[j-1]["study_id"]):
                            self.examples1[z]=self.examples[j]
                            self.examples1[z]["context"]=self.examples[j-1]["ids"]
                            self.examples1[z]["context_mask"]=self.examples[j-1]["mask"]
                            self.examples1[z]["context_image"]=self.examples[j-1]['image_path']
                            self.examples1[z]["context_report"]=self.examples[j-1]['report']
                            z=z+1
        json.dump(self.examples1,file1)

    def __len__(self):
        print(len(self.examples1))
        return len(self.examples1)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples1[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_path1=example["context_image"]
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image1 = Image.open(os.path.join(self.image_dir, image_path1[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            image_2 = self.transform(image1)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        context=example["context"]
        mask=example["context_mask"]
        context_len=len(context)
        sample = (image_id, image, report_ids, report_masks, seq_length,context,mask,context_len,image_2)
        return sample

