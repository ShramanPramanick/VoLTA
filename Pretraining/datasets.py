import torch
import torch.nn as nn
import os
import copy
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from get_id_list import get_id_list_separate
from PIL import Image
from random import shuffle
from utils import *
import yaml


with open('./bt_got_mlm_itm_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class CCImageTextDataset(Dataset):

    def __init__(self, image_list, text_list, maxlen = 100, model_name = 'roberta-base', image_transform = CCImagePairTransform(train_transform = True), text_transform = CCTextPairTransform(train_transform = True)):

        self.maxlen = maxlen
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.image_list = image_list
        self.text_list = text_list
        self.image_transform = image_transform
        self.text_transform = text_transform


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #false_image_list = self.image_list
        #shuffle(false_image_list)

        image = Image.open(self.image_list[idx]).convert('RGB')
        image_sample = self.image_transform(image)

        #false_image = Image.open(false_image_list[idx]).convert('RGB')
        #false_image_sample = self.image_transform(false_image)

        #with open(self.text_list[idx], "r") as f:
        #    text_data = f.read()

        text_data = self.text_list[idx]
        aug_text_data = self.text_transform(text_data)

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(list(aug_text_data), 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
                        
        token_ids = encoded_pair['input_ids']  # tensor of token ids
        attn_masks = encoded_pair['attention_mask']  # binary tensor with "0" for padded values and "1" for the other values
        #token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if len(aug_text_data) == 2:
        
            rand_0 = torch.rand(token_ids[0].shape)
            mask_arr_0 = (rand_0 < config["mlm_prob"]) * (token_ids[0] != 0) * (token_ids[0] != 2) * (token_ids[0] != 1)
            selection_0 = torch.flatten((mask_arr_0).nonzero()).tolist()
            text_mlm_ids_0 = copy.deepcopy(token_ids[0])
            text_mlm_ids_0[selection_0] = 103
            text_mlm_labels_0 = torch.ones(token_ids[0].size(0), dtype=int)*-100
            text_mlm_labels_0[selection_0] = token_ids[0][selection_0]

            rand_1 = torch.rand(token_ids[1].shape)
            mask_arr_1 = (rand_1 < config["mlm_prob"]) * (token_ids[1] != 0) * (token_ids[1] != 2) * (token_ids[1] != 1)
            selection_1 = torch.flatten((mask_arr_1).nonzero()).tolist()
            text_mlm_ids_1 = copy.deepcopy(token_ids[1])
            text_mlm_ids_1[selection_1] = 103
            text_mlm_labels_1 = torch.ones(token_ids[1].size(0), dtype=int)*-100
            text_mlm_labels_1[selection_1] = token_ids[1][selection_1]

        else:

            rand = torch.rand(token_ids.shape)
            mask_arr = (rand < config["mlm_prob"]) * (token_ids != 0) * (token_ids != 2) * (token_ids != 1)
            selection = torch.flatten((mask_arr).nonzero()).tolist()
            text_mlm_ids = copy.deepcopy(token_ids)
            text_mlm_ids[selection] = 103
            text_mlm_labels = torch.ones(token_ids.size(0), dtype=int)*-100
            text_mlm_labels[selection] = token_ids[selection_0]

        

        if len(aug_text_data) == 2:
            return image_sample, (token_ids[0], token_ids[1]), (attn_masks[0], attn_masks[1]), (text_mlm_ids_0, text_mlm_ids_1), (text_mlm_labels_0, text_mlm_labels_1)
        else:
            return image_sample, token_ids, attn_masks, text_mlm_ids, text_mlm_labels
