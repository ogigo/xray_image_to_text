import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from inference import feature_extractor,tokenizer
from PIL import Image
max_length=512

images_captions_df=pd.read_csv("data_directory")

class LoadDataset(Dataset):
    def __init__(self, df):
        self.images = images_captions_df['imgs'].values
        self.captions = images_captions_df['captions'].values

    def __getitem__(self, idx):
        # everything to return is stored inside this dict
        inputs = dict()

        # load the image and apply feature_extractor
        image_path = str(self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = feature_extractor(images=image, return_tensors='pt')

        # load the caption and apply tokenizer
        caption = self.captions[idx]
        labels = tokenizer(
            caption,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )['input_ids'][0]

        # store the inputs and labels in the dict we created
        inputs['pixel_values'] = image['pixel_values'].squeeze()
        inputs['labels'] = labels
        return inputs

    def __len__(self):
        return len(self.images)