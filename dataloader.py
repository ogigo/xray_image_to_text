import torch
from torch.utils.data import Dataset,DataLoader
from inference import feature_extractor,tokenizer
from PIL import Image
from dataset import train_df,test_df
max_length=512


class LoadDataset(Dataset):
    def __init__(self, df):
        self.df=df
        self.images = self.df['imgs'].values
        self.captions = self.df['captions'].values

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
    

train_ds = LoadDataset(train_df)
test_ds = LoadDataset(test_df)