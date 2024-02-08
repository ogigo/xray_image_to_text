import pandas as pd
from sklearn.model_selection import train_test_split

images_captions_df=pd.read_csv("data_directory")

train_df, test_df = train_test_split(images_captions_df, test_size=0.2, shuffle=True, random_state=42)


