import torch
from transformers import AutoFeatureExtractor,AutoTokenizer
from model import vit_model
from PIL import Image

encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "gpt2"

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def get_text(image_file,model,tokenizer):
  # image = Image.open(image_file)
  image=image_file.convert("RGB")
  image = feature_extractor(images=image, return_tensors='pt')
  inputs=image["pixel_values"]
  model.eval()
  with torch.no_grad():
    out=model.generate(inputs,num_beams=4,max_length=30)
  decoded_out = tokenizer.decode(out[0], skip_special_tokens=True,max_length=512)
  return decoded_out
