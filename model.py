from transformers import AutoFeatureExtractor,AutoTokenizer,VisionEncoderDecoderModel
import torch

vit_model = VisionEncoderDecoderModel.from_pretrained("kajol/xray_to_text")

