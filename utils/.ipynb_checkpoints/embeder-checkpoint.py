import gc
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipModel


class ModelEmbed():
    def __init__(self, model_path='blip_model', device="cuda"):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        
    def get_embd(self, img):
        inputs = self.processor(images=Image.fromarray(img), return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        embding = self.model.vision_model(pixel_values)["pooler_output"].detach().cpu().numpy()[0]
        embding /= np.linalg.norm(embding)

        return embding
    
    def get_embd_n_text(self, img):
        inputs = self.processor(images=Image.fromarray(img), return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        embding = self.model.vision_model(pixel_values)["pooler_output"].detach().cpu().numpy()[0]
        embding /= np.linalg.norm(embding)
        
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=164)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return embding, generated_caption

class ModelEmbedSimple():
    def __init__(self, model, processor, device="cuda"):
        self.processor = processor
        self.model = model
        self.device = device
        
    def get_embd(self, img):
        inputs = Image.fromarray(img)
        embding = self.model(self.processor(inputs).unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]
        embding /= np.linalg.norm(embding)

        return embding