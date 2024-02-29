import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import timm


def get_tow_imgs(df, idx):
    row = df.iloc[idx]
    img_1 = cv2.imread(f"data/{row.photo0}")
    img_2 = cv2.imread(f"data/{row.photo1}")
    
    return img_1, img_2

def concat_two_images(img_1, img_2):
    dim = (175, 175)
    img_1_ = cv2.resize(img_1, dim, interpolation = cv2.INTER_AREA)
    img_2_ = cv2.resize(img_2, dim, interpolation = cv2.INTER_AREA)
    
    concated_images = np.hstack([img_1_, img_2_])
    
    return concated_images

def get_concated_img_from_df(df, idx):
    img_1, img_2 = get_tow_imgs(df, idx)
    return concat_two_images(img_1, img_2)

def get_model_n_processor(base_model_name, num_classes, device="cpu"):
    model = timm.create_model(
        base_model_name,
        pretrained=True,
        num_classes=num_classes,  # remove classifier nn.Linear
    )
    model.to(device)
    
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    processor = timm.data.create_transform(**data_config, is_training=False)

    print(f"model: {base_model_name}, loaded")
    
    return model, processor

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x