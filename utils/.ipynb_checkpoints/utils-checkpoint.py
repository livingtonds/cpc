import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


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