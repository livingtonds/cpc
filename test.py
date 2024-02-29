import cv2
import torch

from utils.utils import *
from utils.embeder import ModelEmbed
from utils.storage import AnnoyDB
from utils.inference import InferenceWrapper, InferenceWrapperSimple


base_model_name = 'mobilenetv3_large_100.ra_in1k'
MODEL_PATH = '../model_state_dict.pt' #path to embeder/generation model
DEVICE = "cuda" # device cuda, cpu
F = 1280 # embeding size
VECTOR_STORAGE_PATH = 'data.ann' # path to build n saved annoy storage
DF_PATH = "data.csv" # path to csv with data
        
    
if __name__ == "__main__":
    img_1 = cv2.imread("img_1.png")
    img_2 = cv2.imread("img_2.png")

    model, processor = get_model_n_processor(base_model_name, 61656, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.classifier = Identity()

    # inference_wrapper = InferenceWrapper(
    #              model_path=MODEL_PATH,
    #              device=DEVICE,
    #              f=F,
    #              vectore_storage_path=VECTOR_STORAGE_PATH,
    #              df_path=DF_PATH
    # )

    inference_wrapper = InferenceWrapperSimple(
                 model=model,
                 processor=processor,
                 device=DEVICE,
                 f=F,
                 vectore_storage_path=VECTOR_STORAGE_PATH,
                 df_path=DF_PATH
    )

    result = inference_wrapper.get_res_by_two_images(img_1, img_2, k=2)
    print(result)
