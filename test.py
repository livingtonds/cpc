import cv2
from utils.utils import *
from utils.embeder import ModelEmbed
from utils.storage import AnnoyDB
from utils.inference import InferenceWrapper


MODEL_PATH = 'blip_model' #path to embeder/generation model
DEVICE = "cuda" # device cuda, cpu
F = 768 # embeding size
VECTOR_STORAGE_PATH = 'data.ann' # path to build n saved annoy storage
DF_PATH = "data.csv" # path to csv with data
        
    
if __name__ == "__main__":
    img_1 = cv2.imread("img_1.png")
    img_2 = cv2.imread("img_2.png")
    
    inference_wrapper = InferenceWrapper(
                 model_path=MODEL_PATH,
                 device=DEVICE,
                 f=F,
                 vectore_storage_path=VECTOR_STORAGE_PATH,
                 df_path=DF_PATH
    )
    
    result = inference_wrapper.get_res_by_two_images(img_1, img_2, k=2)
    print(result)
