from utils.utils import *
from utils.embeder import ModelEmbed
from utils.storage import AnnoyDB


class InferenceWrapper():
    def __init__(self,
                 model_path='blip_model',
                 device="cuda",
                 f=768,
                 vectore_storage_path='test.ann',
                 df_path="train_df.csv"):
        self.model_embd = ModelEmbed(model_path=model_path,
                                     device=device)
        
        self.annoy_db = AnnoyDB(f=f)
        self.annoy_db.load(vectore_storage_path=vectore_storage_path,
                           df_path=df_path)
        
    def get_res_by_two_images(self, img_1, img_2, k=2):
        responce = {
            "generated_caption": "",
            "res_dict_from_bd": {}
        }
        concated_images = concat_two_images(img_1, img_2)
        embeding, generated_caption = self.model_embd.get_embd_n_text(concated_images)
        res_dict_from_bd = self.annoy_db.get_data_by_vector(embeding, k=k)
        
        responce["generated_caption"] = generated_caption
        responce["res_dict_from_bd"] = res_dict_from_bd
        
        return responce
        
        