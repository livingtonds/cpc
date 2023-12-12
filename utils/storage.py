import numpy as np
import pandas as pd
from annoy import AnnoyIndex


class AnnoyDB():
    def __init__(self, f=1280, t=10):
        self.f = f
        self.t = t
        self.vectore_storage = AnnoyIndex(f, 'angular')
        self.df = pd.DataFrame()
        
    def build_from_vectors(self, vectors):
        for i in range(len(vectors)):
            self.storage.add_item(i, vectors[i])
            
    def build_from_df(self, df, model_embd):
        for i in range(len(df)):
            try:
                v = model_embd.get_embd(get_concated_img_from_df(df, i))
            except Exception:
                v = np.zeros(self.f)
            self.vectore_storage.add_item(i, v)
        self.df = df
            
    def build_n_save(self, path='test.ann'):
        self.vectore_storage.build(self.t)
        self.vectore_storage.save(path)
        
    def load(self, vectore_storage_path='test.ann', df_path="data.csv"):
        self.vectore_storage = AnnoyIndex(self.f, 'angular')
        self.vectore_storage.load(vectore_storage_path)
        self.df = pd.read_csv(df_path)
        
    def get_data_by_vector(self, embeding):
        ni = self.vectore_storage.get_nns_by_vector(embeding, 1)[0]
        res_dict = self.df.iloc[ni].to_dict()
        
        return res_dict