import numpy as np
import pandas as pd
from annoy import AnnoyIndex


class AnnoyDB():
    def __init__(self, f=1280, t=10):
        self.f = f
        self.t = t
        self.vectore_storage = AnnoyIndex(f, 'angular')
        self.idx_mapping_df = pd.DataFrame()
        
    def load(self, vectore_storage_path='test.ann', df_path="data.csv"):
        self.vectore_storage = AnnoyIndex(self.f, 'angular')
        self.vectore_storage.load(vectore_storage_path)
        self.idx_mapping_df = pd.read_csv(df_path)
        
    def get_data_by_vector(self, embeding, k=2):
        ni, cd = self.vectore_storage.get_nns_by_vector(embeding, k, include_distances=True)#[0]
        print(ni)
        print(cd)
        res_dict = self.idx_mapping_df.iloc[ni].to_dict(orient="list")
        
        return res_dict