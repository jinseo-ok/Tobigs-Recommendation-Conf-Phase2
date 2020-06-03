import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import pickle
import os 


class cosim_item:
    def __init__(self, item_input = None, item_vocab_path = None, item_name_path = None):
        self.item_input = item_input
        # int 변환 임베딩 item idx 저장값 불러오기
        self.item_vocab = pickle.load(open(item_vocab_path,"rb"))
        # item name & item idx 매칭 저장값 불러오기
        self.item_name = pickle.load(open(item_name_path,"rb"))


    def max_cosine_item(self, latent_vector, is_local):
        if is_local == 'local':
            DATA_PATH = os.path.join("..","..","realtime_model","global_loss02.csv")
            latent_vector = pd.read_csv(DATA_PATH)
        elif is_local == 'global':
            DATA_PATH = os.path.join("..","..","realtime_model","local_loss02.csv")
            latent_vector = pd.read_csv(DATA_PATH)

        sim = 0
        max_i = 0
        if self.item_input:
            item_idx = self.item_vocab[self.item_input]
            for i in tqdm(range(latent_vector.shape[1])):
                if sim < cosine_similarity(latent_vector.iloc[:,item_idx], latent_vector.iloc[:,i]):
                    sim = cosine_similarity(latent_vector.iloc[:,item_idx], latent_vector.iloc[:,i])
                    max_i = i
            
        # 특정 열의 아이템 실제 id 가져오기.
        ## 수정 필요 사항 ##
        ## 1. top10. 뽑기
        ## 2. filter 받아 accomodate 제외
        similar_item_idx = [item_id for item_id, idx in self.item_vocab.items() if idx == max_i]
        similar_item = self.item_name[similar_item_idx]
        return similar_item


if __name__ == "__main__":
    import time
    start_time = time.time()
    item_input = 0 # 1차 추천 호텔 중 사용자가 선택한 호텔 입력 받기
    sim = cosim_item(item_input, 
                        item_vocab_path= os.path.join("..","..","realtime_model",'vocab_locationId_global.pickle'), 
                        item_name_path = os.path.join("..","..","data",'item_name.pickle'))
    
    DATA_PATH = os.path.join("..","..","realtime_model","global_loss02.csv")
    latent_vector = pd.read_csv(DATA_PATH)
    output_item = sim.max_cosine_item(latent_vector, 'global')

    print('elapsed time : {}'.format(time.time() - start_time))