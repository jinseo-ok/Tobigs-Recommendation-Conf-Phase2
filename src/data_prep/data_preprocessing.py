import re
import pandas as pd
import multiprocessing as mp
import numpy as np


class Preprocessor:
    def __init__(self):

    ### 이런식으로 제거할 함수 생성해서 넣으시면 됩니다! ###

    # 서울 소재지 place 데이터만 뽑기
    def seoul_place(self, df):
        seoul_idx = []

        for i in range(len(df['place.address'])):
            # 서울 소재지 place index 
            if df['place.address'][i].split(' ')[0] == '서울특별시':
        seoul_idx.append(i)

        return df.iloc[seoul_idx,:]




    # 위에서 만든 함수들 최종적 적용 후 전처리 완료 데이터 생성 
    def clean_data(self, df, options=['all', 'model_data'], document_type = 'dataframe', is_mulitprocess=False):
        if data_type == 'pickle':
            NAVER_PLACE_DATA_PATH = os.path.join("..","..","data","JS_05_reviews.pkl")
            df = pd.read_csv(NAVER_PLACE_DATA_PATH)

        # 서울 소재지 place 데이터
        df = self.seoul_place(df)

        # 전체 데이터 (모든 칼럼 그대로) 뽑을지 
        if 'all' in options:
            df = df
        # 모델에 태울 피쳐만 뽑은 데이터를 뽑을지 option 인자 설정 
        cols = ['rating','visitCount','id','idno','visitedDate.date','place.id','place.name']
        if 'model_data' in options:
            df = df.loc[:,cols]
        return df


if __name__ == "__main__":
    import time
    start_time = time.time()
    preprocessor = Preprocessor()

    NAVER_PLACE_DATA_PATH = os.path.join("..","..","data","JS_05_reviews.csv")
    df = pd.read_csv(NAVER_PLACE_DATA_PATH)

    # 생성한 함수 적용 
    df = preprocessor.clean_data(df, options=[''], is_mulitprocess=False)

    # print(df.head())
    print('elapsed time : {}'.format(time.time() - start_time))
    print('data preprocessed')