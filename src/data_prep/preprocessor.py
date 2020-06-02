import re
import pandas as pd
import multiprocessing as mp
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='preprocessor')
# 전체 데이터 (모든 칼럼 그대로) 뽑을지, 모델에 태울 피쳐만 뽑은 데이터를 뽑을지 option 인자 설정 
parser.add_argument('--cols', default= None , help='')
parser.add_argument('--data_type', default='dataframe', help='')
args = parser.parse_args()


# 전처리 함수 생성 
def seoul_place(df):
        seoul_idx = []

        for i in range(len(df['place.address'])):
            # 서울 소재지 place index 
            if df['place.address'][i].split(' ')[0] == '서울특별시':
        seoul_idx.append(i)

        return df.iloc[seoul_idx,:]


### 이런식으로 전처리 함수 만드시면 됩니다! ###

# 최종 전처리 함수 적용 
def main():
    if args.data_type == 'pickle':
        NAVER_PLACE_DATA_PATH = os.path.join("..","..","data","JS_05_reviews.pkl")
        df = pd.read_pickle(NAVER_PLACE_DATA_PATH)

    NAVER_PLACE_DATA_PATH = os.path.join("..","..","data","JS_05_reviews.csv")
    df = pd.read_csv(NAVER_PLACE_DATA_PATH)

    print('==> Preparing data..')
    df = seoul_place(df) 

    # 전체 데이터 (모든 칼럼 그대로) 뽑을지, 모델에 태울 피쳐만 뽑은 데이터를 뽑을지 option 인자 설정 
    if args.cols == None:
        df = df
    else:
        df = df.loc[:,args.cols]

if __name__=='__main__':
    import time
    start_time = time.time()
    main()

    print('elapsed time : {}'.format(time.time() - start_time))
    print('data preprocessed')