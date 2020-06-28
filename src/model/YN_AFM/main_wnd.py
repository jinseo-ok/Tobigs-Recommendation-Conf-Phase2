from train import train
import argparse
from preprocess import get_modified_data
import pandas as pd 
import os
import config
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
import json
import random
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='afm')
parser.add_argument('--visited_ids', default= None, help='') 
parser.add_argument('--top_n', type = int,  default= 10, help='')
parser.add_argument('--only_new', type='str',default= 'yes', help='')
parser.add_argument('--choosen_hotel_id', type=int,default= None, help='1차 모델 결과 중 선택한 호텔 id')
args = parser.parse_args()


def cosim_id(vec, item_id):
    def cos_sim(A, B):
           return dot(A, B)/(norm(A)*norm(B)) 
    new_vec = vec.copy() 
    sim = []
    
    # 인풋 호텔 정보 데이터에 없는 경우 종료 
    if item_id not in vec.index.tolist():
        return 
        
    for i in range(len(vec)):
        sim.append(cos_sim(vec.loc[item_id,:], vec.iloc[i,:]))

    new_vec['sim'] = sim
    # sim 높은 순 
    new_vec = new_vec['sim'].reset_index().sort_values('sim', ascending=False)
    sim_sorted_ids = new_vec['locationId'].tolist()
    return sim_sorted_ids


if __name__ == '__main__':
    df = pd.read_csv(os.path.join("..","..","data","locationsinfo_preds.csv"))

    # 예측 rating , 평균 평점 높은 순 
    df = df.sort_values(by = ['rating','average_rating'], ascending=False)
    
    # 가장 선호하는 곳과 비슷한 곳 순으로 추천 (wide and deep embedding vector 사용)
    vec = pd.read_csv(os.path.join("data",'wnd_all_vec2.csv'))
    vec.index = vec['locationId']
    vec = vec.drop(columns = ['locationId'], axis=1)
    sim_sorted_ids = cosim_id(vec, args.choosen_hotel_id) # 비슷한 아이템 순 id 
    df.index = df.locationId # index 재지정
    df = df.reindex(index = sim_sorted_ids) # 유사도 순으로 행 재정렬

    # 가본 곳 제외하고 추천 받을지 
    if args.only_new == 'yes':
        df['user_visit_history'] = 0
        df.loc[df['locationId'].isin(args.visited_ids)]['user_visit_history']*10
        visit_idx = df.loc[df['user_visit_history']!=0].index 
        df = df.drop(['visit_idx']) 
    else:
        df = df

    # acm 제외 
    df = df.loc[df['category_l']==0] 

    top_ids = df.head(args.top_n)['locationId'] 
    top_names = df.loc[df['locationdId'].isin(top_ids)]['place.name'].tolist()
    
    print('recommendation restaurant top',args.top_n,'list')
    print('-'*30)
    for i in range(len(top_names)):
        print('top', i+1, top_names[i])
    print('-'*30)