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

parser = argparse.ArgumentParser(description='afm')
parser.add_argument('--visited_ids', default= None, help='') 
parser.add_argument('--top_n', type = int,  default= 10, help='')
parser.add_argument('--only_new', type='str',default= 'yes', help='')
args = parser.parse_args()


if __name__ == '__main__':

    df = pd.read_csv(os.path.join("..","..","data","locationsinfo_preds.csv"))

    # 예측 rating , 평균 평점 높은 순 
    df = df.sort_values(by = ['rating','average_rating'], ascending=False)
    
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

    # top_n 
    top_ids = df.head(args.top_n)['locationId'] 
    top_names = df.loc[df['locationdId'].isin(top_ids)]['place.name'].tolist()
    
    print('recommendation restaurant top',args.top_n,'list')
    print('-'*30)
    for i in range(len(top_names)):
        print('top', i+1, top_names[i])
    print('-'*30)