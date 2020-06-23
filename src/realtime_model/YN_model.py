import re
import pandas as pd
from tqdm.notebook import tqdm
import pickle
import os 
import json
from keras.models import model_from_json 
from numpy import dot
from numpy.linalg import norm
import numpy as np
import random
import argparse
import tensorflow as tf
from keras.models import *

parser = argparse.ArgumentParser(description='model show outputs')
parser.add_argument('--local_gloabal', default= None , help='is local or gloabal?')
parser.add_argument('--model', default= None , help='which model?')
parser.add_argument('--path', default=os.path.join("..","realtime_model"), help='no model path')
parser.add_argument('--item_id', default=None, type = int, help='no item_id')
parser.add_argument('--top', default=10, type = int, help='how many top list do you want?')
args = parser.parse_args()


def cosim_id(df, vec, item_id):
    def cos_sim(A, B):
           return dot(A, B)/(norm(A)*norm(B)) 
    new_vec = vec.copy() 
    sim = []
    
    # 인풋 호텔 정보 데이터에 없는 경우 종료 
    if item_id not in vec.index.tolist():
        return 
        
    for i in range(len(vec)):
        sim.append(cos_sim(vec.loc[args.item_id,:], vec.iloc[i,:]))

    new_vec['sim'] = sim
    # sim 높은 순 
    new_vec = new_vec['sim'].reset_index().sort_values('sim', ascending=False)
    sim_sorted = new_vec['locationId'].tolist()
    # 인풋 호텔정보 빼고 유사도 높은 순대로 id 
    if item_id in sim_sorted:
        sim_sorted.remove(args.item_id) 
    return sim_sorted 


def sim_item(vec, df, item_id, top):
    top_id = cosim_id(df, vec, args.item_id)
    
    if type(top_id) == list :
        df = df.drop_duplicates(['locationId'], keep='last')
        recommend_rst = []
        for x in top_id:
            if df.loc[df['locationId']==x].category.values[0]== 'EAT':
                recommend_rst.append([df.loc[df['locationId']==x][['place.name', 'land.addr']]])

        print('input hotel:', df.loc[df['locationId']==item_id]['place.name'].unique()[0])
        print('-'*10)
        for i in range(len(recommend_rst[:top])):
            print('top', i+1, recommend_rst[i][0]['place.name'].values[0])
            print('  주소', recommend_rst[i][0]['land.addr'].values[0])
        
    else:
        answer_lst = ['해당 호텔 정보가 없습니다. 다른 호텔을 입력해주세요.', '해당 호텔 정보가 없습니다. 다른 호텔을 추천받아보세요.']
        x = random.randint(0, len(answer_lst)-1)
        return answer_lst[x]
    
            

if __name__ == '__main__':

    df = pd.read_csv(os.path.join(args.path,(args.local_gloabal+'_df.csv')))
    vec = pd.read_csv(os.path.join(args.path,(args.model+'_'+args.local_gloabal+'_vec.csv')))
    vec.index = vec['locationId']
    vec = vec.drop(columns = ['locationId'], axis=1)

    print(sim_item(vec, df, args.item_id, args.top))
    