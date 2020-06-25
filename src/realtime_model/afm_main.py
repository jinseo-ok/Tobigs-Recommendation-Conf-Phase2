from afm.train import train
import argparse
from afm.preprocess import get_modified_data
import pandas as pd 
import os
from afm import config
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
import json

parser = argparse.ArgumentParser(description='afm')
parser.add_argument('--visited_ids', default= None, help='') 
parser.add_argument('--top_n', type = int,  default= 10, help='')
parser.add_argument('--only_new', type='str', default= 'yes', help='')
args = parser.parse_args()


if __name__ == '__main__':
    # 모델 불러오기 
    with open(os.path.join("..","YN_AFM", "weights" ,'weights-epoch(50)-batch(2560)-embedding(10)-hidden(64).json','r')) as f:
    model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights(os.path.join("..","YN_AFM", "weights", "weights-epoch(50)-batch(2560)-embedding(10)-hidden(64).h5"))

    locationinfo = pd.read_csv(os.path.join("..","..","data","locationsinfo.csv"))
    df = locationinfo.drop(columns=['place.name'], axis=1)
    df['user_visit_history'] = 0
    # 가본 호텔 또는 좋아하는 호텔에 가중치 부여 (user_visit_history)
    df.loc[df['locationId'].isin(args.visited_ids)]['user_visit_history']*10

    X_modified, num_feature = get_modified_data(df, config.CONT_FIELDS, config.CAT_FIELDS)
    
    test_x = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_modified.values, tf.float32))).shuffle(10000).batch(config.BATCH_SIZE)
    
    y_preds = [] 
    for x in test_x: 
        y_pred = model(test_x)
        y_preds.append(y_pred)
    df['y_pred'] = y_preds

    # 예측 rating , 평균 평점 높은 순 
    df = df.sort_values(by = ['y_pred','average_rating'], ascending=False)
    visit_idx = df.loc[df['user_visit_history']!=0].index 
    # 가본 곳 제외하고 추천 받을지 
    if args.only_new == 'yes':
        df = df.drop(columns = ['visit_idx'], axis=1) 
    else:
        df = df
    df = df.loc[df['category_l']==0]
    top_ids = df.head(args.top_n)['locationId'] 
    top_names = locationinfo.loc[locationinfo['locationdId'].isin(top_ids)]['place.name'].tolist()
    
    print('recommendation restaurant top',args.top_n,'list')
    print('-'*30)
    for i in range(len(top_names)):
        print('top', i+1, top_names[i])
    print('-'*30)