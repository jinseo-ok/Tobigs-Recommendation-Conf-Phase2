from train import train
import argparse
from preprocess import get_modified_data
import pandas as pd 
import os
import config
import tensorflow as tf
from keras.models import load_model

parser = argparse.ArgumentParser(description='afm')
parser.add_argument('--visited_ids', default= None, help='') 
parser.add_argument('--top_n', default= 10, help='')
args = parser.parse_args()


if __name__ == '__main__':
    # 모델 불러오기 
    model = load_model(os.path.join("..","..","data",".h5"))

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
    df = df.drop(columns = ['visit_idx'], axis=1)
    top_ids = df.head(args.top_n)['locationId'] 
    top_names = locationinfo.loc[locationinfo['locationdId'].isin(top_ids)]['place.name'].tolist()
    
    print('recommendation restaurant top',args.top_n,'list')
    print('-'*30)
    for i in range(len(top_names)):
        print('top', i+1, top_names[i])
    print('-'*30)