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
parser.add_argument('--local_global', default= None , help='is local or gloabal?')
parser.add_argument('--model', default= None , help='which model?')
parser.add_argument('--path', default=os.path.join("..","realtime_model"), help='no model path')
parser.add_argument('--item_name', default=None, type = str, help='no item_name')
parser.add_argument('--top', default=10, type = int, help='how many top list do you want?')
args = parser.parse_args()


    
# "..",'..',"data","sim_data"

if __name__ == '__main__':

    with open(os.path.join(args.path,'{}_{}_{}_sim'.format(args.item_name, args.model, args.local_global)), 'rb') as f:
        data = pickle.load(f) 
    
    tops = data[:10]
    for i in range(len(tops)):
        print('top',i+1, tops[i][0].iloc[0,0])
        print('     ', tops[i][0].iloc[0,1])


    