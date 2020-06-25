# Preprocess
import config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# file = pd.read_csv(os.path.join("data","YN_afm_df.csv"))
# X = file.iloc[:, 0:9]
# Y = file.iloc[:, 9]

train = pd.read_csv(os.path.join("..","..","data","YN_afm_df2.csv"))
test = pd.read_csv(os.path.join("..","..","data","locationsinfo.csv"))
test = test.drop(columns=['place.name'], axis=1)
X_train = train.iloc[:, 0:7]
X_test = test.iloc[:, 0:7]
Y_train = train.iloc[:, 7] 

X_train.columns = config.ORIGINAL_FIELDS

def get_modified_data(X, continuous_fields, categorical_fields):

    X_cont = X[continuous_fields]
    X_cat = pd.DataFrame()

    scaler = MinMaxScaler()
    X_cont = pd.DataFrame(scaler.fit_transform(X_cont), columns=X_cont.columns)

    for col in categorical_fields:
        X_cat_col = pd.get_dummies(X[col], prefix=col, prefix_sep='-')
        X_cat = pd.concat([X_cat, X_cat_col], axis=1)

    X_modified = pd.concat([X_cont, X_cat], axis=1)
    num_feature = X_modified.shape[1]


    print('Data Prepared...') 
    print('X shape: {}'.format(X_modified.shape))
    print('num_feature',num_feature)
    # print(X_modified)

    return X_modified, num_feature

