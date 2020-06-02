import pandas as pd

def reset_index(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        result = result.reset_index(drop=True)
        return result
    return inner


@reset_index
def data_prep(df:pd.DataFrame) -> pd.DataFrame:
    return df.iloc[df.index>1, :]


def data_prep2(df):
    return df.iloc[df.index>1, :]


df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(df)
print('='*50)

print(data_prep(df))
print('='*50)
print(data_prep2(df))