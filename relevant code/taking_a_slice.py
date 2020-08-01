import pandas as pd
from sklearn.utils import shuffle


df = pd.read_csv('32.csv',encoding = 'utf-8')

df = shuffle(df)

df.reset_index(inplace = True) 

df = df.loc[1:500]

df.to_csv('taking_slices.csv', encoding='utf-8')