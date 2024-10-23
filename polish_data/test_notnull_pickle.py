# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 01:54:54 2023

@author: xycha
"""



import numpy as np

import pandas as pd

rela_embedding=pd.read_csv('../polish data/new_Embedding_Model_POLISH_D128-rela.csv').drop('Unnamed: 0',axis=1)
tail_embedding=pd.read_csv('../polish data/tail_embedding_POLISH_D128.csv').drop('Unnamed: 0',axis=1)
x_test = pd.read_csv("../polish data/5year-Ts.csv")
polish_train_val=pd.read_csv("../polish data/5year-Tr.csv")
y_train_val=polish_train_val['LABEL']
y_test=x_test['LABEL']
x_test=x_test.drop(['LABEL','Company'],axis=1)
polish_train_val=polish_train_val.drop(['LABEL','Company'],axis=1)
x_trainval_mapping_df=pd.read_csv('../polish data/x_trainval_map_tailv.csv').drop('Unnamed: 0',axis=1)
x_test_mapping_df=pd.read_csv('../polish data/x_test_map_tailv.csv').drop('Unnamed: 0',axis=1)

#---------------------------------------- can be used in synthena unequal event lengths----------------------------------------
test_notnull=[]
for i in range(len(x_test_mapping_df)):
    total_test_reltal=[]
    for j in x_test_mapping_df.columns:
        rela=rela_embedding[rela_embedding.relation==j].drop(['relation'],axis=1).values.tolist()[0]
        tailv=tail_embedding[tail_embedding['tail']==x_test_mapping_df.loc[i,j]].drop(['tail'],axis=1).values.tolist()
        if tailv:
            total_test_reltal.append((rela+tailv[0]))
    test_notnull.append(total_test_reltal)
      
   
test_notnull_len=[]
for i in range(len(test_notnull)):
    test_notnull_len.append(len(test_notnull[i]))
#print(max(test_notnull_len),test_notnull_len.index(max(test_notnull_len)))

print(test_notnull[0])
#############-------------------- store test_notnull in a file and reload this file----------------------
import pickle
filename = 'test_notnull.pickle'

# open the file for writing in binary mode
with open(filename, 'wb') as f:
    # serialize the list and write it to the file
    pickle.dump(test_notnull, f)

# open the file for reading in binary mode
with open(filename, 'rb') as f:
    # deserialize the list and load it from the file
    loaded_test_notnull_lst = pickle.load(f)
#print(loaded_test_notnull_lst[0][0][0])

########-----------------------train_val_notnulllst

train_val_notnulllst=[]
for i in range(len(x_trainval_mapping_df)):
    total_trainval_reltal=[]
    for j in x_trainval_mapping_df.columns:
        rela=rela_embedding[rela_embedding.relation==j].drop(['relation'],axis=1).values.tolist()[0]
        tailv=tail_embedding[tail_embedding['tail']==x_trainval_mapping_df.loc[i,j]].drop(['tail'],axis=1).values.tolist()
        if tailv:
            total_trainval_reltal.append((rela+tailv[0]))
    train_val_notnulllst.append(total_trainval_reltal)
      
 
trainval_notnull_len=[]
for i in range(len(train_val_notnulllst)):
    trainval_notnull_len.append(len(train_val_notnulllst[i]))
 

# open the file for writing in binary mode
with open('train_val_notnull.pickle', 'wb') as f:
    # serialize the list and write it to the file
    pickle.dump(train_val_notnulllst, f)


from collections import Counter

trainval_notnull_count = Counter(trainval_notnull_len)
trainval_notnull_count


#%%
train_val_notnulllst=[]
for i in range(len(x_trainval_mapping_df)):
    total_trainval_reltal=[]
    for j in x_trainval_mapping_df.columns:
        #rela=rela_embedding[rela_embedding.relation==j].drop(['relation'],axis=1).values.tolist()[0]
        tailv=tail_embedding[tail_embedding['tail']==x_trainval_mapping_df.loc[i,j]].drop(['tail'],axis=1).values.tolist()
        if tailv:
            total_trainval_reltal.append((rela+tailv[0]))
    train_val_notnulllst.append(total_trainval_reltal)
#%%      
