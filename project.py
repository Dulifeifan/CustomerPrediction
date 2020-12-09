import re
import string
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import lightgbm

import lightgbm as lgb


import pickle

from sklearn.metrics import roc_auc_score
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


train_data=pd.read_csv('./all/train_data.csv',sep=',')
train_label=pd.read_csv('./all/train_label.csv',sep=',')
test_data=pd.read_csv('./all/test_data.csv',sep=',')





"""fill NA
train_X11=train_data[train_data.columns[3]]
train_X12=train_data[train_data.columns[4]]
train_X13=pd.concat([train_X11, train_X12], axis=1)
#print(train_X13.isnull())
train_X14=train_X13.fillna(axis=1,method='ffill')
train_X1=train_X14[train_X14.columns[1]]"""

train_X11=train_data[train_data.columns[2]]
train_X12=train_data[train_data.columns[3]]
train_X13=train_data[train_data.columns[4]]
train_X10=pd.concat([train_X11,train_X12,train_X13],axis=1)
test_X11=test_data[test_data.columns[2]]
test_X12=test_data[test_data.columns[3]]
test_X13=test_data[test_data.columns[4]]


train_data['Quality'] = train_data['descrption'].str.extract('(quality)', expand=False)
#train_despo['qua'].fillna('normal',inplace = True)
test_data['Quality'] = test_data['descrption'].str.extract('(quality)', expand=False)
#test_despo['qua'].fillna('normal',inplace = True)
train_data['Durable'] = train_data['descrption'].str.extract('(durable)', expand=False)
test_data['Durable'] = test_data['descrption'].str.extract('(durable)', expand=False)
train_data['design'] = train_data['descrption'].str.extract('(design)', expand=False)
test_data['design'] = test_data['descrption'].str.extract('(design)', expand=False)


train_ext0=train_data[train_data.columns[8:]]
test_ext0=test_data[test_data.columns[8:]]
#print(train_data['descrption'])
#print(train_ext0[train_ext0.isnull().values==False])
train_ext=pd.get_dummies(train_ext0)
test_ext=pd.get_dummies(test_ext0)


namestrtrain=[]
for i in range(train_data.shape[0]):
    if(train_label.values[i][1]==1.0):
        namestrtrain.append(str(train_data.values[i][1]))
#print(namestrtrain)
namestrtrainneg=[]
for i in range(train_data.shape[0]):
    if(train_label.values[i][1]==0.0):
        namestrtrainneg.append(str(train_data.values[i][1]))


desstrtrain=[]
for i in range(train_data.shape[0]):
    if(train_label.values[i][1]==1.0):
        namestrtrain.append(str(train_data.values[i][5]))
#print(namestrtrain)
desstrtrainneg=[]
for i in range(train_data.shape[0]):
    if(train_label.values[i][1]==0.0):
        namestrtrainneg.append(str(train_data.values[i][5]))
"""bag of words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(namestrtrain)
#print(X_train_counts.shape)
#print(X_train_counts)
X_train_counts.todense()
print(count_vect.vocabulary_)"""



def keywords():
 dic={}
 corpus = np.array(namestrtrain)
 for item in corpus:
  item = rem(item)
  words = item.split(" ")
  for word in words:
   if word in dic:
    dic[word] = dic[word]+1
   else:
    dic[word] = 1
 del dic['']
 keys=[]
 for each in dic:
  if dic[each]>50:
   keys.append(each)
 # print(keys)
 return keys

def keywordsneg():
 dic={}
 corpus = np.array(namestrtrainneg)
 for item in corpus:
  item = rem(item)
  words = item.split(" ")
  for word in words:
   if word in dic:
    dic[word] = dic[word]+1
   else:
    dic[word] = 1
 del dic['']
 keys=[]
 for each in dic:
  if dic[each]>50:
   keys.append(each)
 # print(keys)
 return keys


def deskeywords():
 dic={}
 corpus = np.array(namestrtrain)
 for item in corpus:
  item = rem(item)
  words = item.split(" ")
  for word in words:
   if word in dic:
    dic[word] = dic[word]+1
   else:
    dic[word] = 1
 del dic['']
 keys=[]
 for each in dic:
  if dic[each]>100:
   keys.append(each)
 # print(keys)
 return keys

def deskeywordsneg():
 dic={}
 corpus = np.array(namestrtrainneg)
 for item in corpus:
  item = rem(item)
  words = item.split(" ")
  for word in words:
   if word in dic:
    dic[word] = dic[word]+1
   else:
    dic[word] = 1
 del dic['']
 keys=[]
 for each in dic:
  if dic[each]>100:
   keys.append(each)
 # print(keys)
 return keys



def rem(sen):
    sen = re.sub(r'&|:|<li>|</li>|ul>|</ul>|\+|\-|\/|\(|\)', '', sen)
    sen = re.sub(r'\s+\d\s+|\s+[a-zA-Z]\s+', '', sen)
    sen = re.sub(r'\s(For|for|With|with|and|to|in|of|Set|amp;)', ' ', sen)
    sen = re.sub(r'_|\.',' ',sen)
    return sen
#print(keywords())
#print(keywordsneg())

keywordsre=list(set(keywords()).difference(set(keywordsneg())))
keywordsreneg=list(set(keywordsneg()).difference(set(keywords())))

deskeywordsre=list(set(deskeywords()).difference(set(deskeywordsneg())))
deskeywordsreneg=list(set(deskeywordsneg()).difference(set(deskeywords())))





for mw in keywordsre:
    mw1='('+mw+')'
    train_data[mw] = train_data['name'].str.extract(mw1, expand=False)
    test_data[mw] = test_data['name'].str.extract(mw1, expand=False)
    # if (train_data[mw].empty| test_data[mw].empty):
    #     train_data.drop(columns=mw)
    #     test_data.drop(columns=mw)
train_naex0=train_data[train_data.columns[11:]]
train_naex=pd.get_dummies(train_naex0)
test_naex0=test_data[test_data.columns[11:]]
test_naex=pd.get_dummies(test_naex0)

naex0=pd.concat([train_naex0,test_naex0],axis=0)
naex=pd.get_dummies(naex0)
train_name=naex[0:train_X10.shape[0]]
test_name=naex[train_X10.shape[0]:]


print(len(keywordsre))
print(train_naex0.columns.size,test_naex0.columns.size)
print(train_naex.columns.size,test_naex.columns.size)
print(train_name.columns.size,test_name.columns.size)

#print(list(set(train_name.columns.values.tolist()).difference(set(test_naex0.columns.values.tolist()))))

for mwneg in keywordsreneg:
    mwneg1='('+mwneg+')'
    train_data[mwneg] = train_data['name'].str.extract(mwneg1, expand=False)
    test_data[mwneg] = test_data['name'].str.extract(mwneg1, expand=False)
    # if(train_data[mwneg].empty|test_data[mwneg].empty):
    #     train_data.drop(columns=mwneg)
    #     test_data.drop(columns=mwneg)
train_naex0neg=train_data[train_data.columns[11+len(keywordsre):]]
train_naexneg=pd.get_dummies(train_naex0neg)
test_naex0neg=test_data[test_data.columns[11+len(keywordsre):]]
test_naexneg=pd.get_dummies(test_naex0neg)


naex0neg=pd.concat([train_naex0neg,test_naex0neg],axis=0)
naexneg=pd.get_dummies(naex0neg)
train_nameneg=naexneg[0:train_X10.shape[0]]
test_nameneg=naexneg[train_X10.shape[0]:]




print(len(keywordsreneg))
print(train_naex0neg.columns.size,test_naex0neg.columns.size)
print(train_naexneg.columns.size,test_naexneg.columns.size)
print(train_nameneg.columns.size,test_nameneg.columns.size)


for desmw in deskeywordsre:
    desmw1='('+desmw+')'
    train_data['des'+desmw] = train_data['descrption'].str.extract(desmw1, expand=False)
    test_data['des'+desmw] = test_data['descrption'].str.extract(desmw1, expand=False)
train_desex0=train_data[train_data.columns[11+len(keywordsre)+len(keywordsreneg):]]
train_desex=pd.get_dummies(train_desex0)
test_desex0=test_data[test_data.columns[11+len(keywordsre)+len(keywordsreneg):]]
test_desex=pd.get_dummies(test_desex0)


for desmwneg in deskeywordsreneg:
    desmwneg1='('+desmwneg+')'
    train_data['des'+desmwneg] = train_data['descrption'].str.extract(desmwneg1, expand=False)
    test_data['des'+desmwneg] = test_data['descrption'].str.extract(desmwneg1, expand=False)
train_desex0neg=train_data[train_data.columns[11+len(keywordsre)+len(keywordsreneg)+len(deskeywordsre):]]
train_desexneg=pd.get_dummies(train_desex0neg)
test_desex0neg=test_data[test_data.columns[11+len(keywordsre)+len(keywordsreneg)+len(deskeywordsre):]]
test_desexneg=pd.get_dummies(test_desex0neg)





train_des0=train_data[train_data.columns[5]]
train_des1=train_data[train_data.columns[5]]
train_des2=train_data[train_data.columns[5]]
train_des3=train_data[train_data.columns[5]]

for i in range(train_des0.shape[0]):
    train_des0.values[i]=re.sub(r'\<[^>]*\>', ' ', str(train_des3.values[i]))
    blob = TextBlob(str(train_des0.values[i]))
    train_des1.values[i]=blob.sentiment.polarity
    #train_des2.values[i] = blob.sentiment.subjectivity



test_des0=test_data[test_data.columns[5]]
test_des1=test_data[test_data.columns[5]]
test_des2=test_data[test_data.columns[5]]
test_des3=test_data[test_data.columns[5]]
for i in range(test_des0.shape[0]):
    test_des0.values[i] = re.sub(r'\<[^>]*\>', ' ', str(test_des3.values[i]))
    blob = TextBlob(str(test_des0.values[i]))
    test_des1.values[i] = blob.sentiment.polarity
    #test_des2.values[i] = blob.sentiment.subjectivity
    #print(train_des1.values[i],test_des1.values[i])

train_despo=(train_des1 - train_des1.min()) / (train_des1.max() - train_des1.min())
test_despo=(test_des1 - test_des1.min()) / (test_des1.max() - test_des1.min())








test_X10=pd.concat([test_X11,test_X12,test_X13],axis=1)
datalvl0=pd.concat([train_X10,test_X10],axis=0)
datalvl1=pd.get_dummies(datalvl0)
train_X1=datalvl1[0:train_X10.shape[0]]
test_X1=datalvl1[train_X10.shape[0]:]

#train_X1=train_data[train_data.columns[4]]
train_X22=train_data[train_data.columns[6]]
train_X3=train_data[train_data.columns[7]]

train_y1=train_label[train_label.columns[1:]]


train_X2=(train_X22 - train_X22.min()) / (train_X22.max() - train_X22.min())
train_X3=pd.get_dummies(train_X3)
train_X = pd.concat([train_X1, train_X2,train_X3,train_despo,train_ext,train_name,train_nameneg], axis=1)
train_y=np.ravel(train_y1)

X_train,X_test, y_train, y_test = train_test_split(train_X,train_y,test_size=0.4, random_state=0)
rf1 = RandomForestRegressor(n_estimators= 70,max_depth=40,  min_samples_split=55,
                                 min_samples_leaf=50,max_features='auto',oob_score=True, random_state=100,n_jobs=-1)
rf1.fit(X_train,y_train)
y_pred=rf1.predict(X_test)
print(log_loss(y_test, y_pred))




"""fill NA
test_X11=test_data[test_data.columns[3]]
test_X12=test_data[test_data.columns[4]]
test_X13=pd.concat([test_X11, test_X12], axis=1)
#print(train_X13.isnull())
test_X14=test_X13.fillna(axis=1,method='ffill')
test_X1=test_X14[test_X14.columns[1]]"""





test_X22=test_data[test_data.columns[6]]
test_X3=test_data[test_data.columns[7]]


test_X2=(test_X22 - test_X22.min()) / (test_X22.max() - test_X22.min())
test_X3=pd.get_dummies(test_X3)
test_X4=test_data[test_data.columns[0]]
test_X = pd.concat([test_X1, test_X2,test_X3,test_despo,test_ext,test_name,test_nameneg], axis=1)


#print(list(set(train_X.columns.values.tolist()).difference(set(test_X.columns.values.tolist()))))
"""
rf = RandomForestRegressor(n_estimators= 70, max_depth=40, min_samples_split=55,
                                 min_samples_leaf=12,max_features='auto',oob_score=True, random_state=100)

#rf = RandomForestRegressor(n_estimators = 30, oob_score =True, n_jobs = -1,random_state =10,
#                                max_features = "auto", min_samples_leaf = 30)
rf.fit(train_X,train_y)



result=rf.predict(test_X)
"""

###########################
###########################
###########################
train_X=train_X.astype('float')
train_y1=train_y1.astype('float')
test_X=test_X.astype('float')
print(train_X[~train_X.applymap(np.isreal).all(1)])
print(train_y1[~train_y1.applymap(np.isreal).all(1)])
X, val_X, y, val_y = train_test_split(train_X,train_y1,test_size = 0.2)
#,random_state = 1,stratify = train_y1
lgbX_train = X
lgby_train = y
lgbX_test = val_X
lgby_test = val_y
lgb_train = lgb.Dataset(lgbX_train, lgby_train)
lgb_eval = lgb.Dataset(lgbX_test, lgby_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 100,
    'learning_rate': 0.02,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'metric': 'binary_logloss',
    'max_depth':50

}
gbm=lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=100)
preds = gbm.predict(test_X, num_iteration=gbm.best_iteration)

column_result = pd.Series(preds, name='score')

csv=pd.concat([test_X4,column_result],axis=1)

csv.to_csv("submission.csv",sep=',',index=0,float_format='%.2f')

###########################
###########################
###########################


#result1=(result - result .min()) / (result .max() - result .min())
"""
column_result = pd.Series(result, name='score')

csv=pd.concat([test_X4,column_result],axis=1)

csv.to_csv("submission.csv",sep=',',index=0,float_format='%.2f')"""