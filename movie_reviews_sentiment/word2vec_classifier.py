# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd

from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from word2vec_model import load_dataset
from word2vec_model import clean_text

def review2vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[word] for word in words if word in model])
    return pd.Series(array.mean(axis=0))


#  Step 1: 读入之前训练好的word2vec模型
model_name = '300features_40minwords_10context.model'
print('Loading model ...')
model = Word2Vec.load(os.path.join('.', 'models', model_name))

# Step 2: 对有标记的训练数据的reviews影评做编码（对句中所有的词向量求平均）
df = load_dataset('labeled_train')
# print(df.head())
train_data_features = df['review'].apply(review2vector)
# print(train_data_features.head())

# Step 3: 用随机森林构建分类器
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(train_data_features, df['sentiment'])

# 在训练集上测试，确保模型能正常work
confusion_matrix(df['sentiment'], forest.predict(train_data_features))

# Step 4: 清理占用内存的变量
del(df)
del(train_data_features)

df = load_dataset('test')
# print(df.head())
test_data_features = df['review'].apply(review2vector)
# print(test_data_features.head())

result = forest.predict(test_data_features)
output = pd.DataFrame({'id': df.id, 'sentiment': result})
output.to_csv(os.path.join('.', 'submission_word2vec.csv'), index=False)
# print(output.head())

del(df)
del(test_data_features)
del(forest)
