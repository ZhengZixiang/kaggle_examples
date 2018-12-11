# -*- coding: utf-8 -*-
import os
import re
import code
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords

import nltk
# nltk.download('stopwords')

# Step 1: 读入数据
datafile = os.path.join('.', 'dataset', 'labeledTrainData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
# print(df.head())


# Step 2: 数据预处理
def display(text, title):
    print(title)
    print('\n----------------------分割线----------------------\n')
    print(text)


# preprocessing example
raw_example = df['review'][0]
display(raw_example, '原始数据')
# 1. 去掉HTML标签
bs_example = BeautifulSoup(raw_example, 'html.parser').get_text()
display(bs_example, '去掉HTML标签的数据')
# 2. 移除标点
letters_example = re.sub(r'[^a-zA-Z]', ' ', bs_example)
display(letters_example, '移除标点的数据')
# 3. 切分成词/token
words = letters_example.lower().split()
display(words, '初步切分成词数据')
# 4. 去掉停用词
clean_words = [word for word in words if word not in stopwords.words('english')]
display(clean_words, '去掉停用词的数据')
# 5. 重组为新的句子
clean_example = ' '.join(clean_words)
display(clean_example, '重组成新句子的数据')


# 根据上面这个example，我们定义一个数据清洗流程
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words('english')]  # word array
    return ' '.join(text)


display(clean_text(raw_example), '调用函数')

# 清洗数据添加到dataframe中
df['clean_review'] = df['review'].apply(clean_text)
print(df.head(1))

# Step 3: 抽取bag of words词袋特征（用sklearn的CounterVectorizer）
vectorizer = CountVectorizer(max_features=5000)
train_data_features = vectorizer.fit_transform(df['clean_review']).toarray()
print(train_data_features.shape)

# Step 4: 训练分类器（RF不适合one-hot数据）
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, df['sentiment'])
# 在训练集上做predict看看效果
confusion_matrix(y_true=df['sentiment'], y_pred=forest.predict(train_data_features))

# Step 5: 预测
# 删除不用的占内容变量
del(df)
del(train_data_features)
# 读取测试数据进行预测
datafile = os.path.join('.', 'dataset', 'testData.tsv')
df = pd.read_csv(datafile, sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df['clean_review'] = df['review'].apply(clean_text)
print(df.head())

test_data_features = vectorizer.transform(df['clean_review']).toarray()
print(test_data_features.shape)

result = forest.predict(test_data_features)
output = pd.DataFrame({'id': df['id'], 'sentiment': result})
print(output.head())

output.to_csv(os.path.join('.', 'submission_bow.csv'), index=False)

del(df)
del(test_data_features)
