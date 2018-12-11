# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd

from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from word2vec_model import clean_text
from word2vec_model import load_dataset

# 在word2vec classifier中，我们对每个review的向量表示是简单粗暴地使用词向量求平均
# 这里我们考虑通过聚类来编码reviews
# Step 1: 载入word2vec模型
print('Loading model ...')
model_name = '300features_40minwords_10context.model'
model = Word2Vec.load(os.path.join('.', 'models', model_name))

# Step 2: 使用Kmeans进行聚类
print('Clustering ...')
word_vectors = model.wv.syn0  # w2v的词向量
num_clusters = word_vectors.shape[0] // 10  # 统计出w2v模型的词向量数，并除10得到聚类数
kmeans_clustering = KMeans(n_clusters=num_clusters, n_jobs=4)
idx = kmeans_clustering.fit_predict(word_vectors)  # 得到每个词向量对应的聚类

# 保存聚类模型
word_centroid_map = dict(zip(model.wv.index2word, idx))
filename = 'word_centroid_map_10avg.pickle'
with open(os.path.join('.', 'models', filename), 'bw') as f:
    pickle.dump(word_centroid_map, f)

# 输出前10个cluster看看效果
for cluster in range(0, 10):
    print('\ncluster %d' % cluster)
    print([w for w, c in word_centroid_map.items() if c == cluster])

# Step 3: 把reviews转换成cluster bag vectors
print('Transforming reviews to cluster bag vectors ...')
wordset = set(word_centroid_map.keys())


def make_cluster_bag(review):
    words = clean_text(review)
    return (pd.Series([word_centroid_map[w] for w in words if w in wordset])
                      .value_counts()
                      .reindex(range(num_clusters+1), fill_value=0))


df = load_dataset('labeled_train')
# print(df.head())
train_data_features = df['review'].apply(make_cluster_bag)
# print(train_data_features.head())

# Step 4: 随机森林建模
print('Modeling ...')
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(train_data_features, df['sentiment'])

# 在训练集上测试效果，确保模型能正常work
confusion_matrix(df['sentiment'], forest.predict(train_data_features))

# 删除无用的占内存的变量
del(df)
del(train_data_features)

# Step 5: 载入数据进行预测
print('Predict ...')
df = load_dataset('test')
# print(df.head())
test_data_features = df['review'].apply(make_cluster_bag)
# print(test_data_features.head())

result = forest.predict(test_data_features)
output = pd.DataFrame({'id':df['id'], 'sentiment':result})
output.to_csv(os.path.join('.', 'submission_cluster.csv'), index=False)
# print(output.head())

del(df)
del(test_data_features)
del(forest)
