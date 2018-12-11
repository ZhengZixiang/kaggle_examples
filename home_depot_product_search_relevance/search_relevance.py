import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import cross_val_score
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import code

# Step 1: 获取数据集
# 读入训练/测试集、和产品介绍
df_train = pd.read_csv('./dataset/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('./dataset/test.csv', encoding='ISO-8859-1')
df_desc = pd.read_csv('./dataset/product_descriptions.csv')

# 查看数据形式
# print(df_train.head())
# print(df_desc.head())

# 合并训练/测试集，以便于统一做进一步的文本预处理
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# print(df_all.head())
# print(df_all.shape)


# 产品介绍也是一个很有用的信息，我们直接left join
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

# Step 2: 文本预处理（词干提取）
stemmer = SnowballStemmer('english')


def str_stemmer(s):
    return ' '.join([stemmer.stem(word) for word in s.lower().split()])


# 接下来把每一个column都跑一遍，以清洁所有的文本内容
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))


# Step 3：自制文本特征
# 为了计算关键词的有效性，我们可以naive地看看出现了多少次
def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


# 关键词的长度
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
# 标题中有多少关键词重合
df_all['commons_in_title'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_title']), axis=1)
# 描述中有多少关键词重合
df_all['commons_in_desc'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_description']), axis=1)
# 等等等等features可以添加进来，之后我们把不能被机器学习模型处理的column给drop掉，这里因为文字计算机看不懂，所以drop掉
df_all = df_all.drop(['search_term', 'product_title', 'product_description'], axis=1)
# 把之前脱下的衣服再一件件穿回来

# Step 4: 重塑训练/测试集
# 分开训练集和测试集
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index+len(df_train)]
# print(df_train.index, '\t', df_test.index)

# 记录下测试集的id，留着上传的时候对得上号
test_ids = df_test['id']
# print(len(test_ids))

# 分离出y_train
y_train = df_train['relevance'].values

# 把训练集中的label给删去
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

# Step 5: 建立模型
# 随机森林模型，用交叉验证保证结果公正客观性，并调试不同的alpha值
params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

# 画个图来看看
plt.plot(params, test_scores)
plt.title('Params vs CV Error')
# 画图可知大概6-7时达到最优解
plt.show()

# Step 6: 上传结果
rf = RandomForestRegressor(n_estimators=30, max_depth=7)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
# print(len(y_pred), len(test_ids))

# 把拿到的结果，放进DataFrame，做成CSV上传
pd.DataFrame({'id': test_ids, 'relevance': y_pred}).to_csv('submission.csv', index=False)

# 可以尝试修改、调试、升级的部分
# 1. 文本预处理步骤：让数据变得更加清洁
# 2. 自制特征：更多的特征值表示方式，如tf-idf、word2vec、关键词全段重合数量、重合率等等
# 3. 更好的回归模型：Ensemble方法？
