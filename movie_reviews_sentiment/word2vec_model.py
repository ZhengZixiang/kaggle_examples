# -*- coding: utf-8 -*-
import os
import re
import nltk
import logging
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec


def load_dataset(name, num_rows=None):
    datasets = {
        'unlabeled_train': 'unlabeledTrainData.tsv',
        'labeled_train': 'labeledTrainData.tsv',
        'test': 'testData.tsv'
    }
    if name not in datasets:
        raise ValueError(name)
    datafile = os.path.join('.', 'dataset', datasets[name])
    df = pd.read_csv(datafile, sep='\t', escapechar='\\', nrows=num_rows)
    print('Number of reviews: {}'.format(len(df)))
    return df


# 和BoW版本一样的数据预处理，只是这里可以选择是否去除停用词
# 此外，别忘了喂给gensim的是一个个词列表，而不是一个个句子，所以不再需要重组句子这一步（在这里栽了半天）
def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    if remove_stopwords:
        text = [word for word in text if word not in stopwords.words('english')]  # word array
    return text


def print_call_counts(f):
    n = 0
    def wrapped(*args, **kwargs):
        nonlocal n
        n += 1
        if n % 1000 == 1:
            print('method {} called {} times'.format(f.__name__, n))
        return f(*args, **kwargs)
    return wrapped


@print_call_counts
def split_sentences(review):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences

if __name__ == '__main__':
    # Step 1: 读入无标签训练数据
    # 用于训练生成word2vec词向量
    df = load_dataset('unlabeled_train')
    # print(df.head())

    # Step 2: 数据预处理
    sentences = sum(df['review'].apply(split_sentences), [])
    print('{} reviews -> {} sentences'.format(len(df), len(sentences)))
    print(sentences)

    # Step 3: 用gensim训练词嵌入模型
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 设定词向量训练的参数
    num_features = 300      # word vector dimensionality
    min_word_count = 40     # minimun word count
    num_workers = 4         # number of threads to run in parallel
    context = 10            # context window size
    downsampling = 1e-3     # downsample setting for frequent words
    model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)

    print('Training model ...')
    print('sentences[0]')
    model = Word2Vec(sentences, workers=num_workers, size=num_features,
                     min_count=min_word_count, window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(os.path.join('.', 'models', model_name))

    # Step 4: 查看训练结果
    # 找出不相近的词
    print(model.doesnt_match('man woman child kitchen'.split()))
    print(model.doesnt_match('france england germany berlin'.split()))
    # 找出关联度最高的词
    print(model.most_similar('man'))
    print(model.most_similar('queen'))
    print(model.most_similar('awful'))
