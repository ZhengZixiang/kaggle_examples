There are three baseline solutions for a completed Kaggle competition called [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial).

The **bow_classifier.py** provide a baseline with bag of words(BoW) model.

The **word2vec_model.py** builds a word2vec model using unlabeledTrainData.tsv.

The **word2vec_classifier.py** provide a baseline with word2vec model that transform the reviews to vectors by computing the mean of word vectors of a sentence.

Since computing the mean might be too simple and rough without considering the word vectors relationship, 
the **cluster_classfier.py** provide another baseline that clustered word vectors to different clusters and the reviews were represented by frequency of these clusters.