# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
from gensim import models,corpora
import re

def text_cleaner(text):
    text = text.map(lambda x: re.sub('[,\.!?]', '', x))
    text = text.map(lambda x: x.lower())
    return text

def load_data(body_loc,stance_loc):
    body = pd.read_csv(body_loc)
    stances = pd.read_csv(stance_loc)
    data = pd.merge(stances,body,on="Body ID")
    X = data[["Headline","articleBody"]]#.values.tolist()
    y = data["Stance"]#.values.tolist()
    return X,y

def LDA_feature(headline,body,num_topics=50,learning_method="online",num_words=10): # num_topics used =25/50 by athene
    headline = text_cleaner(headline)
    body = text_cleaner(body)
    print("Cleaned data")
    count_vectorizer_head = CountVectorizer(stop_words='english')
    count_data_head = count_vectorizer_head.fit_transform(headline)
    count_vectorizer_body = CountVectorizer(stop_words='english')
    count_data_body = count_vectorizer_body.fit_transform(headline)
    
    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print("Starting LDA analysis")    
    lda_head = LDA(n_components=num_topics, learning_method = learning_method, n_jobs=-1)
    lda_head_data = lda_head.fit_transform(count_data_head)
    print("Topics found via LDA:")
    print_topics(lda_head, count_vectorizer_head, num_words)
    
    lda_body = LDA(n_components=num_topics,learning_method = learning_method, n_jobs=-1)
    lda_body_data=lda_body.fit_transform(count_data_body)
    print("Topics found via LDA:")
    print_topics(lda_body, count_vectorizer_body, num_words)
    return [lda_head_data,lda_body_data]
    
def LSI_feature(headlines, bodies, n_topics=50, include_holdout=False, include_unlbled_test=False):
    
    def combine_and_tokenize_head_and_body(headlines, bodies):
        temp_list = []
        headlines = text_cleaner(headlines)
        bodies = text_cleaner(bodies)
        temp_list.extend(headlines)
        temp_list.extend(bodies)
        head_and_body_tokens = [nltk.word_tokenize(line) for line in temp_list]
        return head_and_body_tokens

    head_and_body = combine_and_tokenize_head_and_body(headlines, bodies)
    dictionary = corpora.Dictionary(head_and_body)
    corpus = [dictionary.doc2bow(text) for text in head_and_body]
    tfidf = models.TfidfModel(corpus)  # https://stackoverflow.com/questions/6287411/lsi-using-gensim-in-python
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)

    # get tfidf corpus of head and body
    corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines, bodies)]
    tfidf_train = models.TfidfModel(corpus_train)
    corpus_train_tfidf = tfidf_train[corpus_train]

    corpus_lsi = lsi[corpus_train_tfidf]

    X_head = []
    X_body = []
    i = 0
    for doc in corpus_lsi:
        if i < int(len(corpus_lsi) / 2):
            X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
            for id, prob in doc:
                X_head_vector_filled[id] = prob
            X_head.append(X_head_vector_filled)
        else:
            X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
            for id, prob in doc:
                X_body_vector_filled[id] = prob
            X_body.append(X_body_vector_filled)
        i += 1

    X = np.concatenate([X_head, X_body], axis=1)

    return X

def NMF_feature(headlines, bodies, n_topics=300, cosinus_dist=True):

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_all_data(head_and_body):
        vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
        X_all = vectorizer_all.fit_transform(head_and_body)
        vocab = vectorizer_all.vocabulary_
        print("NMF_topics: complete vocabulary length=" + str(len(list(vocab.keys()))))
        return X_all, vocab
        
    def get_vocab(head_and_body, filename):
        vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
        X_all = vectorizer_all.fit_transform(head_and_body)
        vocab = vectorizer_all.vocabulary_
        print("NMF_topics: complete vocabulary length=" + str(len(X_all[0])))
        return vocab
    
    head_and_body = combine_head_and_body(headlines, bodies)
    
    X_all, vocab = get_all_data(head_and_body)

    # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
    # more important topic words a body contains of a certain topic, the higher its value for this topic
    nfm = NMF(n_components=n_topics, random_state=1, alpha=.1)

    print("NMF_topics: fit and transform body")
    nfm.fit_transform(X_all)
    vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
    X_train_head = vectorizer_head.fit_transform(headlines)

    vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
    X_train_body = vectorizer_body.fit_transform(bodies)

    print("NMF_topics: transform head and body")
    # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
    # their vectors should be similar
    nfm_head_matrix = nfm.transform(X_train_head)
    nfm_body_matrix = nfm.transform(X_train_body)

    if cosinus_dist == False:
        return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)
    else:
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(nfm_head_matrix)):
            X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
            X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X


X,y = load_data("./fnc-1/train_bodies.csv","./fnc-1/train_stances.csv")
LDA_features = LDA_feature(X["Headline"],X["articleBody"])
LSI_features = LSI_feature(X["Headline"],X["articleBody"])
NMF_features = NMF_feature(X["Headline"],X["articleBody"])
