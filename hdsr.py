import pandas as pd
import numpy as np

import re
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from gensim import matutils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from gensim.models import Word2Vec


def n_similarity(wv, ws1, ws2):
    v1 = [v for v in [wv[word] for word in ws1 if word in wv] if v is not None]
    v2 = [v for v in [wv[word] for word in ws2 if word in wv] if v is not None]

    if v1 and v2:
        return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))
    else:
        return 0


def cleanup(s):
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
    s = s.lower()

    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)

    s = s.replace(" x ", " xby ")
    s = s.replace("*", " xby ")
    s = s.replace(" by "," xby")
    s = s.replace("x0", " xby 0")
    s = s.replace("x1", " xby 1")
    s = s.replace("x2", " xby 2")
    s = s.replace("x3", " xby 3")
    s = s.replace("x4", " xby 4")
    s = s.replace("x5", " xby 5")
    s = s.replace("x6", " xby 6")
    s = s.replace("x7", " xby 7")
    s = s.replace("x8", " xby 8")
    s = s.replace("x9", " xby 9")
    s = s.replace("0x", "0 xby ")
    s = s.replace("1x", "1 xby ")
    s = s.replace("2x", "2 xby ")
    s = s.replace("3x", "3 xby ")
    s = s.replace("4x", "4 xby ")
    s = s.replace("5x", "5 xby ")
    s = s.replace("6x", "6 xby ")
    s = s.replace("7x", "7 xby ")
    s = s.replace("8x", "8 xby ")
    s = s.replace("9x", "9 xby ")

    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

    s = s.replace("whirpool", "whirlpool")
    s = s.replace("whirlpoolga", "whirlpool")
    s = s.replace("whirlpoolstainless","whirlpool stainless")

    s = s.replace("  ", " ")

    return s


class CleanupStemTokenizer:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')

    def tokenize(self, text):
        clean_text = cleanup(text).decode('utf8')
        words = [self.stemmer.stem(w) for w in clean_text.lower().split()]
        return words


class StopwordTokenizer:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))

    def tokenize(self, text):
        clean_text = re.sub('[^a-zA-Z]', ' ', text)
        words = clean_text.lower().split()
        words = [w for w in words if w not in self.stopwords]
        return words


class W2VTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer):
        self.wv = None
        self.tokenizer = tokenizer

    def transform(self, df):

        def similarity(left, right):
            return [n_similarity(self.wv, set(l), set(r)) for l, r in zip(left, right)]

        query_terms = df.search_term.map(self.tokenizer.tokenize)
        title_terms = df.product_title.map(self.tokenizer.tokenize)
        desc_terms = df.product_description.map(self.tokenizer.tokenize)

        title_sim = similarity(query_terms, title_terms)
        desc_sim = similarity(query_terms, desc_terms)

        res = pd.DataFrame({'title_sim': title_sim, 'desc_sim': desc_sim})
        return res

    def fit(self, df, y=None, **fit_params):
        sentences = df.product_title.map(self.tokenizer.tokenize)
        sentences.append(df.product_description.map(self.tokenizer.tokenize))

        self.wv = Word2Vec(sentences, min_count=3, workers=4)
        return self


def same_terms_count(left, right):
    return [len(set(l).intersection(set(r))) for l, r in zip(left, right)]


class QueryMatchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = CleanupStemTokenizer()

    def transform(self, df):
        res = pd.DataFrame()

        query_terms = df.search_term.map(self.tokenizer.tokenize)
        title_terms = df.product_title.map(self.tokenizer.tokenize)
        desc_terms = df.product_description.map(self.tokenizer.tokenize)
        brand_terms = df.brand.map(self.tokenizer.tokenize)

        res['query_len'] = query_terms.map(len)
        res['title_len'] = title_terms.map(len)
        res['desc_len'] = desc_terms.map(len)
        res['brand_len'] = brand_terms.map(len)

        res['query_words_in_title'] = same_terms_count(query_terms, title_terms)
        res['query_words_in_desc'] = same_terms_count(query_terms, desc_terms)
        res['query_words_in_brand'] = same_terms_count(query_terms, brand_terms)

        cleaned_query = df.search_term.map(cleanup)
        cleaned_title = df.product_title.map(cleanup)
        cleaned_desc = df.product_description.map(cleanup)

        res['query_in_title'] = [l.count(r) for l, r in zip(cleaned_title, cleaned_query)]
        res['query_in_desc'] = [l.count(r) for l, r in zip(cleaned_desc, cleaned_query)]

        res['ratio_title'] = res['query_words_in_title']/res['query_len']
        res['ratio_description'] = res['query_words_in_desc']/res['query_len']
        res['ratio_brand'] = res['query_words_in_brand']/res['query_len']

        return res

    def fit(self, df, y=None, **fit_params):
        return self


class QueryMatchAttrTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = CleanupStemTokenizer()

    def transform(self, df):
        res = pd.DataFrame()

        query_terms = df.search_term.map(self.tokenizer.tokenize)
        attr_terms = df.attrs.map(self.tokenizer.tokenize)

        res['query_len'] = query_terms.map(len)
        res['attr_len'] = attr_terms.map(len)

        res['query_words_in_attr'] = same_terms_count(query_terms, attr_terms)

        cleaned_query = df.search_term.map(cleanup)
        cleaned_attrs = df.product_description.map(cleanup)

        res['query_in_attr'] = [l.count(r) for l, r in zip(cleaned_attrs, cleaned_query)]

        res['ratio_attr'] = res['query_words_in_attr']/res['query_len']

        return res

    def fit(self, df, y=None, **fit_params):
        return self


def vectorizer_sum(vectorizer, query, field):
    q_vecs = vectorizer.transform(query)
    f_vecs = vectorizer.transform(field) > 0

    return q_vecs.multiply(f_vecs).sum(axis=1).A1


class QueryMatchScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = CleanupStemTokenizer()
        self.title_vec = None
        self.desc_vec = None
        self.brand_vec = None

    def transform(self, df):
        res = pd.DataFrame()

        query_len = df.search_term.map(self.tokenizer.tokenize).map(len)

        res['query_title_tfidf'] = vectorizer_sum(self.title_vec, df.search_term, df.product_title)
        res['query_desc_tfidf'] = vectorizer_sum(self.desc_vec, df.search_term, df.product_description)
        res['query_brand_tfidf'] = vectorizer_sum(self.brand_vec, df.search_term, df.brand)

        res['tfidf_ratio_title'] = res['query_title_tfidf']/query_len
        res['tfidf_ratio_description'] = res['query_desc_tfidf']/query_len
        res['tfidf_ratio_brand'] = res['query_brand_tfidf']/query_len

        return res

    def fit(self, df, y=None, **fit_params):
        self.title_vec = TfidfVectorizer(analyzer=self.tokenizer.tokenize).fit(df.product_title)
        self.desc_vec = TfidfVectorizer(analyzer=self.tokenizer.tokenize).fit(df.product_description)
        self.brand_vec = TfidfVectorizer(analyzer=self.tokenizer.tokenize).fit(df.brand)
        return self


def rmse(est, features, labels):
    pred = est.predict(features)
    return mean_squared_error(labels, pred)**.5
