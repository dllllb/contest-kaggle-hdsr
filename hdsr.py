# coding=utf-8
# https://www.kaggle.com/c/home-depot-product-search-relevance/
"""
Discussions:
https://www.youtube.com/watch?v=LJH6tnN0WDE
"""

from functools import partial

import pandas as pd
import numpy as np
import re
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from gensim import matutils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_union, make_pipeline
from gensim.models import Word2Vec
from sklearn.preprocessing import FunctionTransformer

from dstools.util.timer import Timer
from dstools.ml.xgboost_tools import XGBoostRegressor
from dstools.util.gcs_cache import gcs_cache


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


def dataset(query_file):
    q_df = pd.read_csv(query_file, encoding='ISO-8859-1', index_col='id')
    q_df['search_term'] = q_df.search_term.str.encode('utf-8')
    q_df['product_title'] = q_df.product_title.str.encode('utf-8')

    prod_df = pd.read_csv('product_descriptions.csv.gz', encoding='ISO-8859-1', index_col='product_uid')
    prod_df['product_description'] = prod_df.product_description.str.encode('utf-8')

    df_attr = pd.read_csv('attributes.csv.gz', encoding='ISO-8859-1').dropna()
    df_attr['product_uid'] = df_attr['product_uid'].astype(int)
    df_attr['name'] = df_attr.name.str.encode('utf-8')
    df_attr['value'] = df_attr.value.str.encode('utf-8')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]]\
        .rename(columns={"value": "brand"}).set_index("product_uid")

    gr_attr_df = df_attr[["product_uid", "value"]].groupby('product_uid')\
        .agg(lambda x: ' '.join(x)).rename(columns={"value": "attrs"})

    df = q_df.merge(prod_df, left_on='product_uid', right_index=True, how='left')
    df = df.merge(df_brand, left_on='product_uid', right_index=True, how='left')
    df = df.merge(gr_attr_df, left_on='product_uid', right_index=True, how='left')

    df = df.fillna('none')

    return df


def cv_test(est):
    with Timer('cv'):
        df = dataset('train.csv.gz')

        transf, estimator = est

        with Timer('feature_gen'):
            features = transf.fit_transform(df.drop('relevance', axis=1), df.relevance)

        scores = cross_val_score(
                estimator=estimator,
                X=features,
                y=df.relevance,
                cv=3,
                n_jobs=1,
                verbose=1,
                scoring=rmse)
        print('mean: {mean}, std: {std}'.format(mean=scores.mean(), std=scores.std()))


def submission(est, name='results'):
    with Timer('submission'):
        df = dataset('train.csv.gz')

        features = df.drop(['relevance'], axis=1)
        target = df.relevance

        transf, estimator = est
        pl = make_pipeline(transf, estimator)

        model = pl.fit(features, target)

        df_test = dataset('test.csv.gz')

        y_pred = model.predict(df_test)

        y_pred[y_pred < 1] = 1
        y_pred[y_pred > 3] = 3

        res = pd.Series(y_pred, index=df_test.index, name='relevance')
        res.to_csv(name+'.csv', index_label='id', header=True)


def column_transformer(name):
    return FunctionTransformer(partial(pd.DataFrame.__getitem__, key=name), validate=False)


def count_vec():
    return CountVectorizer(stop_words=stopwords.words("english"))

transf_count = make_union(
    make_pipeline(
        column_transformer('search_term'),
        count_vec()
    ),
    make_pipeline(
        column_transformer('product_title'),
        count_vec()
    ),
    make_pipeline(
        column_transformer('product_description'),
        count_vec()
    ),
)

transf_wv = W2VTransformer(tokenizer=StopwordTokenizer())

transf3 = make_union(transf_count, transf_wv)


def tfidf_vec():
    return TfidfVectorizer(stop_words=stopwords.words("english"))


def tsvd():
    return TruncatedSVD(n_components=10)

transf_tfidf = make_union(
    make_pipeline(
        column_transformer('search_term'),
        tfidf_vec(),
        tsvd(),
    ),
    make_pipeline(
        column_transformer('product_title'),
        tfidf_vec(),
        tsvd(),
    ),
    make_pipeline(
        column_transformer('product_description'),
        tfidf_vec(),
        tsvd(),
    ),
)

transf_qm = QueryMatchTransformer()

col2dict = FunctionTransformer(
    lambda x: pd.DataFrame(x).to_dict(orient='records'), validate=False)

transf_br = make_pipeline(
    column_transformer('brand'),
    col2dict,
    DictVectorizer(),
)

transf5 = make_union(transf_qm, transf_br)

transf_tfidf2 = make_union(
    make_pipeline(
        column_transformer('search_term'),
        tfidf_vec(),
        tsvd(),
    ),
    make_pipeline(
        column_transformer('product_title'),
        tfidf_vec(),
        tsvd(),
    ),
    make_pipeline(
        column_transformer('brand'),
        tfidf_vec(),
        tsvd(),
    ),
)

transf7 = make_union(
    transf_qm,
    transf_br,
    transf_tfidf2,
)

transf_qms = QueryMatchScoreTransformer()


def tfidf_vec2():
    return TfidfVectorizer(analyzer=CleanupStemTokenizer().tokenize)

transf_tfidf3 = make_union(
    make_pipeline(
        column_transformer('search_term'),
        tfidf_vec2(),
        tsvd(),
    ),
    make_pipeline(
        column_transformer('product_title'),
        tfidf_vec2(),
        tsvd(),
    ),
    make_pipeline(
        column_transformer('brand'),
        tfidf_vec2(),
        tsvd(),
    ),
)

transf10 = make_union(transf_qm, transf_qms, transf_wv)

transf_qma = QueryMatchAttrTransformer()

transf11 = make_union(transf_qm, transf_qma)

transf12 = make_union(
    transf_qm,
    transf_qma,
    transf_qms,
    transf_wv,
    transf_br,
    transf_tfidf3,
)

xgb_params = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "silent": 1,
    "max_depth": 6,
    "num_rounds": 10000,
    "num_es_rounds": 120,
    "es_share": .05,
}

est_xgb = XGBoostRegressor(**xgb_params)

xgb_params2 = {
    "objective": "reg:linear",
    "eta": 0.01,
    "min_child_weight": 6,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "silent": 1,
    "max_depth": 6,
    "num_rounds": 10000,
    "num_es_rounds": 120,
    "es_share": .05,
}

est_xgb2 = XGBoostRegressor(**xgb_params2)

est_rf = RandomForestRegressor(n_estimators=500, n_jobs=2, max_features=10, max_depth=20)

# mean: 0.524247008053, std: 0.0133952508428
# cv execution time: 5424.41613007 sec
pl1 = transf_count, est_xgb

# mean: 0.531791158177, std: 0.00301742594142
# cv execution time: 273.821596146 sec
pl2 = transf_wv, est_xgb

# mean: 0.51385278959, std: 0.0149009902673
# cv execution time: 4414.65872407 sec
pl3 = transf3, est_xgb

# mean: 0.512169639091, std: 0.0127859600449
# cv execution time: 969.822750092 sec
pl4 = transf_tfidf, est_xgb

# mean: 0.495558379131, std: 0.00875619528366
# cv execution time: 1017.98022795 sec
pl5 = transf5, est_xgb

# mean: 0.526997056765, std: 0.0113952020154
# cv execution time: 467.067214966 sec
pl6 = transf7, est_rf

# mean: 0.474707501422, std: 0.0129297984942
# cv execution time: 2155.61947298 sec
pl7 = transf7, est_xgb

# mean: 0.511426873236, std: 0.00848039053981
# cv execution time: 673.606554031 sec
pl8 = transf_qms, est_xgb

# mean: 0.498169906059, std: 0.00804649346606
# cv execution time: 329.776066065 sec
pl9 = transf_qm, est_xgb

# mean: 0.491104751269, std: 0.00663380880545
# cv execution time: 1032.76950693 sec
pl10 = transf10, est_xgb

# mean: 0.498106474773, std: 0.00859122009882
# cv execution time: 345.627609015 sec
pl11 = transf_qm, est_xgb2

# mean: 0.514115117917, std: 0.00339352964921
# cv execution time: 221.978795052 sec
pl12 = transf_qma, est_xgb

# mean: 0.472303356125, std: 0.0111595095046
# cv execution time: 2385.81992888 sec
pl13 = transf12, est_xgb2

cv_test(pl13)
# submission(pl13)
