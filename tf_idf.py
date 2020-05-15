import numpy as np

from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter
import math

def sklearn_tfidf(tag_list):
    
    tfidf_model2 = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(tag_list)
    print(tfidf_model2.vocabulary_)
    # 将文本中的词语转换为词频矩阵  
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b") 
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(tag_list) 
    # print(X)
    transformer = TfidfTransformer(norm=None)  
    # 将词频矩阵X统计成TF-IDF值 
    tfidf = transformer.fit_transform(X)  
    print(tfidf.toarray())

def get_list(tag_list):
    words_list = []
    words = []
    for i in tag_list:
        doc = i.split(' ')
        words_list.append(doc)
        words.extend(doc)
    # print(words_list)
    count_list = []
    for i in words_list:
        count_list.append(Counter(i))
    return count_list, set(words)

def tf(word, count):
    return count[word]

def df(word, count_list):
    return sum(1 for i in count_list if i[word])

def idf(word, count_list):
    return math.log((1+len(count_list)) / (1+df(word, count_list)))+1

def tfidf(tag_list):
    count_list, words = get_list(tag_list)
    print(dict(zip(words, range(0, len(words)))))
    # print(words)
    res = []
    for doc in count_list:
        tf_idf_doc = []
        for word in words:
            tf_idf_doc.append(tf(word, doc) * idf(word, count_list))
        res.append(tf_idf_doc)
    print(np.array(res))
    return np.array(res)

if __name__ == '__main__':
    tag_list = ['我 来到 北京 清华大学',
                '他 来到 了 网易 杭研 大厦',
                '小明 硕士 毕业 与 中国 科学院',
                '我 爱 北京 天安门']
    tfidf(tag_list)
    sklearn_tfidf(tag_list)