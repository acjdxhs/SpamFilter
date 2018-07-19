import jieba
import os
import pickle
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def readfile1(filepath):
    with open(filepath,'r',encoding = "UTF-8")as f:
        content = f.readlines()
    return content

def readfile2(filepath):
    file = open(filepath,"rb")
    data = pickle.load(file)
    file.close()
    return data

def readfile3(filepath):
    clf=joblib.load(filepath)
    return clf

def savefile(savepath,x):
    file = open(savepath,"wb")
    pickle.dump(x,file)
    file.close() 

def savefile2(savepath,x):
    joblib.dump(x,savepath)

def get_stopwords(file_path):
    stpwrd_dic = open(file_path, 'rb')
    stpwrd_content = stpwrd_dic.read()
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()
    return stpwrdlst

def group(corpus_path):
	file_list = os.listdir(corpus_path)
	x = []
	for file_path in file_list:
		full_name = corpus_path + file_path
		content = readfile1(full_name)
		for i in range(len(content)):
			x.append(content[i])
	return x


def Train(corpus,stopwords):
    vectorizer=TfidfVectorizer(stop_words=stopwords)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vector = vectorizer.fit_transform(corpus)
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vector)#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    #weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return vectorizer,transformer,tfidf,word

def split(sentence):
	x = []
	for i in range(len(sentence)):
		content = jieba.cut(sentence[i])
		x.append(" ".join(content))
	return x

def split2(sentence):
	x = []
	content = jieba.cut(sentence)
	x.append(" ".join(content))
	return x

def de_weighting(x):
	return list(set(x))

def get_tfidf(inpu,model1 = "./model/feature/model1_4.0.pkl",model2="./model/feature/model2_4.0.pkl"):
    model1 = readfile3(model1)
    model2 = readfile3(model2)
    vector = model1.transform(inpu)
    tfidf = model2.transform(vector)
    return tfidf    


