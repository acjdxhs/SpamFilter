from LogisticRegression import *
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from scipy.stats import pearsonr
import scipy.sparse
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif
import sklearn.naive_bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import sklearn.metrics


dir="./4.0/"



data=pickle.load(open(dir+"sparse.pkl","rb"))
label=pickle.load(open(dir+"label.pkl","rb"))


#word=pickle.load(open(dir+"word_list.pkl","rb"))

sentence_num=data.shape[0]

print(data.shape)


label=np.reshape(label,[label.shape[0],])

'''
n=range(0,sentence_num)
sentence_num=sentence_num
random_get=random.sample(n,sentence_num)
data=data[random_get]
label=label[random_get]
'''

#sel = VarianceThreshold(threshold=0.001)
#data=sel.fit_transform(data,label)
#print(sel.get_support(indices=True))


#word=pickle.load(open("./word.pkl","rb"))
#sentence=pickle.load(open("./sentence.pkl","rb"))

#ch2 = SelectKBest(chi2, k=90000)
#print(ch2.get_params())


#data=ch2.fit_transform(data,label)

#joblib.dump(ch2,dir+"ch2_50000.pkl")
#ch2.get_support(indices=True)
#with open(dir+"word_canliu_50000.txt","w+") as file:
   # for i in ch2.get_support(indices=True):
       # file.write(word[i]+"\n")




#pickle.dump(ch2.get_support(),open("./ch2_list.pkl","wb"))
#list1=ch2.get_support(indices=True)


threshold1=5e-7
#sel = VarianceThreshold(threshold=threshold1)
#data=sel.fit_transform(data)
#joblib.dump(sel,dir+"variance.jbl")
#list2=sel.get_support(indices=True)
#pickle.dump(list1[list2],open("word_transform_list.pkl","wb"))
#data=data[:,9023:]
#print(data.shape)



n_topics = 200






#threshold1=0

#file=open(dir+"result_num_%d_threshold_%e_not balance_去除字母数字.txt"%(data.shape[1],threshold1),"w+")

X_train,X_test,y_train,y_test=train_test_split(data,label, test_size=0.3, random_state=0,shuffle=True)
import datetime
starttime = datetime.datetime.now()
clf = sklearn.linear_model.LogisticRegression(solver='sag')#class_weight='balanced'

clf.fit(X_train,y_train)
endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
'''
#joblib.dump(clf,dir+"clf_logistic_50000.jbl")
#with open(dir + "coef_50000.txt", "w+") as file_2:
    #for i in clf.coef_[0]:
        #file_2.write(str(i) + "\n")
y_logis = clf.predict(X_test)
s=classification_report(y_test,y_logis)
predict_prob_y = clf.predict_proba(X_test)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
# end svm ,start metrics
print(predict_prob_y)

print(y_test)
test_auc = sklearn.metrics.roc_auc_score(y_test, predict_prob_y[:,1])  # 验证集上的auc值
pre=precision_score(y_test,y_logis)
recall=recall_score(y_test,y_logis)
acc=accuracy_score(y_test,y_logis)


file.write(s)
file.write("\n")
file.write("precision:"+str(pre)+"\n")
file.write("recall:"+str(recall)+"\n")
file.write("auc:"+str(test_auc)+"\n")
file.write("accuracy:"+str(acc)+"\n")

print(s)
print(test_auc)
file.write("\n\n\n")

file.close()

kf=KFold(n_splits=10,shuffle=True)
#下面部分是十折的
for train_index,test_index in kf.split(data,label):
    clf = sklearn.linear_model.LogisticRegression(C=0.1)#class_weight='balanced'
   # tmp_data_train=lda.transform(data[train_index])
    #clf=sklearn.naive_bayes.GaussianNB()


    tmp_data_train=data[train_index]
    tmp_label_train=label[train_index]

    #tmp_data_test=lda.transform(data[test_index])
    tmp_data_test=data[test_index]
    tmp_label_test=label[test_index]

    X_train,X_test=tmp_data_train,tmp_data_test
    y_train,y_test=tmp_label_train,tmp_label_test

    print("training logistic")
    clf.fit(X_train,y_train)
    with open(dir+"coef.txt","w+") as file_2:
        for i in clf.coef_[0]:
            file_2.write(str(i)+"\n")
    print("computing")
    y_logis = clf.predict(X_test)
    s=classification_report(y_test,y_logis)
    predict_prob_y = clf.predict_proba(X_test)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
    # end svm ,start metrics
    print(predict_prob_y)
    print(y_test)
    test_auc = sklearn.metrics.roc_auc_score(y_test, predict_prob_y[:,1])  # 验证集上的auc值
    pre=precision_score(y_test,y_logis)
    recall=recall_score(y_test,y_logis)
    acc=accuracy_score(y_test,y_logis)


    file.write(s)
    file.write("\n")
    file.write("precision:"+str(pre)+"\n")
    file.write("recall:"+str(recall)+"\n")
    file.write("auc:"+str(test_auc)+"\n")
    file.write("accuracy:"+str(acc)+"\n")

    print(s)
    print(test_auc)
    file.write("\n\n\n")

    file.flush()

'''





















