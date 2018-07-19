import sys
import getopt
from model import *
from sklearn.externals import joblib

ch2_path = "./model/train/ch2.jbl"
clf_path = "./model/train/clf_logistic.jbl"

def print_usage():
    print('''
    选项
    -h  帮助
    -s  <sentence>  判断句子是不是垃圾文本
    --file=<input>  将当前目录下的文档分成垃圾文档和正常文档''')


def check_sentence(sentence):
    inpu = split2(sentence)
    tfidf = get_tfidf(inpu)
    ch2 = joblib.load(ch2_path)
    #variance = joblib.load("./model/train/halfset/variance.jbl")
    clf = joblib.load(clf_path)
    data1 = ch2.transform(tfidf)
    #data2 = variance.transform(data1)
    return clf.predict(data1)


def check_file(file):
    try:
        f = open("./" + file, "r", encoding="UTF-8")
    except FileNotFoundError:
        print("当前文件夹下找不到该文件")
        exit(2)
    content = f.readlines()
    inpu = split(content)
    tf_idf = get_tfidf(inpu)
    ch2 = joblib.load(ch2_path)
    clf = joblib.load(clf_path)
    result = clf.predict(ch2.transform(tf_idf))
    print(result)
    spam = open("spam.txt", "w", encoding="UTF-8")
    normal = open("normal.txt", "w", encoding="UTF-8")
    for r, line in zip(result, content):
        if r == 0:
            spam.write(line)
        else:
            normal.write(line)
    f.close()
    spam.close()
    normal.close()


try:
    opts, args = getopt.getopt(sys.argv[1:], "hs:", ["file="])
except getopt.GetoptError:
    print_usage()
    sys.exit(1)
for opt, value in opts:
    if opt == '-h':
        print_usage()
    elif opt == '-s':
        result = check_sentence(value)
        if result[0] == 0:
            print("垃圾文本")
        else:
            print("正常文本")
    elif opt == '--file':
        check_file(value)

