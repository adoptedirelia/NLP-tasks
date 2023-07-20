import numpy as np
import math
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer



#二值分类器
#input x(bs,features)
#output y -> 0/1
class Bayes():
    def __init__(self,features) -> None:
        # p(y|x) = p(x|y) p(y)
        # p(y) 根据样本的数量计算出来
        # p(x|y) 根据正态分布
        # p(xi|yj) = N~(a,b)
        self.classes = 2 

        self.features = features

        self.class_pro = np.zeros(self.classes)

        self.feature_pro = np.zeros((self.classes,self.features,2)) # 均值和方差
        print("************二值贝叶斯分类器******************")

        print(f"类的个数: {self.classes}")
        print(f"特征维度: {self.features}")
        print("**********************************************\n")

    def fit(self,X,y):
        # input X(batch_size,features) y(batch_size,)
        for i in range(self.classes):
            #计算出每个类在训练集中的比例
            X_c = X[y == i]
            self.class_pro[i] = len(X_c) / len(X)

            self.feature_pro[i,:,0] = X_c.mean(axis=0)
            self.feature_pro[i,:,1] = X_c.var(axis=0)


    def pre(self,X,y):
        # input X(batch_size,features) y(batch_size,)
        # output y_pre(batch_size,)

        y_pre = []
        for x in X:
            logpk = np.log(self.class_pro)
            
            theta2 = self.feature_pro[:,:,1]**2 + 1e-6 #防止除0
            mean_ = self.feature_pro[:,:,0]
            pxy = np.sum(-0.5 * np.log(2 * math.pi * theta2)-(0.5 * ((x - mean_) ** 2)) / theta2, axis = 1).T 
            #print(pxy.shape,logpk.shape)
            y_pre.append(np.argmax(logpk + pxy))
        
        y_pre = np.array(y_pre)
        print("************二值贝叶斯分类器预测结果******************")
        print(f'accuracy_score: {accuracy_score(y,y_pre):.4f}')
        print(f'f1_score: {f1_score(y,y_pre):.4f}')
        print(f'precision_score: {precision_score(y,y_pre):.4f}')
        print(f'recall_score: {recall_score(y,y_pre):.4f}\n')
        print("**********************************************\n")

        return y_pre

class multiBayes():
    def __init__(self,classes,features) -> None:

        # p(y|x) = p(x|y) p(y)
        # p(y) 根据样本的数量计算出来
        # p(x|y) 根据正态分布
        # p(xi|yj) = N~(a,b)
        self.classes = classes

        self.features = features

        self.class_pro = np.zeros(self.classes)

        self.feature_pro = np.zeros((self.classes,self.features,2)) # 均值和方差
        
        print("************多值贝叶斯分类器******************")
        print(f"类的个数: {self.classes}")
        print(f"特征维度: {self.features}")
        print("**********************************************\n")


    def fit(self,X,y):
        # input X(batch_size,features) y(batch_size,)
        for i in range(self.classes):
            #计算出每个类在训练集中的比例
            X_c = X[y == i]
            self.class_pro[i] = len(X_c) / len(X)

            self.feature_pro[i,:,0] = X_c.mean(axis=0)
            self.feature_pro[i,:,1] = X_c.var(axis=0)


    def pre(self,X,y):
        # input X(batch_size,features) y(batch_size,)
        # output y_pre(batch_size,)

        y_pre = []
        for x in X:
            logpk = np.log(self.class_pro)
            
            theta2 = np.power(self.feature_pro[:,:,1],2) + 1e-6 #防止除0
            mean_ = self.feature_pro[:,:,0]
            pxy = np.sum(-0.5 * np.log(2 * math.pi * theta2)-(0.5 * ((x - mean_) ** 2)) / theta2, axis = 1).T 
            y_pre.append(np.argmax(logpk + pxy))

        y_pre = np.array(y_pre)
        print("************多值贝叶斯分类器预测结果******************")

        print(f'accuracy_score: {accuracy_score(y,y_pre):.4f}')
        print(f'f1_score: { f1_score(y,y_pre,average="macro"):.4f}')
        print(f'precision_score: {precision_score(y,y_pre,average="macro"):.4f}')
        print(f'recall_score: {recall_score(y,y_pre,average="macro"):.4f}')
        print("**********************************************\n")
        

        return y_pre
    

def data_pre():
    file = open('./webkb-train-stemmed.txt')

    lines = file.readlines()

    word_lst = []

    label_lst = []

    for line in lines:
        line = line.strip()
        words = line.split("\t")
        label = words[0]
        if len(words) == 1:
            continue 
        words = words[1].split(" ")
        if label not in label_lst:
            label_lst.append(label)

        for word in words:
            if word not in word_lst:
                word_lst.append(word)

    return word_lst,label_lst

def mydata(word_lst,label_lst):

    file = open('./webkb-train-stemmed.txt')

    lines = file.readlines()


    label = []
    feature = []

    for line in lines:
        tmp = [0]*7287
        line = line.strip()
        words = line.split("\t")
        mlabel = words[0]
        label.append(label_lst.index(mlabel))
        if len(words) == 1:
            feature.append(tmp)
            continue 
        words = words[1].split(" ")
        
        for word in words:
            if word in word_lst:
                tmp[word_lst.index(word)]=1
        feature.append(tmp)
    trainX = np.array(feature)
    trainY = np.array(label)

    file = open('./webkb-test-stemmed.txt')

    lines = file.readlines()


    label = []
    feature = []

    for line in lines:
        tmp = [0]*7287
        line = line.strip()
        words = line.split("\t")
        mlabel = words[0]
        label.append(label_lst.index(mlabel))
        if len(words) == 1:
            feature.append(tmp)
            continue 
        words = words[1].split(" ")
        
        for word in words:
            if word in word_lst:
                tmp[word_lst.index(word)]=1

        feature.append(tmp)
    
    testX = np.array(feature)
    testY = np.array(label)
    return trainX,trainY,testX,testY
                

if __name__ == '__main__':

    # 二值贝叶斯分类器
    #X, y = make_classification(n_samples=5000, n_features=5, n_informative=5, n_redundant=0, n_clusters_per_class=1, random_state=42)
    data = load_breast_cancer()
    X = data.data
    y = data.target

    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, random_state = 32)
    
    biBayes = Bayes(30)

    biBayes.fit(trainX,trainY)

    pre = biBayes.pre(testX,testY)

    # 多值贝叶斯分类器
    #X, y = make_classification(n_samples=5000, n_features=5, n_informative=5, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=42)

    #trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, random_state = 32)
    #word_lst,label_lst = data_pre()
    #trainX,trainY,testX,testY = mydata(word_lst,label_lst)

    iris_dataset = load_iris()

    trainX, testX, trainY, testY = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    muBayes = multiBayes(3,4)


    muBayes.fit(trainX,trainY)

    pre = muBayes.pre(testX,testY)
