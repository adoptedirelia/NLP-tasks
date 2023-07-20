import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class HMM:
    def __init__(self,cls_lst) -> None:

        # 有 B M S E 四种状态
        self.states = [c for c in cls_lst]
        self.transition_matrix = np.zeros((len(cls_lst),len(cls_lst)))
        self.transition_num = np.zeros((len(cls_lst),len(cls_lst)))

        self.emission_matrix = {}
        for cls in cls_lst:
            self.emission_matrix[cls] = {'total':0}

        self.init_pro = np.zeros(len(cls_lst))
        self.init_num = np.zeros(len(cls_lst))

        self.map = {}
        for idx,cls in enumerate(cls_lst):
            self.map[cls] = idx
        print("\n*******************HMM构建成功******************")
        
    def fit(self,X,y):
        print("正在训练...")
        for sentenceX,sentencey in tqdm(zip(X,y)):
            # 计算初始矩阵
            self.init_num[self.map[sentencey[0]]] += 1

            # 计算转移矩阵

            length = len(sentencey)
            for i in range(length-1):
                now,next = self.map[sentencey[i]],self.map[sentencey[i+1]]
                self.transition_num[now,next] += 1
                
            for i,j in zip(sentenceX,sentencey):
                # i,j 分别是字和状态
                self.emission_matrix[j][i] = self.emission_matrix[j].get(i,0) + 1
                self.emission_matrix[j]["total"] += 1
            # 计算发射矩阵

        self.init_pro = self.init_num/self.init_num.sum()
        self.transition_matrix = self.transition_num/self.transition_num.sum(axis=1)

        for state in self.emission_matrix:
            
            for word in self.emission_matrix[state]:
                if word == 'total':
                    continue
                else:
                    self.emission_matrix[state][word] /= self.emission_matrix[state]['total']/1000

    def pre(self,X,y_true):
        # 维特比算法
        y_lst = []
        y_pre = []
        y_pre_lst = []
        # 转化成数字方便计算各种指标
        for sentence in y_true:
            lst = []
            for ch in sentence:
                lst.append(self.map[ch])
            lst = np.array(lst)
            y_lst.append(lst)

        print("正在测试...")
        for sentence in tqdm(X):
            V = [{}]
            path = {}

            for y in self.states:
                V[0][y] = self.init_pro[self.map[y]] * self.emission_matrix[y].get(sentence[0],0)
                path[y] = [y]

            for t in range(1,len(sentence)):
                V.append({})
                newpath = {}

                flag = sentence[t] not in self.emission_matrix[self.states[0]]
                for state in self.states:
                    flag = sentence[t] not in self.emission_matrix[state] and flag
                

                for y in self.states:

                    emit_pro = self.emission_matrix[y].get(sentence[t],0) if not flag else 1

                    #print([(V[t - 1][y0] * self.transition_matrix[self.map[y0],self.map[y]] * emit_pro, y0)  for y0 in self.states if V[t - 1][y0] > 0])
                    (prob, state) = max([(V[t - 1][y0] * self.transition_matrix[self.map[y0],self.map[y]] * emit_pro, y0)  for y0 in self.states])
                    V[t][y] = prob
                    newpath[y] = path[state] + [y]
                
                path = newpath
                #print(path)


            (prob,state) = max([(V[len(sentence)-1][y],y) for y in self.states])
            y_pre.append(path[state])

        for sentence in tqdm(y_pre):
            lst = []
            for ch in sentence:
                lst.append(self.map[ch])

            lst = np.array(lst)
            y_pre_lst.append(lst)

        count = len(y_lst)

        accuracy = 0
        f1 = 0
        precision = 0
        recall = 0

        print("正在计算指标...")
        for i,j in zip(y_lst,y_pre_lst):
            accuracy += accuracy_score(i,j)
            f1 += f1_score(i,j,average='macro')
            precision += precision_score(i,j,average='macro')
            recall += recall_score(i,j,average='macro')

        print("************预测结果******************")
        print(f'accuracy_score: {accuracy/count}')
        print(f'f1_score: { f1/count}')
        print(f'precision_score: {precision/count}')
        print(f'recall_score: {recall/count}')
        print("**********************************************\n")   

        return y_pre
    
    def test(self,X):

        sentence = X 

        V = [{}]
        path = {}

        for y in self.states:
            V[0][y] = self.init_pro[self.map[y]] * self.emission_matrix[y].get(sentence[0],0)
            path[y] = [y]

        for t in range(1,len(sentence)):
            V.append({})
            newpath = {}
            flag = sentence[t] not in self.emission_matrix[self.states[0]]
            for state in self.states:
                flag = flag not in self.emission_matrix[state] and flag
            

            for y in self.states:

                emit_pro = self.emission_matrix[y].get(sentence[t],0) if not flag else 1

                #print([(V[t - 1][y0] * self.transition_matrix[self.map[y0],self.map[y]] * emit_pro, y0)  for y0 in self.states if V[t - 1][y0] > 0])
                (prob, state) = max([(V[t - 1][y0] * self.transition_matrix[self.map[y0],self.map[y]] * emit_pro, y0)  for y0 in self.states])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            
            path = newpath
            #print(path)


        (prob,state) = max([(V[len(sentence)-1][y],y) for y in self.states])


        result = ""

        for t,s in zip(sentence,path[state]):
            result += t
            result += '/'
            result += s
        return result
    

def make_train_set1998():
    print("正在读取数据集...")
    file = open('../dataset/pku_train&test_corpus/train1998人民日报.txt')

    lines = file.readlines()

    classset = []
    wordset = []
    trainX = []
    trainY = []
    for line in tqdm(lines):
        if line == "\n":
            continue
        sentence = line.split("  ")[:-1]
        flag = False
        index = 0
        a = []
        b = []
        for i,word in enumerate(sentence):
            tmp = word.split('/')
            

            w = tmp[0]
            c = tmp[1]
            if '[' in w:
                index = i
                flag = True  

            if not flag:
                a.append(w)
                b.append(c)
                if c not in classset:
                    classset.append(c)


            #找到结尾
            if ']' in c:
                string = ""
                for j in range(index,i+1):
                    #tmp = j.split('/')

                    tmp = sentence[j].split('/')[0]
                    cls = sentence[j].split('/')[1]
                    if '[' in tmp:
                        a.append(tmp[1:])
                    else:
                        a.append(tmp)

                    if ']' in cls:
                        tt = cls.split(']')
                        if tt[0] not in classset:
                            classset.append(tt[0])
                        if tt[1] not in classset:
                            classset.append(tt[1])
                        b.append(tt[0])
                    else:
                        b.append(cls)

                wordset.append(string[1:])
                flag = False
        trainX.append(a[1:])
        trainY.append(b[1:])

    return trainX,trainY,classset,wordset

def make_test_set1998():
    print("正在读取数据集")
    file = open('../dataset/pku_train&test_corpus/test1998人民日报.txt')

    lines = file.readlines()

    testX = []
    testY = []
    wordset = []
    classset = []
    for line in tqdm(lines):
        if line =='\n':
            continue
        sentence = line.split("  ")[:-1]
        flag = False
        index = 0
        a = []
        b = []
        for i,word in enumerate(sentence):
            tmp = word.split('/')

            w = tmp[0]
            c = tmp[1]
            if '[' in w:
                index = i
                flag = True  

            if not flag:
                a.append(w)
                b.append(c)
                if c not in classset:
                    classset.append(c)
            #找到结尾
            if ']' in c:
                string = ""
                for j in range(index,i+1):
                    #tmp = j.split('/')

                    tmp = sentence[j].split('/')[0]
                    cls = sentence[j].split('/')[1]
                    if '[' in tmp:
                        a.append(tmp[1:])
                    else:
                        a.append(tmp)

                    if ']' in cls:
                        tt = cls.split(']')
                        if tt[0] not in classset:
                            classset.append(tt[0])
                        if tt[1] not in classset:
                            classset.append(tt[1])
                        b.append(tt[0])
                    else:
                        b.append(cls)

                wordset.append(string[1:])
                flag = False
        testX.append(a[1:])
        testY.append(b[1:])

    return testX,testY,classset,wordset



if __name__ == '__main__':
    a,b,c,d = make_train_set1998()
    
    a2,b2,c2,d2 = make_test_set1998()
    
    hmm = HMM(c)
    hmm.fit(a,b)
    hmm.pre(a2,b2)