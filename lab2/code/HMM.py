import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


class HMM:
    def __init__(self) -> None:

        # 有 B M S E 四种状态
        self.states = ['B','M','S','E']
        self.transition_matrix = np.zeros((4,4))
        self.transition_num = np.zeros((4,4))

        self.emission_matrix = {'B':{'total':0},'M':{'total':0},'S':{'total':0},'E':{'total':0}}

        self.init_pro = np.zeros(4)
        self.init_num = np.zeros(4)

        self.map = {'B':0,'M':1,'S':2,'E':3}
        print("*******************HMM构建成功******************\n")
        
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
                    self.emission_matrix[state][word] /= self.emission_matrix[state]['total']

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
                flag =  sentence[t] not in self.emission_matrix['S'].keys() and \
                        sentence[t] not in self.emission_matrix['B'].keys() and \
                        sentence[t] not in self.emission_matrix['M'].keys() and \
                        sentence[t] not in self.emission_matrix['E'].keys()
                

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
        
        for sentence in y_pre:
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

        print("\n正在计算指标...")
        for i,j in tqdm(zip(y_lst,y_pre_lst)):
            accuracy += accuracy_score(i,j)
            f1 += f1_score(i,j,average='macro')
            precision += precision_score(i,j,average='macro')
            recall += recall_score(i,j,average='macro')

        print("\n*******************预测结果******************")
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
            flag =  sentence[t] not in self.emission_matrix['S'].keys() and \
                    sentence[t] not in self.emission_matrix['B'].keys() and \
                    sentence[t] not in self.emission_matrix['M'].keys() and \
                    sentence[t] not in self.emission_matrix['E'].keys()
            

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
            if s == 'S' or s == 'E':
                result += " "
        return result


def make_train_set():
    print("正在读取数据集...")
    file = open('../dataset/pku_train&test_corpus/pku_training.txt','r',encoding='gbk')
    
    #结果
    trainX = []
    trainY = []

    lines = file.readlines()
    for line in tqdm(lines):
        #有回车符会读入，所以不读取最后一个
        sentence = line.split("  ")[:-1]
        trainstr = ""
        lablestr = ""
        for word in sentence:
            trainstr += word
            if len(word)==1:
                lablestr += 'S'
            else:
                for i,w in enumerate(word):
                    if i==0:
                        lablestr += 'B'
                    elif i==len(word)-1:
                        lablestr += 'E'
                    else:
                        lablestr += 'M'
        if trainstr == "":
            continue
        trainX.append(trainstr)
        trainY.append(lablestr)

    file.close()
    print("读取完成\n")
    return trainX,trainY

def make_test_set():
    print("正在读取数据集...")
    file = open('../dataset/pku_train&test_corpus/pku_test_gold.txt','r',encoding='gbk')
    
    #结果
    testX = []
    testY = []

    lines = file.readlines()
    for line in tqdm(lines):
        #有回车符会读入，所以不读取最后一个
        sentence = line.split("  ")[:-1]
        teststr = ""
        lablestr = ""
        for word in sentence:
            teststr += word
            if len(word)==1:
                lablestr += 'S'
            else:
                for i,w in enumerate(word):
                    if i==0:
                        lablestr += 'B'
                    elif i==len(word)-1:
                        lablestr += 'E'
                    else:
                        lablestr += 'M'
        if teststr == "":
            continue
        testX.append(teststr)
        testY.append(lablestr)

    file.close()
    print("读取完成")
    return testX,testY


if __name__ == '__main__':

    a,b = make_train_set()
    
    c,d = make_test_set()
    
    hmm = HMM()
    hmm.fit(a,b)
    hmm.pre(c,d)

    test_sentence = "数学王子泰老师"
    print(f"原句:\t{test_sentence}\n分词后:\t{hmm.test(test_sentence)}")