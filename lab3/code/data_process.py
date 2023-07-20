#crf++
from tqdm import tqdm

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

    file = open('./train.data','w')
    string = ""
    for s1,s2 in zip(a,b):
        for word,state in zip(s1,s2):
            #print(word,state)
            file.write(f"{word}\t{state}\n")
        file.write("\n")

    a,b,c,d = make_test_set1998()

    file = open('./test.data','w')
    string = ""
    for s1,s2 in zip(a,b):
        for word,state in zip(s1,s2):
            #print(word,state)
            file.write(f"{word}\t{state}\n")
        file.write("\n")

    