import os
import numpy as np
import matplotlib.pyplot as plt

class FileProcess():

    def __init__(self):
        self.filePath = "./data"

    def contentAcquire(self):
        """
        Acquire content from the processed data
        :return: SP_list, HA_list
        [[word1, word2, ...], [word1, word2, ...], ...]
        """
        #---acquire spam contents---
        sp_dirs=os.listdir(self.filePath+"/SPAM")
        SP_list=[]
        for file in sp_dirs:
            with open(self.filePath+"/SPAM/"+file, encoding="ANSI") as f:
                text=f.read()
            SP_list.append(text.split(","))
        SP_test=SP_list[int(len(SP_list)*0.8):]
        SP_list=SP_list[:int(len(SP_list)*0.8)]
        #---acquire ham contents---
        ha_dirs = os.listdir(self.filePath + "/HAM")
        HA_list = []
        for file in ha_dirs:
            with open(self.filePath+"/HAM/"+file, encoding="ANSI") as f:
                text = f.read()
            HA_list.append(text.split(","))
        HA_test = HA_list[int(len(HA_list)*0.8):]
        HA_list = HA_list[:int(len(HA_list)*0.8)]
        print("已获取内容")
        return SP_list, HA_list, SP_test, HA_test

    def wordCount(self, SP_list, HA_list, SP_test, HA_test):
        """
        Count the number of words in each file
        :param SP_list: [[word1, word2, ...], [word1, word2, ...], ...]
        :param HA_list: [[word1, word2, ...], [word1, word2, ...], ...]
        :return: SP_dic_list, HA_dic_list
        [{word1:#1, word2:#2, ...}, {word1:#1, word2:#2, ...}, ...]
        """
        SP_dic_list=[]
        for file in SP_list:
            SP_dic_list.append({word:file.count(word) for word in file})
        HA_dic_list = []
        for file in HA_list:
            HA_dic_list.append({word: file.count(word) for word in file})
        SP_dic_test=[]
        for file in SP_test:
            SP_dic_test.append({word:file.count(word) for word in file})
        HA_dic_test = []
        for file in HA_test:
            HA_dic_test.append({word: file.count(word) for word in file})
        print("已完成计数")
        """
        这四个循环可以写到一起吗？？？
        """
        return SP_dic_list, HA_dic_list, SP_dic_test, HA_dic_test

    def wordlistAcquire(self, SP_dic_list, HA_dic_list):
        """
        Count the number of words in all files
        :param SP_dic_list: [{word1:#1, word2:#2, ...}, {word1:#1, word2:#2, ...}, ...]
        :param HA_dic_list:
        :return: word_dic
        """
        word_dic={}
        for dic in SP_dic_list:
            for word in dic:
                if word in word_dic:
                    word_dic[word]+=dic[word]
                else:
                    word_dic[word]=dic[word]
        for dic in HA_dic_list:
            for word in dic:
                if word in word_dic:
                    word_dic[word] += dic[word]
                else:
                    word_dic[word] = dic[word]
        return word_dic

    def tfidfTransform(self,SP_dic_list, HA_dic_list):
        """
        Convert number dic of words in spam and ham contents
        into TF matrix
        Then, convert it into IDF matrix
        And get TF-IDF matrix (X) and label matix (label)
        :param SP_dic_list: [{word1:#1, word2:#2, ...}, {word1:#1, word2:#2, ...}, ...]
        :param HA_dic_list:
        :return: X, label
        """
        #----TF----
        #SP_dic_list:[{word1:#1, word2:#2, ...}, {word1:#1, word2:#2, ...}, ...]
        X_TF=np.zeros(((len(SP_dic_list)+len(HA_dic_list)), len(word_list)))
        for dic in SP_dic_list:
            total=sum(dic.values())
            for word in dic:
                if word in word_list:
                    X_TF[SP_dic_list.index(dic), word_list.index(word)]=dic[word]/total
        for dic in HA_dic_list:
            total=sum(dic.values())
            for word in dic:
                if word in word_list:
                    X_TF[(HA_dic_list.index(dic)+len(SP_dic_list)), word_list.index(word)]=dic[word]/total
        #---IDF---
        wordTimes=np.sum(np.int64(X_TF>0) ,axis=0) #How many files a word appears in
        X_IDF=np.log10(((len(SP_dic_list)+len(HA_dic_list)))/(1+wordTimes))
        #---TF-IDF---
        X=X_TF*X_IDF
        one=np.ones(len(SP_dic_list)+len(HA_dic_list))
        X=np.insert(X,0,one, axis=1)
        print("已完成TF-IDF转化")
        #---label---
        label=np.zeros(((len(SP_dic_list)+len(HA_dic_list)), 1))
        for i in range(len(SP_dic_list)):
            label[i,0]+=1
        return X, label

    def GradientDescent(self, SP_dic_test, HA_dic_test):
        """
        Use gradient descent to find the smallest loss function.
        :param SP_dic_test:
        :param HA_dic_test:
        :return:
        """
        w=np.zeros([len(word_list)+1,1]) #set the initial w
        #---initialize the loss function---
        Z=np.dot(X,w)
        Z=1/(1+np.exp(-Z))
        Loss=-(label*np.log(Z)+(1-label)*np.log(1-Z))
        L=np.sum(Loss, axis=0)
        eta=10 #set the length of steps
        #---initialize the plot---
        fig, ax=plt.subplots()
        step_list=[]; Accuracy_list=[]
        for i in range(2000):
            dw = np.dot(X.T, Z - label) / (len(SP_dic_list) + len(HA_dic_list))
            w = w - eta * dw
            Z = np.dot(X, w)
            Z = 1 / (1 + np.exp(-Z))
            loss = -(label * np.log(Z) + (1 - label) * np.log(1 - Z))
            l=np.sum(loss, axis=0)
            if L[0] > l[0]:
                L[0]=l[0]
            if i//100==i/100:
                #---paint the plot---
                print(i, L[0])
                accuracy=self.test(w, SP_dic_test, HA_dic_test)
                step_list.append(i); Accuracy_list.append(accuracy)
                ax.scatter(step_list, Accuracy_list, label="ACCU")
                ax.legend()
                plt.pause(1)


    def test(self, w, SP_dic_test, HA_dic_test):
        """
        Test the w
        """
        T=0; F=0
        #---test spam contents---
        for dic in SP_dic_test:
            array_list=[]
            for word in word_list:
                if word in dic:
                    array_list.append(dic[word])
                else:
                    array_list.append(0)
            np_test=np.array(array_list)
            test_TF=np_test/sum(dic.values())
            wordTimes = np.sum(np.int64(test_TF > 0), axis=0)  # How many files a word appears in
            test_IDF = np.log10(((len(SP_dic_test) + len(HA_dic_test))*4) / (1 + wordTimes))
            test_np=test_TF*test_IDF
            test_np = np.insert(test_np, 0, 1)
            y=np.dot(test_np,w)
            y = 1 / (1 + np.exp(-y))
            if y >= 0.5: T+=1
            else: F+=1
        #---test ham contents---
        for dic in HA_dic_test:
            array_list=[]
            for word in word_list:
                if word in dic:
                    array_list.append(dic[word])
                else:
                    array_list.append(0)
            np_test=np.array(array_list)
            test_TF=np_test/sum(dic.values())
            wordTimes = np.sum(np.int64(test_TF > 0), axis=0)  # How many files a word appears in
            test_IDF = np.log10(((len(SP_dic_test) + len(HA_dic_test))*4) / (1 + wordTimes))
            test_np=test_TF*test_IDF
            test_np = np.insert(test_np, 0, 1)
            y=np.dot(test_np,w)
            y = 1 / (1 + np.exp(-y))
            if y >= 0.5: F+=1
            else: T+=1
        print("T =", T, "F =", F)
        print("accuracy =", 100*T/(T+F), "%")
        return 100*T/(T+F)


if __name__=="__main__":
    global label
    global X  # (len(SP_dic_list)+len(HA_dic_list))*(len(word_list)+1)
    global word_list
    obj = FileProcess()
    if "numpy_save.npz" in os.listdir("./"):
        #---load necessary matrices---
        np.load.__defaults__=(None, True, True, "ASCII")
        npzfile = np.load("numpy_save.npz")
        np.load.__defaults__ = (None, False, True, "ASCII")
        X=npzfile["X"]
        label=npzfile["label"]
        word_list_np=npzfile["word_list_np"]
        SP_dic_test_np=npzfile["SP_dic_test_np"]
        HA_dic_test_np=npzfile["HA_dic_test_np"]
        SP_dic_list_np=npzfile["SP_dic_list_np"]
        HA_dic_list_np=npzfile["HA_dic_list_np"]
        word_list=word_list_np.tolist()
        SP_dic_test=SP_dic_test_np.tolist()
        HA_dic_test=HA_dic_test_np.tolist()
        SP_dic_list=SP_dic_list_np.tolist()
        HA_dic_list=HA_dic_list_np.tolist()
        print("载入成功")
    else:
        #---store necessary matrices---
        (SP_list, HA_list, SP_test, HA_test)=obj.contentAcquire()
        (SP_dic_list, HA_dic_list, SP_dic_test, HA_dic_test)=obj.wordCount(SP_list, HA_list, SP_test, HA_test)
        word_dic=obj.wordlistAcquire(SP_dic_list, HA_dic_list)
        word_list=[word for word in word_dic if word_dic[word]>100]
        X,label=obj.tfidfTransform(SP_dic_list, HA_dic_list)
        #Convert lists into matrices
        word_list_np=np.array(word_list)
        SP_dic_list_np=np.array(SP_dic_list)
        HA_dic_list_np=np.array(HA_dic_list)
        SP_dic_test_np=np.array(SP_dic_test)
        HA_dic_test_np=np.array(HA_dic_test)
        #Store the matrices: X, label, word_list, SP_dic_test, HA_dic_test
        np.savez("numpy_save", X=X, label=label, word_list_np=word_list_np,
                 SP_dic_test_np=SP_dic_test_np, HA_dic_test_np=HA_dic_test_np,
                 SP_dic_list_np=SP_dic_list_np, HA_dic_list_np=HA_dic_list_np)
        print("保存成功")

    obj.GradientDescent(SP_dic_test, HA_dic_test)