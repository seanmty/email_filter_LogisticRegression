import jieba
import os
import time
import random

def data_Process(path_lable_list):
    """
    Randomrize the list
    Delete stopwords,
    and use jieba to cut word,
    and store them in a direction for further use
    :param path_lable_list:
    :return:
    """
    swfile = open("./cn_stopwords.txt", encoding='utf-8', errors='ignore')
    stopword = swfile.readlines()
    stopword = [word.strip("\n") for word in stopword]
    leng=len(path_lable_list); seg=leng//50
    i=0; start=time.perf_counter()
    random.shuffle(path_lable_list)
    for file in path_lable_list:
        for word in stopword:
            file[2]=file[2].replace(word,"")
        file[2] = jieba.lcut(file[2])
        file[2] = list(filter(('\n').__ne__, file[2]))
        #完成停用词去除和分词
        dir_name=file[0]
        file_path="./data/"+dir_name+"/"+file[1].replace("/", "-")+".txt"
        txt=open(file_path, "w+", errors="ignore")
        for word in file[2]:
            txt.write(word+",") #use "," to seperate words
        txt.close()
        a = "*"*(i//seg); b="."*(leng//seg-i//seg); c=((i//seg)/50)*100
        dur=time.perf_counter()-start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end='')
        i+=1
    print("\n"+"已完成数据处理")

def content_Acquire():
    """
    Acquire label, path, and content
    and put them in a list(path_label_list).
    For every element in the list, it is also a list:
    [label, path, content]
    :return:
    """
    path_lable_list = []  # initialize path and label list of e-mails
    INDEX = open("./trec06c/full/index", encoding='utf-8', errors='ignore')
    indexlines = INDEX.readlines()
    for line in indexlines:
        line = line.strip("\n")
        label = line.split(" ")[0]
        path = "./trec06c" + line.split(" ")[1].replace("..", "")
        path_lable_list.append([label.upper(), path])
    print("已获取路径")
    for file in path_lable_list:
        f = open(file[1], encoding='ANSI', errors='ignore')
        file[1]=file[1].replace("./trec06c/data/", "")
        content = f.readlines()
        content = content[(content.index("\n")) + 1:]
        Content = "".join(content)
        file.append(Content)
    print("已获取内容")
    return path_lable_list

if __name__=="__main__":
    if os.path.exists("./data/SPAM"):
        print("数据已处理")
    else:
        os.makedirs("./data/SPAM")
        os.makedirs("./data/HAM")
        path_lable_list=content_Acquire()
        data_Process(path_lable_list)
