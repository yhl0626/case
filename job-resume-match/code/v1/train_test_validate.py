# coding=UTF-8

import os
import random
from sklearn.model_selection import train_test_split

def createDir(dirpath):
    if os.path.exists(dirpath) == False :
        os.makedirs(dirpath)

# 数据分配，80%训练集，10%验证集，10%测试集
def dataAllocation():
    inpath = conf["train_raw_path"]
    outdir = conf["train_test_valid_path"]
    poslist = []
    neglist = []
    with open(inpath , "r") as file :
        line = file.readline()
        while line:
            line = line.strip()
            srr = line.split("\t")
            if len(srr) == 3 :
                label = srr[0]
                if label == "1":
                    poslist.append(line)
                else:
                    neglist.append(line)
            line = file.readline()

    # 划分训练集和中间数据
    train_pos , temp_pos = train_test_split(poslist , test_size=0.2)
    # 划分验证集和测试集
    valid_pos, test_pos = train_test_split(temp_pos, test_size =0.5)

    # 划分训练集和中间数据
    train_neg, temp_neg = train_test_split(neglist, test_size=0.2)
    # 划分验证集和测试集
    valid_neg, test_neg = train_test_split(temp_neg, test_size=0.5)

    createDir(outdir)

    outpath = os.path.join(outdir , "train.dat")
    with open(outpath , "w") as file :
        tmplist = train_neg + train_pos
        random.shuffle(tmplist)
        for line in tmplist:
            file.write(line + "\n")
            file.flush()

    outpath = os.path.join(outdir, "valid.dat")
    with open(outpath, "w") as file:
        tmplist = valid_neg + valid_pos
        random.shuffle(tmplist)
        for line in tmplist:
            file.write(line + "\n")
            file.flush()

    outpath = os.path.join(outdir, "test.dat")
    with open(outpath, "w") as file:
        tmplist = test_neg + test_pos
        random.shuffle(tmplist)
        for line in tmplist:
            file.write(line + "\n")
            file.flush()

conf = {
    "train_raw_path":"../../data/train/train-raw.dat",
    "train_test_valid_path":"../../data/train/v1/",
}

def execute():
    dataAllocation()
    print("Successfully partitioned training, testing, and validation sets")

if __name__ == '__main__':
    execute()