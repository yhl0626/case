# coding=UTF-8

import pandas as pd
import re

# 从岗位信息中提取特征

def featureAware(s , statis):
    srr = s.split("\r\n")
    pattern = re.compile("[A-Z\/ ]+:")
    for line in srr :
        line = line.strip()
        arr = pattern.findall(line)

        if len(arr) == 1 :
            idx = line.find(":")
            s = line[:idx+1]
            t = arr[0]
            if s == t:
                if t not in statis:
                    statis[t] = 1
                statis[t] = statis[t] + 1


def demo():
    inpath = "../../../data/common/data-job-posts.csv"
    statis = {}
    df = pd.read_csv(inpath)
    for i in df.values :
        t = i[1].replace("\r\n","\t###\\r\\n###\n")
        featureAware(i[1] , statis)
        for line in i[1].split("\r\n"):
            if "INTENDED AUDIENCE:" in line :
                print(line)

    for i in statis :
        freq = statis[i]
        if freq > 100 :
            print(i, statis[i])

def demo1():
    inpath = "../../../data/common/data-job-posts.csv"
    df = pd.read_csv(inpath)
    print(df.columns)

if __name__ == '__main__':
    demo()