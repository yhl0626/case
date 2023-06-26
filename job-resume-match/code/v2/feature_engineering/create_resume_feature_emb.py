# coding=UTF-8

import pandas as pd
import re
import os
import traceback
from tqdm import tqdm
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

def createDir(filepath) :
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False :
        os.makedirs(parent_dir)

# 加载模型
def loadBert():
    # model_path = "nreimers/albert-small-v2"
    # model_path = "C:/Users/123/.cache/torch/sentence_transformers/nreimers_albert-small-v2"
    model_path = conf["model_path"]
    model = SentenceTransformer(model_path)
    return model

def getStopWord():
    inpath = conf["stopword_path"]
    stopWord = {}
    with open(inpath , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            if len(line) > 1:
                stopWord[line] = None
            line = file.readline()
    return stopWord

def getEmb(input , stopWord , model):

    if isinstance(input , str) and input.strip() != "NAN" :
        input = input.replace(",", ".").replace("\r\n", ".").strip()
        list1 = input.split(".")
        length = len(list1)
        for i in range(length):
            item = list1[i]
            item = re.sub("[^a-z ]", "", item.lower()).strip()
            item = re.sub("[ ]+", " ", item)
            words = item.split(" ")
            tlist = []
            for w in words:
                if w not in stopWord:
                    tlist.append(w)
            list1[i] = " ".join(tlist)

        embeddings = model.encode(list1)
        emb = list(embeddings.mean(axis=0))

        for i in range(len(emb)):
            emb[i] = str(emb[i])

        return ",".join(emb)
    else:
        return ",".join(["0" for i in range(768)])

conf = {
    "model_path":"../../../data/model/pretrain/nreimers_albert-small-v2",
    "stopword_path":"../../../data/common/stopword.dat",
    "resume_feature_path":"../../../data/v2/feature_engineering/resume_feature.csv",
    "resume_feature_emb_path":"../../../data/v2/feature_engineering/resume-feature-emb.csv",
}

def execute():
    model = loadBert()
    inpath = conf["resume_feature_path"]
    stopWord = getStopWord()
    df = pd.read_csv(inpath)

    cols = list(df.columns)
    for i in range(len(cols)) :
        cols[i] = cols[i].replace(":","").lower().strip()

    embCols = { i for i in [0,1,2,3,4,5,6,7,8] }
    datalist = []
    for row in tqdm(df.values) :
        try:
            record = {}
            for idx in range(len(row)):
                if idx in embCols :
                    emb = getEmb(row[idx], stopWord, model)
                    record[cols[idx]] = emb
                else:
                    record[cols[idx]] = row[idx]
            datalist.append(record)
        except Exception as e :
            traceback.print_exc()


    df = DataFrame(datalist)
    outpath = conf["resume_feature_emb_path"]
    df.to_csv(outpath , index=False)

    print("Extracted resume feature embedding information")


if __name__ == '__main__':
    execute()