# coding=UTF-8

import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer, InputExample, losses , evaluation , util

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score , confusion_matrix

def loadModel(model_path):
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

def getData(train_path, job_path, resume_path, stopword):
    resumeIds = {}
    jobIds = {}
    data = {}
    with open(train_path, "r") as file:
        line = file.readline()
        while line:
            line = line.strip()
            if len(line) > 1:
                # label , resumeId , jobId
                srr = line.split("\t")
                label = srr[0]
                resumeId = srr[1]
                jobId = srr[2]

                resumeIds[resumeId] = None
                jobIds[jobId] = None
                data[(resumeId, jobId)] = float(label)
            line = file.readline()

    # 提取job文本信息
    df = pd.read_csv(job_path)
    for record in df.values:
        id = str(record[0])
        JobRequirment = record[13]
        if id in jobIds and isinstance(JobRequirment, float) == False:
            arr = JobRequirment.lower().replace("\r", "").replace("\n", "").replace(";", ".").split(".")
            desc = []
            for item in arr:
                item = re.sub("[^a-z]", " ", item)
                item = re.sub("[ ]+", " ", item).strip()
                srr = item.split(" ")
                for i in srr:
                    i = i.strip()
                    if i != "" and i not in stopword:
                        desc.append(i)

            desc = " ".join(desc[:32])
            jobIds[id] = desc

    # 提取resume文本数据
    df = pd.read_csv(resume_path)
    for record in df.values:
        id = str(record[0])
        content = record[1]
        if id in resumeIds and isinstance(content, float) == False:
            arr = content.lower().replace("\r", "").replace("\n", "").replace(";", ".").split(".")
            desc = []
            for item in arr:
                item = re.sub("[^a-z]", " ", item)
                item = re.sub("[ ]+", " ", item).strip()
                srr = item.split(" ")
                for i in srr:
                    i = i.strip()
                    if i != "" and i not in stopword:
                        desc.append(i)

            desc = " ".join(desc[:32])
            resumeIds[id] = desc

    # 得到样本数据（sbert模型要求格式）
    embedding1 = []
    embedding2 = []
    labels = []
    for resumeId , jobId in data :
        label = data[(resumeId , jobId)]

        if resumeId in resumeIds and jobId in jobIds :
            resumeInfo = resumeIds[resumeId]
            jobInfo = jobIds[jobId]

            if resumeInfo is not None and jobInfo is not None:
                embedding1.append(resumeInfo)
                embedding2.append(jobInfo)
                labels.append(label)
    return embedding1 , embedding2 , labels

def _evaluate(embedding1, embedding2, labels , model , threshold=0.5):
    emb1 = model.encode(embedding1)
    emb2 = model.encode(embedding2)

    pred = []
    predInt = []
    for e1 , e2  in zip(emb1 , emb2):
        consine = util.cos_sim(e1 , e2).numpy()[0][0]
        pred.append(consine)

        predInt.append((int)(1 if consine > threshold else 0))

    auc_score = roc_auc_score(labels, pred )
    precision = precision_score(labels, predInt,)
    recall = recall_score(labels, predInt)
    f1 = f1_score(labels, predInt)
    return auc_score , precision , recall , f1

def evaluate():
    # test_path = "../../../data/train/v1/test.dat"
    # job_path = "../../../data/data-job-posts-1.csv"
    # resume_path = "../../../data/data-resume.csv"
    #
    # model_path_32 = "../../../data/model/v1/sbert-resume-job-32X32.bin"
    # model_path_64 = "../../../data/model/v1/sbert-resume-job-64X64.bin"
    # model_path_128 = "../../../data/model/v1/sbert-resume-job-128X128.bin"
    # model_path_256 = "../../../data/model/v1/sbert-resume-job-256X256.bin"

    test_path = conf["test_path"]
    job_path = conf["job_path"]
    resume_path = conf["resume_path"]
    model_path_32 = conf["model_path_32"]
    model_path_64 = conf["model_path_64"]
    model_path_128 = conf["model_path_128"]
    model_path_256 = conf["model_path_256"]

    model_32 = loadModel(model_path_32)
    model_64 = loadModel(model_path_64)
    model_128 = loadModel(model_path_128)
    model_256 = loadModel(model_path_256)
    print("\n")

    stopword = getStopWord()
    embedding1, embedding2, labels = getData(test_path, job_path, resume_path, stopword )

    # threshold = 0.43
    # for i in range(10) :
    #     threshold = threshold + 0.005
    #     auc_score, precision, recall, f1 = _evaluate(embedding1, embedding2, labels, model_32,threshold)
    #     print("### modelX32 auc", threshold , auc_score)
    #     print("### modelX32 precision", threshold , precision)
    #     print("### modelX32 recall", threshold , recall)
    #     print("### modelX32 f1", threshold , f1)
    #     print("")

    pass
    auc_score, precision, recall, f1 = _evaluate(embedding1, embedding2, labels, model_32, 0.46)
    print("### modelX32 auc",auc_score)
    print("### modelX32 precision", precision)
    print("### modelX32 recall", recall)
    print("### modelX32 f1", f1)
    print("\n")

    auc_score, precision, recall, f1 = _evaluate(embedding1, embedding2, labels, model_64 , 0.426)
    print("### modelX64 auc", auc_score)
    print("### modelX64 precision", precision)
    print("### modelX64 recall", recall)
    print("### modelX64 f1", f1)
    print("\n")

    auc_score, precision, recall, f1 = _evaluate(embedding1, embedding2, labels, model_128 , 0.33)
    print("### modelX128 auc", auc_score)
    print("### modelX128 precision", precision)
    print("### modelX128 recall", recall)
    print("### modelX128 f1", f1)
    print("\n")

    auc_score, precision, recall, f1 =  _evaluate(embedding1, embedding2, labels, model_256 , 0.62)

    print("### modelX256 auc", auc_score)
    print("### modelX256 precision", precision)
    print("### modelX256 recall", recall)
    print("### modelX256 f1", f1)

conf = {
    "stopword_path":"../../data/common/stopword.dat",
    "test_path": "../../data/train/v1/test.dat",
    "job_path": "../../data/common/data-job-posts.csv",
    "resume_path": "../../data/common/data-resume.csv",
    "model_path_32": "../../data/model/v1/sbert-resume-job-32X32.bin",
    "model_path_64": "../../data/model/v1/sbert-resume-job-64X64.bin",
    "model_path_128": "../../data/model/v1/sbert-resume-job-128X128.bin",
    "model_path_256": "../../data/model/v1/sbert-resume-job-256X256.bin",
}

def execute():
    evaluate()

if __name__ == '__main__':
    execute()