# coding=UTF-8

import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer, InputExample, losses , evaluation
from torch.utils.data import DataLoader

def createDir(filepath) :
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False:
        os.makedirs(parent_dir)

def loadModel():
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

# 得到数据
def getData(train_path, job_path, resume_path, stopword):
    resumeIds = {}
    jobIds = {}
    data = {}
    with open(train_path , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            if len(line) > 1 :
                # label , resumeId , jobId
                srr = line.split("\t")
                label = srr[0]
                resumeId = srr[1]
                jobId = srr[2]

                resumeIds[resumeId] = None
                jobIds[jobId] = None
                data[(resumeId , jobId)] = float(label)
            line = file.readline()

    # 提取job文本信息
    df = pd.read_csv(job_path)
    for record in df.values :
        id = str(record[0])
        JobRequirment = record[13]
        if id in jobIds and isinstance(JobRequirment , float) == False:
            arr = JobRequirment.lower().replace("\r", "").replace("\n", "").replace(";" , ".").split(".")
            desc = []
            for item in arr:
                item = re.sub("[^a-z]", " ", item)
                item = re.sub("[ ]+", " ", item).strip()
                srr = item.split(" ")
                for i in srr :
                    i = i.strip()
                    if i != "" and i not in stopword:
                        desc.append(i)

            desc = " ".join(desc[:top_token_num])
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

            desc = " ".join(desc[:top_token_num])
            resumeIds[id] = desc

    # 得到样本数据（sbert模型要求格式）
    examples = []
    for resumeId , jobId in data :
        label = data[(resumeId , jobId)]

        if resumeId in resumeIds and jobId in jobIds :
            resumeInfo = resumeIds[resumeId]
            jobInfo = jobIds[jobId]

            if resumeInfo is not None and jobInfo is not None:
                examples.append(InputExample(texts=[resumeInfo, jobInfo], label=label))

    return examples

def getValid(train_path, job_path, resume_path, stopword):
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

            desc = " ".join(desc[:top_token_num])
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

            desc = " ".join(desc[:top_token_num])
            resumeIds[id] = desc

    # 得到样本数据（sbert模型要求格式）
    senetence1 = []
    senetence2 = []
    labels =[]
    for resumeId , jobId in data :
        label = data[(resumeId , jobId)]

        if resumeId in resumeIds and jobId in jobIds :
            resumeInfo = resumeIds[resumeId]
            jobInfo = jobIds[jobId]

            if resumeInfo is not None and jobInfo is not None:
                senetence1.append(resumeInfo)
                senetence2.append(jobInfo)
                labels.append(label)
    return senetence1 , senetence2 , labels


def fineTuning(model):
    train_path = conf["train_path"]
    valid_path = conf["valid_path"]
    job_path = conf["job_path"]
    resume_path = conf["resume_path"]
    model_path = conf["train_model_path"]

    createDir(model_path)
    stopword = getStopWord()
    train_examples = getData(train_path, job_path, resume_path, stopword)
    sentence1 , sentence2 , labels = getValid(valid_path, job_path, resume_path, stopword)

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentence1, sentence2, labels , write_csv= True)
    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],  evaluator= evaluator, epochs=1, warmup_steps=100)

    model.save(model_path)

top_token_num = 32
conf = {
    "top_token_num":top_token_num,
    "model_path":"../../data/model/pretrain/nreimers_albert-small-v2",
    "stopword_path":"../../data/common/stopword.dat",
    "train_path":"../../data/train/v1/train.dat",
    "valid_path":"../../data/train/v1/valid.dat",
    "job_path":"../../data/common/data-job-posts.csv",
    "resume_path":"../../data/common/data-resume.csv",
    "train_model_path":"../../data/model/v1/sbert-resume-job-%dX%d.bin" % (top_token_num,top_token_num),
}

def execute():
    model = loadModel()
    fineTuning(model)

    print("end of fine-tuning")

if __name__ == '__main__':
    execute()
