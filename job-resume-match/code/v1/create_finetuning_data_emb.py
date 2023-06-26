# coding=UTF-8

# 基于fine-tuning模型生成embedding
import pandas as pd
import re
import os
import traceback
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# 提取职位的岗位要求信息
# 提取简历信息

def createDir(filepath) :
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False :
        os.makedirs(parent_dir)

# 加载模型
def loadBert():
    model_path = conf["model_path"]
    model = SentenceTransformer(model_path)
    return model

def getStopWord():
    inpath = conf["stopword_path"]
    stopWord = {}
    with open(inpath, "r") as file:
        line = file.readline()
        while line:
            line = line.strip()
            if len(line) > 1:
                stopWord[line] = None
            line = file.readline()
    return stopWord


# 得到文本信息
def getJob():
    path = "../../../data/data-job-posts-1.csv"
    path = conf["job_raw_path"]
    df = pd.read_csv(path )

    # Title:职位名称, JobDescription：职位描述 , JobRequirment：职位需求
    datainfo = []
    for row in df.values :
        id = row[0]
        title = row[3]
        JobDescription = row[12]
        JobRequirment = row[13]
        if isinstance(title , float) == False and isinstance(JobDescription  , float) == False \
            and isinstance(JobRequirment , float) == False:

            element = {}
            element["id"] = id
            element["Title"] = title.lower()
            # element["JobDescription"] = JobDescription
            element["JobRequirment"] =  JobRequirment.lower()
            datainfo.append(element)

    return datainfo

# 得到简历信息
def getResume():
    path = conf["resume_raw_path"]
    df = pd.read_csv(path)

    datainfo = []
    for row in df.values:
        element = {}
        id = row[0]
        category = row[-1]

        if isinstance(category , float) == False :
            arr = row[1].lower().replace("\r", "").replace("\n", "").split(".")
            desc = []
            for item in arr:
                item = re.sub("[^a-z]", " ", item)
                item = re.sub("[ ]+", " ", item).strip()

                if item != "" and len(item) > 1:
                    desc.append(item)
            element["id"] = id
            element["desc"] = desc
            element["category"] = category
            datainfo.append(element)

    return datainfo

# 得到职位的embedding，是JobRequirment的句子平均值
# @param rawJob：job的原始文件
# @param model：sbert模型
def saveJobEmb(rawJob , model , stopword) :
    save_path = conf["job_emb_path"]
    createDir(save_path)

    with open(save_path , "w") as file :
        pattern = "[^a-z]+"
        for job in tqdm(rawJob):
            try:
                id = job["id"]
                title = job["Title"].replace("\t", "/").replace("\r", "").replace("\n", "")

                JobRequirment = job["JobRequirment"]

                arr = JobRequirment.replace(".", "- ").replace("\r\n", " ").split("- ")
                skills = []
                for skill in arr:
                    skill = skill.strip()
                    for skill_sub in skill.split(";"):
                        if skill_sub != "":
                            skill_sub = re.sub(pattern, " ", skill_sub)
                            skills.append(skill_sub)

                for idx in range(len(skills)):
                    sentence = skills[idx]
                    tlist = []
                    for word in sentence.split(" "):
                        if word not in stopword:
                            tlist.append(word)

                    skills[idx] = " ".join(tlist)

                embeddings = model.encode(skills)
                emb = list(embeddings.mean(axis=0))

                for i in range(len(emb)):
                    emb[i] = str(emb[i])

                record = [str(id), title, ",".join(emb)]
                record = "\t".join(record)
                file.write(record + "\n")
                file.flush()
            except Exception as e :
                traceback.print_exc()


def saveResumeEmb(rawResume , model , stopword ) :
    save_path = conf["resume_emb_path"]
    createDir(save_path)

    with open(save_path , "a") as file :
        for resume in tqdm(rawResume):
            try:
                id = resume["id"]
                category = resume["category"]
                drr = resume["desc"]

                for idx in range(len(drr)) :
                    sentence = drr[idx]
                    tlist = []
                    for word in sentence.split(" "):
                        if word not in stopword:
                            tlist.append(word)
                    drr[idx] = " ".join(tlist)

                embeddings = model.encode(drr)
                emb = list(embeddings.mean(axis=0))

                for i in range(len(emb)):
                    emb[i] = str(emb[i])

                record = [str(id), category, ",".join(emb)]
                record = "\t".join(record)
                file.write(record + "\n")
                file.flush()

            except Exception  as e :
                traceback.print_exc()

conf = {
    "model_path": "../../data/model/v1/sbert-resume-job-32X32.bin",
    "stopword_path":"../../data/common/stopword.dat",
    "job_raw_path":"../../data/common/data-job-posts.csv",
    "resume_raw_path":"../../data/common/data-resume.csv",
    "job_emb_path": "../../data/v1/predict/emb/job-embedding-1.dat",
    "resume_emb_path": "../../data/v1/predict/emb/resume-embedding-1.dat",
}

def execute():
    model = loadBert()
    jobinfo = getJob()
    resumeinfo = getResume()
    stopword = getStopWord()
    saveJobEmb(jobinfo, model, stopword)
    saveResumeEmb(resumeinfo, model, stopword)

    print("created fine-tuning embedding")

if __name__ == '__main__':
    execute()