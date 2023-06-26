# coding=UTF-8

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


# 得到文本信息
def getJob():
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
            element["JobRequirment"] = JobRequirment.lower()

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
def saveJobEmb1(rawJob , model) :
    save_path = conf["job_emb_path"]
    createDir(save_path)

    with open(save_path , "w") as file :
        pattern = "[^a-z]+"
        for job in tqdm(rawJob):
            id = job["id"]
            title = job["Title"].replace("\t","/").replace("\r","").replace("\n","")

            JobRequirment = job["JobRequirment"]

            arr = JobRequirment.replace(".", "- ").replace("\r\n", " ").split("- ")
            skills = []
            for skill in arr:
                skill = skill.strip()
                for skill_sub in skill.split(";"):
                    if skill_sub != "":
                        skill_sub = re.sub(pattern, " ", skill_sub)
                        skills.append(skill_sub)

            embeddings = model.encode(skills)
            emb = list(embeddings.mean(axis=0))

            for i in range(len(emb)):
                emb[i] = str(emb[i])

            record = [str(id), title, ",".join(emb)]
            record = "\t".join(record)
            file.write(record +"\n")
            file.flush()

# 得到职位的embedding，是JobRequirment的句子平均值
# @param rawJob：job的原始文件
# @param model：sbert模型
def saveJobEmb2(rawJob , model) :
    save_path = conf["job_emb_path_2"]
    createDir(save_path)

    with open(save_path , "w") as file :
        pattern = "[^a-z]+"
        for job in tqdm(rawJob):
            id = job["id"]
            title = job["Title"].replace("\t","/").replace("\r","").replace("\n","")

            JobRequirment = job["JobRequirment"]

            arr = JobRequirment.replace(".", "- ").replace("\r\n", " ").split("- ")[:6]
            skills = []
            for skill in arr:
                skill = skill.strip()
                for skill_sub in skill.split(";"):
                    if skill_sub != "":
                        skill_sub = re.sub(pattern, " ", skill_sub)
                        skills.append(skill_sub)

            embeddings = model.encode(skills)
            emb = list(embeddings.mean(axis=0))

            for i in range(len(emb)):
                emb[i] = str(emb[i])

            record = [str(id), title, ",".join(emb)]
            record = "\t".join(record)
            file.write(record +"\n")
            file.flush()

def saveResumeEmb1(rawResume , model) :
    save_path = conf["resume_emb_path"]
    createDir(save_path)

    with open(save_path , "a") as file :
        for resume in tqdm(rawResume):
            try:
                id = resume["id"]
                category = resume["category"]
                drr = resume["desc"]

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

def saveResumeEmb2(rawResume , model) :
    save_path = conf["resume_emb_path_2"]
    createDir(save_path)

    with open(save_path , "a") as file :
        for resume in tqdm(rawResume):
            try:
                id = resume["id"]
                category = resume["category"]
                drr = resume["desc"][:6]

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
    "model_path":"../../data/model/pretrain/nreimers_albert-small-v2",
    "job_raw_path":"../../data/common/data-job-posts.csv",
    "resume_raw_path":"../../data/common/data-resume.csv",
    "job_emb_path":"../../data/label/job-require-all-embedding-albert.dat",
    "job_emb_path_2":"../../data/label/job-require-all-embedding-albert-2.dat",
    "resume_emb_path":"../../data/label/resume-embedding-albert.dat",
    "resume_emb_path_2":"../../data/label/resume-embedding-albert-2.dat"
}

def execute():
    model = loadBert()
    rawJob = getJob()
    saveJobEmb1(rawJob, model)
    saveJobEmb2(rawJob, model)

    rawResume = getResume()
    saveResumeEmb1(rawResume, model)
    saveResumeEmb2(rawResume , model)

    print("End of extracting resume and position information")

if __name__ == '__main__':
    execute()