# coding=UTF-8

import pandas as pd
import random
import os
import re

# 创建可以训练的标注数据
# 方法：语义相似度匹配+人工标注
def createParentDir(filepath):
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False :
        os.makedirs(parent_dir)

# 得到岗位分类信息（人工标注因为时间有限，仅仅标注部分信息）
def getJobCategory():
    inpath = conf["job_category_path"]

    cateinfo = {}
    with open(inpath , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            srr =line.split("\t")
            category = srr[0]
            jobname = srr[1]

            if category not in cateinfo :
                cateinfo[category] = {}
            cateinfo[category][jobname] = None
            line = file.readline()

    return cateinfo

# 得到可训练的数据（正样本：求职者匹配简历（人工标注），负样本：求职者不匹配简历（随机采样））
# @param cateinfo：岗位分类信息
def getMapping(cateinfo):
    # 创建正样本
    poslist = []
    resume_job_match_path = conf["resume_job_match_path"]

    with open(resume_job_match_path , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            srr = line.split("\t")
            resumeId = srr[0]
            category = srr[1]
            jobs = srr[2].split(",")

            if category in cateinfo:
                for item in jobs:
                    trr = item.split(":")
                    jobId = trr[0]
                    jobName = "".join(trr[1:-1])

                    if jobName in cateinfo[category] :
                        poslist.append((resumeId , jobId))
                        break

            line = file.readline()

    # 创建负样本（与正样本是1:2）
    neglist = []
    job_raw_path = conf["job_raw_path"]

    df = pd.read_csv(job_raw_path)
    jobInfo = {}
    for record in df.values:
        id = str(record[0])
        jobName = record[3]
        if isinstance(jobName , float) == False :
            jobName = jobName.lower().replace("\t","/").replace("\r","").replace("\n","")
            jobName = re.sub("[ ]+", " ", jobName)
            jobInfo[id] = jobName

    # 随机抽取一个作为负样本，正样本和负样本的比例是1:3
    ids = list(jobInfo.keys())
    for resumeId , jobId in poslist:
        for i in range(3) :
            negJobId = random.choice(ids)
            neglist.append((resumeId, negJobId))

    # 保留原始训练集
    outpath = conf["train_data_path"]
    createParentDir(outpath)
    with open(outpath , "w") as file :
        for  resumeId , jobId in poslist:
            line = ["1" , resumeId , jobId]
            line = "\t".join(line)
            file.write(line + "\n")
            file.flush()

        for  resumeId , jobId in neglist:
            line = ["0" , resumeId , jobId]
            line = "\t".join(line)
            file.write(line + "\n")
            file.flush()

conf = {
    "job_category_path":"../../data/label/job_category.dat",
    "resume_job_match_path":"../../data/label/mapping.dat",
    "job_raw_path":"../../data/common/data-job-posts.csv",
    "train_data_path":"../../data/train/train-raw.dat",
}

def execute():
    cateinfo = getJobCategory()
    getMapping(cateinfo)

    print("End of creating training set")

if __name__ == '__main__':
    execute()