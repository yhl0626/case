# coding=UTF-8

import faiss
import os
import numpy as np

# 向求雇主推荐求职者列表
def createDir(filepath):
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False :
        os.makedirs(parent_dir)

def getJobEmb():
    path = conf["job_emb_path"]
    ids = []
    titles = {}
    embeddings = []
    with open(path , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            srr = line.split("\t")
            id = int(srr[0])
            title = srr[1]
            emb = srr[2]

            ids.append(id)
            titles[id] = title
            embeddings.append(emb.split(","))
            line = file.readline()

    return ids , titles , np.array(embeddings).astype(np.float32)

def getResumeEmb():
    path = conf["resume_emb_path"]
    ids = []
    titles = {}
    embeddings = []
    with open(path , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            srr = line.split("\t")
            id = int(srr[0])
            title = srr[1].lower()
            emb = srr[2].split(",")

            ids.append(id)
            titles[id] = title
            embeddings.append(emb)
            line = file.readline()

    return ids,  titles , np.array(embeddings).astype(np.float32)

def jobResumeMatch(jobIds , jobTitles , jobEmb , resumeIds , resumeTitles , resumeEmb):
    dimension = 768
    index = faiss.IndexFlatL2(dimension)
    index.add(jobEmb)

    k = 5
    D, I = index.search(resumeEmb, k)
    length = len(resumeEmb)
    for idx in range(length) :
        rid = resumeIds[idx]
        resume_title = resumeTitles[rid]

        tmap = {}
        for jid in I[idx]:
            job_title = jobTitles[jobIds[jid]]
            for item in job_title.split(",") :
                item = item.strip()
                if item is not tmap :
                    tmap[item] = 1
                tmap[item] = tmap[item] + 1

        t = resume_title + "\t" + map2str(tmap)

def jobResumeMatchId(jobIds , jobTitles , jobEmb , resumeIds , resumeTitles , resumeEmb):
    dimension = 768
    index = faiss.IndexFlatIP(dimension)
    index.add(resumeEmb)

    k = k_num
    D, I = index.search(jobEmb, k)
    length = len(jobEmb)
    outpath = conf["resuem_job_match_path"]
    with open(outpath , "w") as file :
        for idx in range(length):
            rid = jobIds[idx]
            job_title = jobTitles[rid]

            tlist = []
            for jid, score in zip(I[idx], D[idx]):
                _jid = resumeIds[jid]
                tlist.append(str(_jid)  + ":" + str(score))

            t = str(rid) + "\t" + job_title + "\t" + ",".join(tlist)+"\n"
            file.write(t)
            file.flush()

def map2str(map):
    tlist = []
    for k , v in  sorted(map.items(), key=lambda d: d[1], reverse=True):
        tlist.append(k + ":" + str(v))
    return ",".join(tlist)

k_num = 20
conf = {
    "job_emb_path":"../../data/v1/predict/emb/job-embedding-1.dat",
    "resume_emb_path":"../../data/v1/predict/emb/resume-embedding-1.dat",
    "resuem_job_match_path":"../../data/v1/predict/topk/job-resume-match.dat",
}

def execute():
    jobIds, jobTitles, jobEmb = getJobEmb()
    resumeIds, resumeTitles, resumeEmb = getResumeEmb()

    jobResumeMatchId(jobIds, jobTitles, jobEmb, resumeIds, resumeTitles, resumeEmb)
    print("End of recommended resumes")

if __name__ == '__main__':
    execute()