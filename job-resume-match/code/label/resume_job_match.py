# coding=UTF-8

import faiss
import numpy as np

# 基于文本的简历-岗位匹配
# 1、得到语义信息：sbert -> embedding
# 2、语义匹配：使用faiss完成
# 3、保存结果

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
    print(index.is_trained)
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
        print(t)

def jobResumeMatchId(jobIds , jobTitles , jobEmb , resumeIds , resumeTitles , resumeEmb):
    dimension = 768
    index = faiss.IndexFlatL2(dimension)
    print(index.is_trained)
    index.add(jobEmb)

    k = 5
    D, I = index.search(resumeEmb, k)
    length = len(resumeEmb)
    outpath = conf["resuem_job_match_path"]
    with open(outpath , "w") as file :
        for idx in range(length):
            rid = resumeIds[idx]
            resume_title = resumeTitles[rid]

            tlist = []
            for jid, score in zip(I[idx], D[idx]):
                _jid = jobIds[jid]
                job_title = jobTitles[_jid]
                tlist.append(str(_jid) + ":" + job_title + ":" + str(score))

            t = str(rid) + "\t" + resume_title + "\t" + ",".join(tlist)+"\n"
            file.write(t)
            file.flush()

def map2str(map):
    tlist = []
    for k , v in  sorted(map.items(), key=lambda d: d[1], reverse=True):
        tlist.append(k + ":" + str(v))
    return ",".join(tlist)

conf = {
    "job_emb_path":"../../data/label/job-require-all-embedding-albert.dat",
    "resume_emb_path":"../../data/label/resume-embedding-albert.dat",
    "resuem_job_match_path":"../../data/label/mapping.dat",
}

def execute():
    jobIds, jobTitles, jobEmb = getJobEmb()
    resumeIds, resumeTitles, resumeEmb = getResumeEmb()

    jobResumeMatchId(jobIds, jobTitles, jobEmb, resumeIds, resumeTitles, resumeEmb)
    print("Resume job matching completed")

if __name__ == '__main__':
    execute()