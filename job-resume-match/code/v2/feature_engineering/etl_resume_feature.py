# coding=UTF-8
import re

import pandas as pd
import re
from pandas import DataFrame
import os

def createDir(filepath):
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False :
        os.makedirs(parent_dir)

def parse(input):
    keywords = [
        "#Summary#",
        "#Highlights#",
        "#Accomplishments#",
        "#Experience#",
        "#Education#",
        "#Skills#",
        "#Qualifications#",
        # "#Career Focus#",
        # "#Core Qualifications#",
        # "#Education and Training#"
    ]
    t_input = re.sub("[ ]{2,}","#" , input)
    tmap = {}
    for k in keywords :
        tmap[k] = t_input.find(k)

    tlist = []
    for k , v in sorted(tmap.items() , key = lambda x : x[1] , reverse=False) :
        tlist.append((k ,v))

    _min = 100000
    last = len(tlist) -1
    for i in range(last) :
        k , v = tlist[i]
        k1 , v1 = tlist[i+1]
        if v != -1 :
            if _min > v:
                _min = v
            scope = (v , v1)
        else:
            scope = (v , -1)

        tlist[i] = (k , scope)

    k , v = tlist[last]
    tlist[last] = (k , (v , len(t_input)))

    tlist.append(("#Jobposition#" , (0 , _min)))

    record = {}
    for k , score in tlist :
        begin , end = score
        k = k.replace("#","")
        if begin > 0 :
            record[k] = t_input[begin + len(k) : end].replace("#","   ")
        elif begin == 0 :
            record[k] = t_input[begin: end].replace("#","   ")
        else:
            record[k] = "NAN"

    return record

def execute():
    inpath = "../../../data/common/data-resume.csv"
    outpath = "../../../data/v2/feature_engineering/resume_feature.csv"
    # E:\pycharmWorkspace\tf23\case\resume-job\resume-job-similartiy\data\data-resume.csv
    df = pd.read_csv(inpath)
    datalist = []
    for row in df.values:
        id = row[0]
        t = row[1]
        tmap = parse(t)
        tmap["id"] = id
        datalist.append(tmap)

    createDir(outpath)
    df = DataFrame(datalist)
    df.to_csv(outpath, index=False)

    print("Extracted resume feature information")

if __name__ == '__main__':
    execute()