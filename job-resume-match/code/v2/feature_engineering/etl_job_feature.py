# coding=UTF-8

import pandas as pd
from pandas import DataFrame
import re
import os

def createDir(filepath):
    parent_dir = os.path.dirname(filepath)
    if os.path.exists(parent_dir) == False :
        os.makedirs(parent_dir)

def getMapping():
    path = conf["feature_mapping_path"]
    featureNameMap = {}
    with open(path , "r") as file :
        line = file.readline()
        while line :
            line = line.strip()
            srr = line.split(":")
            if len(srr) == 2:
                feature_name = srr[0] +":"
                featureNameMap[feature_name] = "NAN"
            line = file.readline()
    return featureNameMap

def etlFeatureValue(s , pattern , featureNameMap):
    srr = s.split("\r\n")
    companyName = srr[0].strip()
    featureName = None
    tlist = None
    recordMap = featureNameMap.copy()
    recordMap["COMPANYNAME"] = companyName
    for line in srr[1:]:
        line = line.strip()
        arr = pattern.findall(line)
        if len(arr) != 0 :
            if featureName :
                recordMap[featureName] = "\r\n".join(tlist).replace("\r\n\r\n","\r\n")
            for i in arr[0] :
                if i != "" :
                    featureName = i
                    break
            tlist = [line[line.find(":")+1:].strip()]
        elif tlist :
            tlist.append(line)

    return recordMap

def execute():
    featureNameMap = getMapping()
    pattern = re.compile("(%s)" % ")|(".join(featureNameMap.keys()))
    datalist = []

    inpath = conf["job_raw_path"]
    df = pd.read_csv(inpath)
    id = 1
    for item in df.values :
        s = item[1]
        recordMap = etlFeatureValue(s , pattern , featureNameMap)
        recordMap["ID"] = id
        datalist.append(recordMap)
        id = id +1

    df = DataFrame(datalist)

    outpath = conf["job_feature_path"]
    createDir(outpath)
    df.to_csv(outpath , index=False)

    print("Extracted job feature information")

conf ={
    "feature_mapping_path":"feature.dat",
    "job_raw_path":"../../../data/common/data-job-posts.csv",
    "job_feature_path": "../../../data/v2/feature_engineering/job-feature.csv",
}

if __name__ == '__main__':
    execute()