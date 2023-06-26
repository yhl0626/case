# coding=UTF-8
import os
import pandas as pd
import xgboost as xgb

def getLabel(inpath):
    datamap = {}
    with open(inpath , "r") as file:
        line = file.readline()
        while line :
            line = line.strip()
            srr = line.split("\t")
            if len(srr) == 3:
                label = float(srr[0])
                resumeId = int(srr[1])
                jobId = int(srr[2])
                datamap[(resumeId , jobId)] = label
            line = file.readline()
    return datamap

def getJobEmb(trainMap , validMap , testMap , featureNum):
    jobIds = {}

    for resumeId , jobId in trainMap:
        jobIds[jobId] = None

    for resumeId , jobId in validMap:
        jobIds[jobId] = None

    for resumeId , jobId in testMap:
        jobIds[jobId] = None

    jobEmbPath = conf["jobEmbPath"]
    embCols = {i for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12,  15, 16, 17, 18, 19, 20, 21, 23, 24, 25]}
    df = pd.read_csv(jobEmbPath)

    for row in df.values :
        id = row[-1]
        if id in jobIds :
            jobIds[id] = []

            for idx in embCols :
                for v in row[idx].split(",") :
                    jobIds[id].append(float(v))
                # jobIds[id] = row[idx]

    jobFeatureIdxMapping = {}
    count = featureNum
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25]:
        column_name = "job_%s" % df.columns[i]
        if column_name not in jobFeatureIdxMapping:
            jobFeatureIdxMapping[column_name] = []
        for i in range(768) :
            jobFeatureIdxMapping[column_name].append('f%d'%count)
            count = count + 1

    return jobIds , jobFeatureIdxMapping , count

def getResumeEmb(trainMap , validMap , testMap , featureNum):
    resumeIds = {}

    for resumeId , jobId in trainMap:
        resumeIds[resumeId] = None

    for resumeId , jobId in validMap:
        resumeIds[resumeId] = None

    for resumeId , jobId in testMap:
        resumeIds[resumeId] = None

    resumeEmbPath = conf["resumeEmbPath"]
    embCols = {i for i in [0,1,2,3,4,5,6,7]}
    df = pd.read_csv(resumeEmbPath)

    for row in df.values :
        id = row[-1]
        if id in resumeIds :
            resumeIds[id] = []

            for idx in embCols :
                for v in row[idx].split(",") :
                    resumeIds[id].append(float(v))

    resumeFeatureIdxMapping = {}
    count = featureNum
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        column_name = "resume_%s" %df.columns[i]
        if column_name not in resumeFeatureIdxMapping:
            resumeFeatureIdxMapping[column_name] = []
        for i in range(768):
            resumeFeatureIdxMapping[column_name].append('f%d' % count)
            count = count + 1

    return resumeIds ,resumeFeatureIdxMapping , count

def train(trainMap , validMap , jobEmb , resumeEmb , jobFeatureIdxMapping , resumeFeatureIdxMapping):
    train_features = []
    train_labels = []
    valid_features = []
    valid_labels = []
    for resumeId , jobId in trainMap :
        label = trainMap[(resumeId , jobId)]

        feature = []
        for v in resumeEmb[resumeId] :
            feature.append(v)

        for v in jobEmb[jobId] :
            feature.append(v)

        train_features.append(feature)
        train_labels.append(label)

    for resumeId , jobId in validMap :
        label = validMap[(resumeId , jobId)]

        feature = []
        for v in resumeEmb[resumeId] :
            feature.append(v)

        for v in jobEmb[jobId] :
            feature.append(v)

        valid_features.append(feature)
        valid_labels.append(label)

    data_train = xgb.DMatrix(train_features , train_labels)
    data_valid = xgb.DMatrix(valid_features, valid_labels)
    param = {'max_depth': 5, 'eta': 0.1, 'silent': 1, 'subsample': 0.7, 'colsample_bytree': 0.7,
             'objective': 'binary:logistic'}

    # 设定watchlist用于查看模型状态
    watchlist = [(data_valid, 'eval'), (data_train, 'train')]
    num_round = 10
    bst = xgb.train(param, data_train, num_round, watchlist)

    print("\n")

    # # 使用模型预测
    # preds = bst.predict(data_valid)

    # # 判断准确率
    # labels = data_valid.get_label()
    # print('错误类为%f' % \
    #       (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

    # 计算特征重要性
    feature_scores = bst.get_score()

    feature_weights = []
    for feature_name in resumeFeatureIdxMapping:
        tlist = []
        for k in feature_scores :
            if k in resumeFeatureIdxMapping[feature_name] :
                v = feature_scores[k]
                tlist.append(v)

        feature_weights.append((feature_name ,sum(tlist) / len(resumeFeatureIdxMapping[feature_name]) ))

    for feature_name in jobFeatureIdxMapping:
        tlist = []
        for k in feature_scores :
            if k in jobFeatureIdxMapping[feature_name] :
                v = feature_scores[k]
                tlist.append(v)

        feature_weights.append((feature_name ,sum(tlist) / len(jobFeatureIdxMapping[feature_name]) ))

    feature_weights.sort(key=lambda x: x[1], reverse=True)
    for feature_name , weight in feature_weights :
        print(feature_name+"：" , weight)

conf = {
    "train_path": "../../data/train/v1/train.dat",
    "valid_path": "../../data/train/v1/valid.dat",
    "test_path": "../../data/train/v1/test.dat",
    "jobEmbPath": "../../data/v2/feature_engineering/job-feature-emb.csv",
    "resumeEmbPath": "../../data/v2/feature_engineering/resume-feature-emb.csv",

}

def exeute():
    train_path = conf["train_path"]
    valid_path = conf["valid_path"]
    test_path = conf["test_path"]

    trainMap = getLabel(train_path)
    validMap = getLabel(valid_path)
    testMap = getLabel(test_path)

    jobEmb , jobFeatureIdxMapping , featureNum = getJobEmb(trainMap, validMap, testMap , 0 )
    resumeEmb , resumeFeatureIdxMapping , featureNum = getResumeEmb(trainMap, validMap, testMap , featureNum)

    train(trainMap, testMap, jobEmb, resumeEmb , jobFeatureIdxMapping , resumeFeatureIdxMapping)


if __name__ == '__main__':
    exeute()
