# coding=UTF-8

import etl_job_resume_text as step1
import resume_job_match as step2
import create_train_data as step3

def execute():
    # 将文本转变成embedd
    # ing的形式
    step1.execute()
    # 找出简历和岗位匹配的信息，这个是通过faiss来实现的
    step2.execute()
    # 生成可以训练的数据
    step3.execute()

if __name__ == '__main__':
    execute()