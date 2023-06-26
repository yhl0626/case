import etl_resume_feature as step1
import etl_job_feature as step2
import create_job_feature_emb as step3
import create_resume_feature_emb as step4

def execute():
    step1.execute()
    step2.execute()
    step3.execute()
    step4.execute()

if __name__ == '__main__':
    execute()