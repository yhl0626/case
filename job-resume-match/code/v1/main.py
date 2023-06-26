# coding=UTF-8
import train_test_validate as step1
import fine_tuning as step2
import evaluate as step3
import create_finetuning_data_emb as step4
import topKJob as step5
import topKResume as step6

def execute():
    step1.execute()
    step2.execute()
    step3.execute()
    step4.execute()
    step5.execute()
    step6.execute()

if __name__ == '__main__':
    execute()