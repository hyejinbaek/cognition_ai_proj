# openai version = 0.28.0
import os
from dotenv import load_dotenv
import openai
import sys
import io
import time

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# .env 파일 로드
load_dotenv()

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

openai.api_key = openai_api_key

# Chat 형식 데이터 파일 업로드
training_file = openai.File.create(
    file=open("./dataset/chat_output.jsonl", "rb"),  
    purpose="fine-tune"
)

# Fine-Tuning 작업 생성
fine_tune_job = openai.FineTuningJob.create(
    training_file=training_file['id'],  
    hyperparameters={
        "n_epochs": 4,
    },
    model = "gpt-4o-2024-08-06"
)

# Retrieve and print logs periodically
def print_fine_tune_logs(job_id):
    while True:
        result = openai.FineTuningJob.retrieve(id=job_id)
        status = result['status']
        print(f"Status: {status}")
        
        if status in ['running', 'pending']:
            logs = openai.FineTuningJobLogs.retrieve(id=job_id)
            for line in logs:
                print(line['message'])
                
            time.sleep(30)
        else:
            print("Fine-tuning job completed.")
            break

# 파인튜닝 작업 로그 출력 시작
print_fine_tune_logs(fine_tune_job['id'])

# 모델의 응답을 테스트할 데이터셋 준비
test_dataset = [
    {"role": "user", "content": "요청 : (비업무)개인시간_흡연 등, 요청 사유 : 화장실인 경우에 승인/거절/보류 어느것에 해당하니?"},
    {"role": "user", "content": "요청 : (비업무)개인시간_흡연 등, 요청 사유 : 담배인 경우에 승인/거절/보류 어느것에 해당하니?"},
    {"role": "user", "content": "요청 : (비업무)개인시간_흡연 등, 요청 사유 : 편의점인 경우에 승인/거절/보류 어느것에 해당하니?"},
    {"role": "user", "content": "요청 : (업무)회의, 요청 사유 : IoT 프로젝트 관련 모다플 본사 방문인 경우에 승인/거절/보류 어느것에 해당하니?"}
]

# 파인튜닝된 모델 평가 함수
def evaluate_model():
    for test_case in test_dataset:
        response = openai.ChatCompletion.create(
            model=fine_tune_job['fine_tuned_model'],
            messages=[test_case]
        )
        print("Input:", test_case["content"])
        print("Response:", response['choices'][0]['message']['content'])
        print("\n")

# 파인튜닝된 모델의 ID를 가져오기 위해 대기
while True:
    result = openai.FineTuningJob.retrieve(id=fine_tune_job['id'])
    if result['status'] == 'completed':
        fine_tune_job['fine_tuned_model'] = result['fine_tuned_model']
        print("Fine-tuned model ID:", fine_tune_job['fine_tuned_model'])
        evaluate_model()
        break
    elif result['status'] == 'failed':
        print("Fine-tuning failed.")
        break
    else:
        print("Waiting for fine-tuning to complete...")
        time.sleep(30)