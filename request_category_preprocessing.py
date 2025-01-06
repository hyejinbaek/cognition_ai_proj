# '요청' 카테고리 '/' 기준 자르기

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # 모델 저장 및 불러오기 위해 joblib 사용

# 요청, 요청 사유, 상태

# 1. 엑셀 파일 읽기
file_path = './dataset/sample.xlsx' 
data = pd.read_excel(file_path)
data = data.dropna(subset=['상태'])
data['요청'] = data['요청'].apply(lambda x: x.split("/")[-1].strip() if pd.notnull(x) else x)

valid_requests = ['(비업무)개인시간_흡연 등', '(업무)회의', '(업무)기타업무', '(업무)출장,이동,외근']
d = data[data['요청'].isin(valid_requests)]
print(d)


output_path = './dataset/processed_sample.xlsx'
d.to_excel(output_path, index=False)

print(f"Processed data saved to {output_path}")