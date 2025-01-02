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
d = data[['요청', '요청 사유', '상태']]

# 상태 카테고리 정리
status_mapping = {'승인됨': 0, '거절됨': 1}
d['상태'] = d['상태'].map(status_mapping)

# '상태'가 NaN인 행 제거 (승인됨, 거절됨 외의 값 제거)
d = d.dropna(subset=['상태'])

# 요청 데이터 정리
d['요청'] = d['요청'].apply(lambda x: x.split("/")[-1].strip() if pd.notnull(x) else x)

valid_requests = ['(비업무)개인시간_흡연 등', '(업무)회의', '(업무)기타업무', '(업무)출장,이동,외근']
d = d[d['요청'].isin(valid_requests)]
print(d)