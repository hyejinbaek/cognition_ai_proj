# RandomForestClassifier
# 내용 분석 필요

'''
1. Precision, Recall, F1-Score 개념
- Precision: 모델이 특정 클래스를 "맞았다"고 할 때, 실제로 맞는 경우의 비율
- Recall: 실제로 해당 클래스인 데이터 중에서 모델이 맞췄는지의 비율
- F1-Score: Precision과 Recall의 조화 평균. 두 지표 간 균형을 확인
- Support: 각 클래스의 실제 샘플 수

2. 클래스별 성능
a) 거절됨 클래스
- Precision: 1.00
    → 모델이 "거절됨"이라고 예측한 모든 경우가 정확히 맞음
- Recall: 0.61
    → 실제 "거절됨" 데이터의 61%만 제대로 분류
- F1-Score: 0.76
    → Precision과 Recall 간의 균형이 상대적으로 떨어짐
- 문제점: Recall이 낮음. 즉, 실제 "거절됨" 데이터를 많이 놓쳤다는 의미

b) 승인됨 클래스
- Precision: 0.97
    → "승인됨"이라고 예측한 데이터 중 97%가 정확
- Recall: 1.00
    → 실제 "승인됨" 데이터를 모두 올바르게 분류
- F1-Score: 0.99
    → 매우 높은 성능
- 장점: Precision과 Recall이 모두 높아서 해당 클래스의 예측이 안정적임

c) 취소됨 클래스
- Precision: 1.00
    → "취소됨"이라고 예측한 데이터는 정확했음
- Recall: 0.02
    → 실제 "취소됨" 데이터의 2%만 올바르게 분류
- F1-Score: 0.03
    → Precision과 Recall 간의 불균형으로 인해 성능이 매우 낮음
- 문제점: Recall이 극도로 낮아서 대부분의 "취소됨" 데이터를 놓침

3. 전체 성능
a) Accuracy: 0.97
- 전체 데이터 중 약 97%를 정확히 분류
- 하지만 Accuracy만으로 성능을 평가하기는 어려움. 특히 데이터가 불균형한 경우, 정확도는 높아도 일부 클래스가 제대로 예측되지 않을 수 있음

b) Macro Avg
- Precision: 0.99
- Recall: 0.54
- F1-Score: 0.59
    → 각 클래스별 성능을 평균낸 값
    → Recall이 낮아 전체적인 성능이 떨어짐
    
c) Weighted Avg
- Precision: 0.97
- Recall: 0.97
- F1-Score: 0.97
    → 데이터의 클래스 비율을 가중치로 계산한 값
    → "승인됨" 클래스가 대부분(7553/7920)이라 전체 성능이 높아 보임

4. 데이터 불균형 문제
- 승인됨 클래스의 샘플 수(7553)가 대부분이고, 취소됨(114)과 거절됨(253)은 소수임
- 이로 인해 모델이 승인됨에 편향된 경향을 보임
- 취소됨 클래스의 Recall이 매우 낮아 해당 데이터를 거의 예측하지 못함

5. 개선 방안
    1. 데이터 불균형 해결
    - 취소됨과 거절됨 클래스의 데이터를 증강(SMOTE, 오버샘플링)하거나, 승인됨 데이터를 줄이는 방식(언더샘플링) 적용
    2. 가중치 조정
    - 모델 학습 시 클래스별 샘플 불균형을 고려해 class_weight='balanced' 옵션 추가
    3. 특징 공학
    - 모델이 소수 클래스(취소됨, 거절됨)를 더 잘 예측하도록 추가 특징(feature)을 설계
    4. 다른 모델 사용
    - 랜덤 포레스트 대신 XGBoost나 LightGBM 같은 부스팅 모델을 사용해 소수 클래스 예측 성능 개선 시도

6. 결론
- 전체적으로 높은 Accuracy(0.97)를 보이지만, 소수 클래스의 Recall이 낮아 개선 필요
- 불균형 데이터를 처리하고, Recall을 높이는 방향으로 모델 개선 작업이 권장
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


file_path = './dataset/sample.xlsx'
data = pd.read_excel(file_path)

data['요청 종류'] = data['요청 종류'].apply(lambda x: x.split("/")[-1].strip() if pd.notnull(x) else x)

data['요청_사유_길이'] = data['요청 사유'].apply(len)


data['중복_여부'] = data['승인권자 노트'].apply(lambda x: 1 if '중복' in str(x) else 0)

X = data[['요청 종류', '요청_사유_길이', '중복_여부']] 
y = data['상태']

# 범주형 변수 처리 (요청 종류는 문자열이므로 One-Hot Encoding 필요)
X = pd.get_dummies(X, columns=['요청 종류'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier : 여러개의 decision tree를 결합하여 prediction 성능을 높이는 앙상블 방법
# 비선형 데이터에 적합하며, 이상치에 강하고 다양한 데이터 특성 처리에 유연한 모델
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))