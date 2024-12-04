from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

file_path = './dataset/sample.xlsx'
df = pd.read_excel(file_path)

result = df[['요청 사유', '상태']]

# 1: 승인, 0: 거절
status_mapping = {'승인됨': 0, '거절됨': 1, '취소됨': 2, '대기중': 3, 'NaN' : 4}
reverse_mapping = {v: k for k, v in status_mapping.items()}  # 역매핑 생성
result['상태'] = result['상태'].map(status_mapping)

# NaN 값 확인 및 제거
print("***********2 *********NaN 값 확인:", result[result['상태'].isna()])

# 텍스트 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(result['요청 사유'])

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, result['상태'], test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
print("X_test === ", X_test)
print("y_pred === ", y_pred)

# 정확도 출력
print("Accuracy:", accuracy_score(y_test, y_pred))

# 요청 사유와 상태 원래 값으로 확인
# X_test에 해당하는 원래 텍스트와 상태를 찾기 위해 인덱스 복원
test_indices = y_test.index
original_requests = df.loc[test_indices, '요청 사유']  # 원래 요청 사유
original_status_actual = df.loc[test_indices, '상태']  # 원래 상태 (실제값)
predicted_status = [reverse_mapping[val] for val in y_pred]  # 예측 상태 (역매핑)

# 결과 데이터프레임 생성
test_results = pd.DataFrame({
    "요청 사유": original_requests.values,
    "실제 상태": original_status_actual.values,
    "예측 상태": predicted_status
})

print("\n테스트 데이터 결과:")
print(test_results)

output_file_path = './test_results.xlsx'
test_results.to_excel(output_file_path, index=False)
print(f"테스트 결과가 {output_file_path}에 저장되었습니다.")
