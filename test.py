import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # 모델 저장 및 불러오기 위해 joblib 사용

# 1. 엑셀 파일 읽기
file_path = './dataset/sample.xlsx' 
data = pd.read_excel(file_path)

# 2. 데이터 전처리
# 요청 카테고리에서 '/' 뒤 텍스트만 추출
data['요청_카테고리'] = data['요청'].str.split('/').str[-1].str.strip()

# 결합하여 모델 학습용 텍스트 생성
data['입력_텍스트'] = data['요청_카테고리'] + ' ' + data['요청 사유'].fillna('')

# '상태'를 레이블로 설정
data['상태'] = data['상태'].map({'승인됨': 1, '거절됨': 0})

# '승인됨'과 '거절됨' 외의 상태를 가진 행을 제거
data = data[data['상태'].isin([1, 0])]

# 유효한 데이터만 사용
data = data.dropna(subset=['입력_텍스트', '상태'])

# 3. 학습 데이터와 테스트 데이터 분리
X = data['입력_텍스트']
y = data['상태']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train_vec, y_train)

# 6. 모델 저장
joblib.dump(model, 'random_forest_model.pkl')  # 모델을 파일로 저장
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # 벡터라이저도 저장

# 7. 테스트 데이터로 예측
y_pred = model.predict(X_test_vec)

# 8. 결과 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy * 100:.2f}%")

# 9. 모델과 벡터라이저 불러오기
def load_model_and_vectorizer():
    model = joblib.load('random_forest_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# 10. 새로운 데이터 입력 및 예측 함수
def predict_status(new_request, new_reason):
    # 모델과 벡터라이저 불러오기
    model, vectorizer = load_model_and_vectorizer()
    
    # 입력 텍스트 생성
    input_text = new_request + ' ' + new_reason
    input_vec = vectorizer.transform([input_text])
    
    # 예측
    prediction = model.predict(input_vec)
    return '승인됨' if prediction[0] == 1 else '거절됨'

# 11. 사용자로부터 요청 카테고리 및 사유 입력 받기
new_request = input("요청 카테고리를 입력하세요 (예: 개인시간): ")
new_reason = input("사유를 입력하세요: ")

# 예측 결과 출력
print(f"예측 결과: {predict_status(new_request, new_reason)}")
