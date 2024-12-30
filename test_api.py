import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# 모델과 벡터라이저 불러오기
def load_model_and_vectorizer():
    model = joblib.load('random_forest_model.pkl')  # 저장된 모델 불러오기
    vectorizer = joblib.load('tfidf_vectorizer.pkl')  # 저장된 벡터라이저 불러오기
    return model, vectorizer

# 예측 함수
def predict_status(new_request, new_reason):
    model, vectorizer = load_model_and_vectorizer()  # 모델과 벡터라이저 불러오기
    
    # 입력 텍스트 생성
    input_text = new_request + ' ' + new_reason
    input_vec = vectorizer.transform([input_text])  # 벡터화
    
    # 예측
    prediction = model.predict(input_vec)
    return '승인됨' if prediction[0] == 1 else '거절됨'

# 기본 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html', decision=None)

# 예측 요청 처리
@app.route('/', methods=['POST'])
def predict():
    # POST 요청에서 데이터 받기
    request_category = request.form.get('request_type')  # 요청 카테고리
    reason = request.form.get('request_reason')  # 요청 사유
    
    if not request_category or not reason:
        return render_template('index.html', decision='요청 카테고리와 사유는 필수 항목입니다.')
    
    # 예측
    result = predict_status(request_category, reason)
    return render_template('index.html', decision=result)

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
