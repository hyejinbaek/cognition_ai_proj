경영지원본부 Shiftee 이석 사유 승인/거절 AI모델 개발

1. 데이터 준비
- 승인/거절된 이석 사유의 데이터셋
- test용 xlsx 파일

2. 데이터 전처리
- 텍스트 정제:
    * 중복 텍스트 제거
    * 불필요한 기호나 단어 제거
- 레이블링(고려필요):
    * 승인 가능한 사유는 1로, 거절해야 하는 사유는 0으로 레이블링
3. 모델 개발
- 모델 선택:
    * 자연어 처리를 위한 BERT 기반 모델 또는 OpenAI의 GPT 모델 사용
- 모델 훈련:
    * 텍스트를 벡터로 변환(TFIDF, 임베딩 등)하여 모델에 입력
    * 이진 분류(Binary Classification) 모델로 학습
4. 승인 기준 정립
- 승인이 되는 경우와 거절되는 경우의 명확한 기준을 정의 필요
- 예:
화장실, 흡연과 같이 개인적 필요가 있는 경우 → 승인
특정 업무와 관련된 사유 → 승인
불분명하거나 부적절한 사유 → 거절
5. 모델 검증
- 기존 데이터 중 일부를 검증 데이터로 활용하여 모델의 정확도를 평가
- 평가지표(고려필요): 정확도(Accuracy), 정밀도(Precision), 재현율(Recall) 등
6. 배포 (팀장님과 논의 필요)
- 모델을 Flask, FastAPI와 같은 백엔드 프레임워크와 통합
- 시프티와 연결(불가능)하여 실시간으로 이석 사유를 분석하고 승인/거절 결과를 반환
7. 추가 고려 사항
- 실시간 학습: 새로운 이석 사유와 결과 데이터를 수집해 모델을 정기적으로 업데이트
- 규칙 기반 보완: 명확한 승인 기준이 있는 경우 AI 모델과 규칙 기반 시스템을 함께 사용