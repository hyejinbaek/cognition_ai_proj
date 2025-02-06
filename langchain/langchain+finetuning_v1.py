
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import unicodedata
from datetime import datetime
import pandas as pd
import threading
from flask import Flask, request, render_template
from langchain.chains import LLMChain
import openai

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv('FLASK_SECRET_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
# fine_tuned_model = 'ft:gpt-3.5-turbo-1106:auton::AnKA31K3'
fine_tuned_model = 'gpt-4o'

# LangChain을 통한 모델 설정
llm = ChatOpenAI(model=fine_tuned_model, temperature=0)

# 벡터 저장소 설정 (FAISS 예시)
# 이미 만들어놓은 벡터 저장소를 로드합니다.
# 예시에서는 FAISS를 사용하지만 다른 저장소도 가능합니다.
embeddings = OpenAIEmbeddings()
# vector_store = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)

# 벡터 스토어 경로 설정
vectorstore_path = "./vectorstore"

# 벡터 스토어 로드 또는 생성
if os.path.exists(vectorstore_path):
    print("벡터 스토어 로드 중...")
    vectorstore = FAISS.load_local(
        vectorstore_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True  
    )
else:
    print("벡터 스토어 생성 중...")
    vectorstore = FAISS.from_documents(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    vectorstore.save_local(vectorstore_path)

# 메시지 및 규칙을 템플릿으로 정의
prompt_template = """
당신은 직원 요청 승인 시스템입니다. 요청 종류와 요청 사유를 바탕으로 승인, 거절 중 하나를 판단합니다. 아래 정보를 참고하여 요청을 판단하세요:

### 문서 정보
{context}

### 승인 기준:
1. (비업무)개인시간_흡연 등 : 개인적 활동은 모두 승인.
2. (업무)회의 : 장소, 내용, 참석자가 명확히 포함된 경우 승인. 하나라도 누락된 경우 거절.
    - 장소는 특정 단어로 나타나며, 사내외 장소일 수 있습니다.
    - 참석자는 사람 이름(예: 이승재 등) 또는 직급(예: 팀장, 이사, 대표)이 포함되어야 합니다.
3. (업무)기타업무 : 요청 사유가 설명된 경우 승인.
4. (업무)출장,이동,외근 : 출장 및 외근 장소와 내용이 명시되어 있어야 함. 또한 문서번호('AUTON')가 포함된 경우 승인.

### 요청 세부사항
요청 종류: {request_type}
요청 사유: {request_reason}

### 결정 형식
결정 및 이유를 아래 형식으로 작성하세요:
- 결정: (승인, 거절 중 하나)
- 사유: (거절 이유를 간단히 설명)

결정 및 이유:
"""



prompt = PromptTemplate(input_variables=["request_type", "request_reason"], template=prompt_template)
# chain = LLMChain(llm=llm, prompt=prompt)

retriever = vectorstore.as_retriever(k=3)
chain = (
    retriever | prompt | llm | StrOutputParser()
)


def request_decision(request_type, request_reason):
    # 벡터 저장소에서 관련 문서 검색
    search_results = vectorstore.similarity_search(request_reason, k=3, distance_metric='cosine')
    
    # 검색 결과 확인용 출력
    print("검색된 문서:")
    for i, result in enumerate(search_results):
        print(f"문서 {i+1}:\n{result.page_content}\n")

    # 검색된 문서들로 텍스트 조합
    context = "\n".join([result.page_content for result in search_results])
    print(f" ===== 생성된 context:\n{context}")
    
    # 나머지 로직 그대로 유지
    input_data = {
        "request_type": request_type,
        "request_reason": request_reason,
        "context": context
    }
    prompt_text = prompt.format(**input_data)
    chain = LLMChain(llm=llm, prompt=prompt)
    decision = chain.run(input_data)
    
    return decision.strip()







def save_to_excel(request_type, request_reason, decision):
    filename = "user_requests.xlsx"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        '시간': [now],
        '요청 종류': [request_type],
        '요청 사유': [request_reason],
        '판단 결과': [decision]
    }
    df = pd.DataFrame(data)
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            startrow = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
            df.to_excel(writer, index=False, header=startrow == 0, startrow=startrow)
    except FileNotFoundError:
        df.to_excel(filename, index=False)

@app.route("/", methods=["GET", "POST"])
def index():
    decision = None
    if request.method == "POST":
        request_type = request.form.get("request_type")
        request_reason = request.form.get("request_reason")
        decision = request_decision(request_type, request_reason)
        save_to_excel(request_type, request_reason, decision)
    return render_template("index.html", decision=decision)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
