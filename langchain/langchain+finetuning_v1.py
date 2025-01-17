
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader, TextLoader
import unicodedata
from langchain.schema import Document
from docx import Document as DocxDocument
from datetime import datetime
import pandas as pd
import threading
from flask import Flask, request, render_template
from langchain.chains import LLMChain

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv('FLASK_SECRET_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
# fine_tuned_model = 'ft:gpt-3.5-turbo-1106:auton::AnKA31K3'
fine_tuned_model = 'gpt-4o-mini'

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
당신은 직원 요청 승인 시스템으로, 요청 정보와 요청 사유를 바탕으로 승인, 거절, 보류 중 하나를 판단합니다. 
또한, 필수적으로 거절사유도 함께 설명해주어야 합니다.

주어진 아래의 내용을 바탕으로 요청 상태와 요청 사유가 일치해야합니다.

1. (비업무)개인시간_흡연 등
    - 승인 : 요청 사유가 업무 외의 모든 사항 가능
    - 거절 : 개인시간이 아닌 업무적 사유일 경우(거절 사유 설명 필요)
2. (업무)회의
    - 승인 : 회의내용, 장소, 참석자가 모두 명확히 포함되어 있을 경우
    - 거절 : 회의내용, 장소, 참석자가 누락되는 경우(거절 사유 설명 필요)
    - 보류 : 요청 정보가 명백히 불충분하여 승인 또는 거절 여부를 판단할 수 없는 경우(예: 요청 사유가 단어 하나로만 이루어진 경우)
3. (업무)기타업무
    - 승인 : 요청 사유가 구체적이고 명확히 설명된 경우(예 : 거래처 자료 전달)
    - 거절 : 요청 종류와 요청 사유가 일치하지 않는 경우(우드룸에서 프로젝트 관련 팀장님과 회의)(거절 사유 설명 필요)
    - 보류 : 요청 사유가 매우 짧거나 모호하여 추가 정보가 없으면 판단이 불가능한 경우
4. (업무)출장,이동,외근
    - 승인 : 요청 사유에 출장 및 외근 장소와 내용이 명시되어 있어야 함. 또한 문서번호가 'AUTON'으로 시작하며 숫자가 포함되어 있을 경우
    - 거절 : 장소와 내용이 하나라도 명시되지 않거나 문서번호 형식이 맞지 않을 경우(거절 사유 설명 필요)
    - 보류 : 요청 정보가 불충분하여 판단하기 명백히 어려운 경우
    
참고로 회의실 종류에는 '3층', '1층', '2층', '4층', '5층', '로지', 'ROSY', 'CLOVER', '클로버', '우드', 'WOOD', '오션', 'OCEAN' 등 층수 또는 명사형으로 되어있다.
참고로 회의 참석자에는 사람 이름(예시 : 이승재 등), 직급(팀장, 이사, 대표 등)이 등장한다.

요청 종류: `{request_type}`
요청 사유: `{request_reason}`

결정을 다음 중 하나로 작성하세요: '승인', '거절', '보류'.
결정:
"""




prompt = PromptTemplate(input_variables=["request_type", "request_reason"], template=prompt_template)
# chain = LLMChain(llm=llm, prompt=prompt)

retriever = vectorstore.as_retriever(k=3)
chain = (
    retriever | prompt | llm | StrOutputParser()
)


def request_decision(request_type, request_reason):
    # 벡터 저장소에서 관련 문서 검색
    search_results = vectorstore.similarity_search(request_reason, k=3)  # k는 검색할 문서의 수
    
    # 검색된 문서들로 텍스트 조합
    context = "\n".join([result.page_content for result in search_results])
    
    # 입력 데이터는 반드시 dict 형태여야 합니다.
    input_data = {
        "request_type": request_type,
        "request_reason": request_reason,
        "context": context
    }
    
    # 템플릿에 맞는 형태로 입력 데이터를 구성
    prompt_text = prompt.format(**input_data)  # 여러 입력값을 dict 형태로 전달

    # LLMChain을 사용하여 텍스트를 chain에 전달
    chain = LLMChain(llm=llm, prompt=prompt)  # LLMChain을 별도로 생성하여 사용

    # 의사결정을 생성합니다.
    decision = chain.run(input_data)  # chain.run을 사용하여 직접 dict 데이터를 전달합니다.
    
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
