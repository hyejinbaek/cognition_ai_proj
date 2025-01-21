from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os
from dotenv import load_dotenv

# LangChain and AI library imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# User credentials and API keys
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Validate presence of essential environment variables
if not EMAIL or not PASSWORD:
    raise ValueError("EMAIL 또는 PASSWORD 환경 변수가 설정되지 않았습니다.")

# Integrate OpenAI Model and FAISS Vector Store
fine_tuned_model = 'gpt-4o-mini'
llm = ChatOpenAI(model=fine_tuned_model, temperature=0)
embeddings = OpenAIEmbeddings()

vectorstore_path = "./vectorstore"
if os.path.exists(vectorstore_path):
    print("기존 벡터 스토어 로드")
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("새로운 벡터 스토어 생성")
    # documents = ...  # Load your documents here to create vector store
    # vectorstore = FAISS.from_documents(documents, embeddings)
    # vectorstore.save_local(vectorstore_path)

# Function to make decision based on request details
def request_decision(request_type, request_reason):
    print(f"유사도 검색 시작: 요청사유 = {request_reason}")
    search_results = vectorstore.similarity_search(request_reason, k=3)
    context = "\n".join([result.page_content for result in search_results])

    print("컨텍스트 준비 완료, AI 모델에 요청 처리 시작")
    prompt_template = """
    당신은 직원 요청 승인 시스템입니다. 요청 정보와 요청 사유를 바탕으로 승인, 거절, 보류 중 하나를 판단합니다.
    아래 문서의 정보를 참고하여 요청을 판단하세요:
    ### 문서 정보
    {context}
    ### 요청 세부사항
    요청 종류: {request_type}
    요청 사유: {request_reason}
    결정 및 이유:
    """
    prompt = PromptTemplate(input_variables=["request_type", "request_reason"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    input_data = {"request_type": request_type, "request_reason": request_reason, "context": context}
    decision = chain.run(input_data)
    print(f"결정 완료: {decision.strip()}")
    return decision.strip()

# Selenium setup
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument('--headless')  # Headless mode
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window()

# Selenium workflow for accessing and processing requests
print("브라우저 열기")
driver.get("https://shiftee.io/ko/accounts/login")

try:
    # Perform login
    print("로그인 시작")
    email_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "email-address"))
    )
    email_field.send_keys(EMAIL)

    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(PASSWORD)

    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary.btn-block.mt-3"))
    )
    login_button.click()
    time.sleep(2)
    print("로그인 완료")

    # Access requests page and process requests
    print("요청 페이지로 이동")
    driver.get("https://shiftee.io/app/companies/1855160/manager/requests")

    # Open and toggle dropdowns, select checkboxes
    dropdown_toggle = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "sft-multi-select .btn-tertiary.dropdown-toggle"))
    )
    dropdown_toggle.click()
    print("드롭다운 토글 클릭")

    # Repeat previous logic for dropdown interaction

    # Process request table
    print("요청 테이블 처리 시작")
    table_rows = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody > tr"))
    )

    for row in table_rows:
        # Extract request information and detail
        row_id = row.find_element(By.CSS_SELECTOR, "input.sft-table-row-checkbox").get_attribute("sft-data-table-row-id")
        request_type = "PC 사용 기록"  # Define relevant type logic

        request_details = row.find_element(By.CSS_SELECTOR, "td:nth-child(6) div.sft-request-detail")
        request_detail_element = row.find_element(By.CSS_SELECTOR, "td.sft-request-tags-table div.sft-request-detail")
        driver.execute_script("arguments[0].click();", request_detail_element)

        popup = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.sft-middle-item"))
        )

        reason_elements = WebDriverWait(popup, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.sft-note"))
        )

        request_reason = [
            element.text.strip()
            for element in reason_elements
            if element.is_displayed() and element.text.strip()
        ]

        if request_reason:
            print("요청사유: ", request_reason[0])

        close_button = WebDriverWait(popup, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.close"))
        )
        close_button.click()

        ########################################################################################
        ## 승인 클릭 부분 
        ########################################################################################
        # # Get decision from AI model
        # decision = request_decision(request_type, request_reason[0])
        # print(f"Row ID {row_id}에 대한 결정: {decision}")

        # Implement actions based on decision (approve/reject/hold)
        # Example: if decision == "승인":
        #     approve_button = ...
        #     approve_button.click()

        break  # For demo purposes, handling just one row, remove this line to process all

    print("모든 작업 완료.")
except Exception as e:
    print(f"오류 발생: {e}")
finally:
    driver.quit()
    print("브라우저 닫힘")