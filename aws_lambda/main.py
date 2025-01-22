from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException 
from selenium.webdriver.chrome.options import Options 
import time 
import pandas as pd 
import requests 
import os 
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from datetime import datetime 
from langchain.chains import LLMChain 

# 환경 변수 로드
load_dotenv() 

# 사용자 정보 설정 
EMAIL = os.getenv("EMAIL") 
PASSWORD = os.getenv("PASSWORD") 
openai_api_key = os.getenv("OPENAI_API_KEY") 

# LangChain 설정
fine_tuned_model = 'gpt-4o'
llm = ChatOpenAI(model=fine_tuned_model, temperature=0) 
embeddings = OpenAIEmbeddings() 

# 벡터 스토어 경로 설정 및 로드 
vectorstore_path = "./vectorstore" 
if os.path.exists(vectorstore_path): 
    vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True)
else: 
    raise ValueError("Error :: 벡터 스토어 생성 필요")

# 프롬프트 템플릿 설정
prompt_template = """
당신은 직원 요청 승인 시스템입니다. 요청 종류와 요청 사유를 바탕으로 승인, 거절, 보류 중 하나를 판단합니다. 아래 정보를 참고하여 요청을 판단하세요:

### 문서 정보
{context}

### 승인 기준:
1. (비업무)개인시간_흡연 등 : 개인적 활동은 모두 승인.
2. (업무)회의 : 장소, 내용, 참석자가 명확히 포함된 경우 승인. 하나라도 누락된 경우 거절.
    - 참고로 회의실 종류에는 '3층', '1층', '2층', '4층', '5층', '로지', 'ROSY', 'CLOVER', '클로버', '우드', 'WOOD', '오션', 'OCEAN' 등 층수 또는 명사형으로 되어있다.
    - 참고로 회의실은 사내 회의실 외 외부 장소도 나온다. 그래서 장소는 명사형으로 판단해라. 
    - 참고로 회의 참석자에는 사람 이름(예시 : 이승재, 이하우 등), 직급(팀장, 이사, 대표 등)이 등장한다. 둘 중에 하나만 있어도 가능하다.
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
retriever = vectorstore.as_retriever(k=3)

def request_decision(request_type, request_reason):
    search_results = vectorstore.similarity_search(request_reason, k=3, distance_metric='cosine')
    context = "\n".join([result.page_content for result in search_results])
    input_data = {
        "request_type": request_type,
        "request_reason": request_reason,
        "context": context
    }
    # RunnableSequence 사용
    chain = prompt | llm  
    decision = chain.invoke(input_data)
    
    # 디버깅을 위해 결정 결과를 콘솔에 출력
    print("Raw decision output:", decision)
    
    # AIMessage 객체의 content 속성에서 텍스트 추출
    decision_text = decision.content.strip()  
    return decision_text


# 브라우저 설정
chrome_options = Options() 
chrome_options.add_argument("--no-sandbox") 
chrome_options.add_argument("--disable-dev-shm-usage") 
# chrome_options.add_argument('--headless')  # 헤드리스 모드 
driver = webdriver.Chrome(options=chrome_options) 
driver.maximize_window() 

try: 
    # 로그인 페이지로 이동
    print("로그인 시작")
    driver.get("https://shiftee.io/ko/accounts/login") 

    # 이메일 입력 
    email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "email-address"))) 
    email_field.send_keys(EMAIL)

    # 비밀번호 입력
    password_field = driver.find_element(By.ID, "password") 
    password_field.send_keys(PASSWORD) 

    # 로그인 버튼 클릭
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary.btn-block.mt-3"))
    ) 
    login_button.click() 

    time.sleep(2)
    print("로그인 완료")
    
    print("요청 페이지로 이동")
    driver.get("https://shiftee.io/app/companies/1855160/manager/requests") 

    dropdown_toggle = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "sft-multi-select .btn-tertiary.dropdown-toggle"))
    ) 
    dropdown_toggle.click() 
    print("드롭다운 토글 클릭")

    dropdown_menu = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.dropdown-menu.show"))
    ) 

    select_all = dropdown_menu.find_element(By.CSS_SELECTOR, "a.sft-dropdown-item-select-all") 
    select_all.click() 

    checkboxes = dropdown_menu.find_elements(By.CSS_SELECTOR, "li.dropdown-item") 
    for checkbox in checkboxes: 
        label = checkbox.find_element(By.TAG_NAME, "span").text 
        if label in ["PC 사용기록"]: 
            checkbox_input = checkbox.find_element(By.CSS_SELECTOR, "div.sft-container > div") 
            if "sft-selected" not in checkbox_input.get_attribute("class"): 
                checkbox.click() 
    
    dropdown_toggle.click() 
    time.sleep(2) 

    print("요청 테이블 처리 시작")
    table_rows = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody > tr"))
    ) 
    for row in table_rows:
        row_id = row.find_element(By.CSS_SELECTOR, "input.sft-table-row-checkbox").get_attribute("sft-data-table-row-id")
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
            decision = request_decision('PC 사용기록', request_reason[0])
            print(f"Row ID: {row_id}, 요청사유: {request_reason[0]}, 결정 및 사유: {decision}")
            
            try:
                decision_type_line = next(line for line in decision.split('\n') if line.startswith('결정:'))
                decision_type = decision_type_line.split('결정: ')[1].strip()
                print(" === 요청사유 결과(승인/거절) : ", decision_type)
            except StopIteration:
                print(f"결정 형식을 찾을 수 없습니다. 결과: {decision}")
                decision_type = "보류"

            # if decision_type == "승인":
            #     # 승인 버튼 클릭
            #     approve_button = WebDriverWait(row, 10).until(
            #         EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-approve.border-info"))
            #     )
            #     approve_button.click()

            #     # 최종 승인 버튼 클릭
            #     final_approve_button = WebDriverWait(driver, 10).until(
            #         EC.element_to_be_clickable((By.CSS_SELECTOR, "div.modal-footer button.btn.btn-primary"))
            #     )
            #     final_approve_button.click()

            #     print(f"Row ID {row_id} 승인 완료.")
            # elif decision_type == "거절":
            #     # 거절 버튼 클릭
            #     reject_button = WebDriverWait(row, 10).until(
            #         EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-danger.border-danger.btn-danger"))
            #     )
            #     reject_button.click()

            #     # 모달에서 노트 입력란 대기
            #     modal = WebDriverWait(driver, 10).until(
            #         EC.presence_of_element_located((By.CSS_SELECTOR, "div.modal-content"))
            #     )
            #     note_input = WebDriverWait(modal, 10).until(
            #         EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[formcontrolname='note']"))
            #     )
                
            #     # 거절 사유 입력
            #     rejection_reason = decision.split('- 사유: ')[1]  # 거절 사유 추출
            #     note_input.send_keys(rejection_reason)

            #     # 거절하기 버튼 클릭
            #     reject_confirm_button = WebDriverWait(modal, 10).until(
            #         EC.element_to_be_clickable((By.CSS_SELECTOR, "div.modal-footer button.btn.btn-primary"))
            #     )
            #     reject_confirm_button.click()

            #     print(f"Row ID {row_id} 거절 완료.")
            # else:
            #     print(f"Row ID {row_id}는 보류되었습니다.")
        else:
            print(f"Row ID: {row_id}, 요청 사유를 찾을 수 없습니다.")

        # close_button = WebDriverWait(popup, 10).until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR, "button.close"))
        # )
        # close_button.click()

    print("모든 작업 완료.")

except TimeoutException:
    print("요소를 찾는 데 시간이 초과되었습니다.")
finally:
    driver.quit()