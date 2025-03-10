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
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException


# os.chdir("/home/shiftee/aws_lambda")

# /var/task
# /home/shiftee/aws_lambda 

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
    - 참고로 회의실은 사내 회의실 외 단순히 '외부장소', '자리', '타운홀'과 영어로도 나온다. 그래서 장소는 명사형 및 외부장소, 자리라고 되어있어도 승인해라. 
    - 참고로 회의 참석자에는 사람 이름(예시 : 이승재, 이하우 등), 직급(팀장, 이사, 대표 등)이 등장한다. 둘 중에 하나만 있어도 가능하다.
3. (업무)기타업무 : 요청 사유가 설명된 경우 승인.
4. (업무)출장,이동,외근 : 출장 및 외근 장소와 내용이 명시되어 있어야 함.

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

# 요청사항을 '/' 기준으로 분리하는 함수
def split_request_detail(request_detail):
    if '/' in request_detail:
        return request_detail.split('/')[-1].strip()
    return request_detail


# request_decision 함수를 수정하여 요청사유의 첫 번째 요소만 사용하도록 수정합니다.
def request_decision(request_type, request_reason):
    search_results = vectorstore.similarity_search(request_reason, k=3, distance_metric='cosine')
    context = "\n".join([result.page_content for result in search_results])
    input_data = {
        "request_type": request_type,
        "request_reason": request_reason,
        "context": context
    }
    chain = prompt | llm  
    decision = chain.invoke(input_data)
    print("Raw decision output:", decision)
    decision_text = decision.content.strip()  
    return decision_text


# 브라우저 설정
chrome_options = Options() 
chrome_options.add_argument("--no-sandbox") 
chrome_options.add_argument("--disable-dev-shm-usage") 
# chrome_options.add_argument('--headless')  # 헤드리스 모드 
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window() 
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument('--remote-debugging-port=9222')
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
)


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

    while True:
        try:
            print("요청 테이블 처리 시작")

            # 현재 화면에서 최신 row 목록 가져오기
            table_rows = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody > tr"))
            )

            if not table_rows:
                print("모든 요청을 처리 완료했습니다. 종료합니다.")
                break  # 남아있는 요청이 없으면 종료

            while table_rows:
                try:
                    row = table_rows[0]  # 항상 첫 번째 row 선택
                    row_id = row.find_element(By.CSS_SELECTOR, "input.sft-table-row-checkbox").get_attribute("sft-data-table-row-id")
                    request_detail_element = row.find_element(By.CSS_SELECTOR, "td.sft-request-tags-table div.sft-request-detail")
                    request_details = request_detail_element.text.strip()
                    request_type = split_request_detail(request_details)

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
                        decision = request_decision(request_type, request_reason[0])
                        print(f"Row ID: {row_id}, 요청사유: {request_reason[0]}, 결정 및 사유: {decision}")

                        if "결정: 승인" in decision:
                            approve_button = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, "//div[@class='sft-footer']//button[contains(text(), '승인')]"))
                            )
                            driver.execute_script("arguments[0].click();", approve_button)
                            time.sleep(2)

                            final_approve_button = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, "//sft-action-request-modal//button[contains(text(), '승인하기')]"))
                            )
                            final_approve_button.click()
                            print(f"Row ID {row_id} 승인 완료.")

                        elif "결정: 거절" in decision:
                            try:
                                close_buttons = popup.find_elements(By.CSS_SELECTOR, "button.close")
                                if close_buttons:
                                    driver.execute_script("arguments[0].click();", close_buttons[0])
                                    print("거절 팝업 닫기 완료")
                                WebDriverWait(driver, 5).until(EC.invisibility_of_element(popup))
                                print("팝업 완전 종료 확인")
                            except Exception as e:
                                print(f"거절 팝업 닫기 중 에러 발생, 무시하고 pass: {e}")
                    else:
                        print(f"Row ID: {row_id}, 요청 사유를 찾을 수 없습니다.")

                    # 현재 row 목록에서 제거
                    table_rows.pop(0)

                except StaleElementReferenceException:
                    print("StaleElementReferenceException 발생, 다시 시도합니다.")
                    break  # 다시 목록을 가져오도록 설정

        except TimeoutException:
            print("더 이상 요청이 없습니다. 종료합니다.")
            break  # 남아있는 요청이 없으면 종료




finally:
    driver.quit()
