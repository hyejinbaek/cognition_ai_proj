# shiftee 원격접속
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
import json
from dotenv import load_dotenv

load_dotenv()

# 사용자 정보 설정
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")


if not EMAIL or not PASSWORD:
    raise ValueError("EMAIL 또는 PASSWORD 환경 변수가 설정되지 않았습니다.")



chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument('--headless')  # 헤드리스 모드
driver = webdriver.Chrome(options=chrome_options)
driver.maximize_window() #전체 화면 시행 

# Selenium 드라이버 설정
driver.get("https://shiftee.io/ko/accounts/login")  # 로그인 페이지 URL로 이동

# API 정보
API_URL = "https://example.com/approval"  # API 호출 URL

try:
    # 이메일 입력
    email_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "email-address"))
    )
    email_field.send_keys(EMAIL)

    # 비밀번호 입력
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(PASSWORD)

    # 로그인 버튼 클릭
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary.btn-block.mt-3"))
    )
    login_button.click()
    # 추가 대기 (로그인 성공 여부 확인)
    time.sleep(2)

    driver.get("https://shiftee.io/app/companies/1855160/manager/requests")  # 로그인 페이지 URL로 이동

    dropdown_toggle = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "sft-multi-select .btn-tertiary.dropdown-toggle"))
    )
    dropdown_toggle.click()
    print("드롭다운 토글 클릭 완료")

    # 드롭다운 메뉴가 열릴 때까지 대기
    dropdown_menu = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.dropdown-menu.show"))
    )

    # "모두 선택" 클릭
    select_all = dropdown_menu.find_element(By.CSS_SELECTOR, "a.sft-dropdown-item-select-all")
    select_all.click()
    print("모두 선택 클릭 완료")

    # 개별 체크박스 선택 (예: 'PC 사용기록'과 '휴가'만 선택)
    checkboxes = dropdown_menu.find_elements(By.CSS_SELECTOR, "li.dropdown-item")
    for checkbox in checkboxes:
        label = checkbox.find_element(By.TAG_NAME, "span").text
        if label in ["PC 사용기록"]:
            checkbox_input = checkbox.find_element(By.CSS_SELECTOR, "div.sft-container > div")
            # 체크박스 선택 확인 및 클릭
            if "sft-selected" not in checkbox_input.get_attribute("class"):
                checkbox.click()
                print(f"체크박스 '{label}' 선택 완료")

    dropdown_toggle.click()
    print("드롭다운 토글 클릭 완료")
    time.sleep(2)

  # 테이블 로드 대기
    table_rows = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody > tr"))
    )
    
    for row in table_rows:
            # 각 row에서 ID와 관련 데이터를 수집
            row_id = row.find_element(By.CSS_SELECTOR, "input.sft-table-row-checkbox").get_attribute("sft-data-table-row-id")
            request_details = row.find_element(By.CSS_SELECTOR, "td:nth-child(6) div.sft-request-detail")
            
            print(f"Row ID: {row_id} 작업중...")

            time.sleep(2)

            # <td> 내 요청 상세를 클릭하여 팝업 열기
            request_detail_element = row.find_element(By.CSS_SELECTOR, "td.sft-request-tags-table div.sft-request-detail")
            driver.execute_script("arguments[0].click();", request_detail_element)

            # 팝업이 로드될 때까지 대기
            popup = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.sft-middle-item"))
            )

            time.sleep(2)

              # 요청 사유 요소가 로드될 때까지 대기
            reason_elements = WebDriverWait(popup, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.sft-note"))
            )

            # 텍스트 추출 (숨겨진 요소 제외)
            request_reason = [
                element.text.strip() 
                for element in reason_elements 
                if element.is_displayed() and element.text.strip()  # 표시된 요소만 처리
             ]
            
            if request_reason:
                 print("요청사유: ", request_reason[0])  # 요청 사유가 여러 개라면 첫 번째 값을 반환
            else:
                 print("요청 사유를 찾을 수 없습니다.")

            # 팝업 닫기 버튼 찾기
            close_button = WebDriverWait(popup, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.close"))
            )

            # 닫기 버튼 클릭
            close_button.click()
            print(f"Row ID {row_id}의 요청 상세 클릭하여 팝업 닫기 완료.")


            # # 승인테스트
            # approve_button = WebDriverWait(row, 10).until(
            # EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-approve.border-info"))
            # )
            # approve_button.click()
            
            # # 팝업에서 최종 "승인하기" 버튼 클릭
            # final_approve_button = WebDriverWait(driver, 10).until(
            #     EC.element_to_be_clickable((By.CSS_SELECTOR, "div.modal-footer button.btn.btn-primary"))
            # )
            # final_approve_button.click()
            # print(f"Row ID {row_id} 승인 완료.")
        
            # break

            # 거절테스트
            reject_button = WebDriverWait(row, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-danger.border-danger.btn-danger"))
            )
            reject_button.click()
            
            time.sleep(2)

              # 모달이 나타날 때까지 대기
            modal = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.modal-content"))
            )
            print("거절 모달이 열렸습니다.")

            # 노트 입력란이 로드될 때까지 대기
            note_input = WebDriverWait(modal, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[formcontrolname='note']"))
            )

            # 노트 입력
            note_input.send_keys("거절사유 입니다")
            # print(f"거절 사유 입력 완료: {rejection_note}")

            # "거절하기" 버튼 클릭
            # reject_button = WebDriverWait(modal, 10).until(
            #     EC.element_to_be_clickable((By.CSS_SELECTOR, "div.modal-footer button.btn.btn-primary"))
            # )
            # reject_button.click()
            print("거절하기 버튼 클릭 완료.")
                
            time.sleep(2)

            break
 
            # # API 호출 (구현 예정)
            # response = requests.post(API_URL, json={"id": row_id, "details": request_details, "reason": request_reason[0]})
            
            # if response.status_code == 405:
            #     # result = response.json().get("status")
            #     # if result == "approved":
            #     if True:
            #         print(f"Row ID {row_id} 승인 중...")
                    
            #         # "승인" 버튼 클릭
            #         approve_button = WebDriverWait(row, 10).until(
            #         EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-approve.border-info"))
            #         )
            #         approve_button.click()
                    
            #         # 팝업에서 최종 "승인하기" 버튼 클릭
            #         final_approve_button = WebDriverWait(driver, 10).until(
            #             EC.element_to_be_clickable((By.CSS_SELECTOR, "div.modal-footer button.btn.btn-primary"))
            #         )
            #         # final_approve_button.click()
            #         print(f"Row ID {row_id} 승인 완료.")
                    
            #     elif result == "denied":
            #         print(f"Row ID {row_id}는 거절되었습니다.")
            #     elif result == "holding":
            #         print(f"Row ID {row_id}는 대기 상태입니다.")
            #     else:
            #         print(f"알 수 없는 상태: {result}")
            # else:
            #     print(f"API 호출 실패 (Row ID {row_id}, 상태 코드: {response.status_code})")
            # break
        
    print("모든 작업 완료.")
    
except TimeoutException:
    print("요소를 찾는 데 시간이 초과되었습니다.")
finally:
    # 브라우저 닫기
    driver.quit()