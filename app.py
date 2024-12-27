# 서버용 api 버전

from flask import Flask, request, render_template
import openai
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def request_decision(request_type, request_reason):
    messages = [
        {"role": "system", "content": "당신은 직원 요청 승인 시스템으로, 요청 정보를 바탕으로 승인, 거절, 보류 중 하나를 판단합니다."},
        {
            "role": "user",
            "content": (
                "다음 요청 정보를 바탕으로 승인, 거절, 보류 중 하나를 판단하세요.\n"
                "### 승인/거절/보류 조건:\n"
                "#### 1. (업무) 회의 (휴게 X):\n"
                "- **승인**: 요청 사유에 **회의 내용**, **회의 장소**, **참석자**가 모두 명확히 포함되어 있을 경우.\n"
                "- **거절**: 회의 내용, 장소, 참석자 중 하나라도 누락되거나 불명확할 경우.\n"
                "- **보류**: 요청 정보가 명백히 불충분하여 승인 또는 거절 여부를 판단할 수 없는 경우(예: 요청 사유가 단어 하나로만 이루어진 경우).\n"
                "- **예시**:\n"
                "  - 승인: '화장실 고도화 미팅, 우드룸, 이승재'\n"
                "  - 승인: '흡연 방지 회의, 5층, 이승재'\n"
                "  - 거절: '회의 참석'\n"
                "  - 보류: '회의'\n\n"
                "#### 2. (비업무) 개인시간 (휴게 O):\n"
                "- **승인**: 요청 사유가 **흡연**, **화장실** 등 명확한 개인적 이유일 경우.\n"
                "- **거절**: 개인시간으로 보기 어려운 업무적 사유일 경우.\n"
                "- **보류**: 요청 사유가 애매하거나 개인적 이유인지 명확하지 않을 경우.\n"
                "- **예시**:\n"
                "  - 승인: '흡연'\n"
                "  - 승인: '화장실'\n"
                "  - 거절: '문서 검토'\n"
                "  - 보류: '시간 필요'\n\n"
                "#### 3. (업무) 기타업무 (휴게 X):\n"
                "- **승인**: 요청 사유가 구체적이고 명확히 설명된 경우.\n"
                "- **거절**: 요청 사유가 구체적이지 않거나 설명이 부족할 경우.\n"
                "- **보류**: 요청 사유가 매우 짧거나 모호하여 추가 정보가 없으면 판단이 불가능한 경우.\n"
                "- **예시**:\n"
                "  - 승인: '거래처 자료 전달 업무'\n"
                "  - 거절: '업무 요청'\n"
                "  - 보류: '업무 관련 요청'\n\n"
                "#### 4. (업무) 출장/이동/외근 (휴게 X):\n"
                "- **승인**: 요청 사유에 **출장/외근 장소**와 **내용**이 모두 명시되어 있고, 외근 신청서 문서번호가 'AUTON'으로 시작하며 연속된 숫자가 포함되어 있을 경우.\n"
                "- **거절**: 장소와 내용 중 하나라도 명시되지 않거나 문서번호 형식이 맞지 않을 경우.\n"
                "- **보류**: 요청 정보가 불충분하여 승인 또는 거절을 판단하기 명백히 어려운 경우.\n"
                "- **예시**:\n"
                "  - 승인: 'AUTON20240101, 서울 본사 미팅'\n"
                "  - 거절: '출장 요청'\n"
                "  - 보류: '서울 출장'\n\n"
                "### 보류 기준:\n"
                "1. 보류는 극히 제한적인 경우에만 사용합니다.\n"
                "2. 보류는 요청 정보가 **명백히 불충분**하거나, 승인/거절을 명확히 판단할 수 없는 경우에만 선택하세요.\n"
                "3. 승인과 거절 중 하나로 판단 가능한 경우에는 보류를 사용하지 마세요.\n\n"
                f"요청 종류: `{request_type}`\n"
                f"요청 사유: `{request_reason}`\n\n"
                "결정을 다음 중 하나로 작성하세요: '승인', '거절', '보류'.\n\n"
                "결정:"
            )
        }
    ]


    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=150,
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"오류: {e}"
    
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
        # 엑셀 파일이 있으면 기존에 추가 (append)
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # 추가 데이터는 헤더를 표시하지 않도록 설정 (신규 시트에는 헤더 추가 필요)
            startrow = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
            df.to_excel(writer, index=False, header=startrow == 0, startrow=startrow)
    except FileNotFoundError:
        # 엑셀 파일이 없는 경우 새로 생성
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