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
        {"role": "system", "content": "당신은 직원 요청 승인 시스템으로, 요청 정보를 기반으로 승인 또는 거절을 판단합니다."},
        {
            "role": "user",
            "content": (
                "다음 요청 정보를 기반으로 승인을 결정하세요. \n\n"
                "승인 조건:\n"
                "1. (업무)회의 (휴게 X):\n"
                "   - 요청 사유에 회의실 장소, 내용, 참석자가 모두 포함되어야 승인.\n"
                "   - 하나라도 빠지면 거절.\n\n"
                "   - 예시 : 화장실 고도화 미팅, 우드룸, 이승재 일 경우, 회의 내용/장소/참석자 모두 포함되었으므로 승인.\n"
                "   - 예시 : 흡연 방지 회의, 이승재, 5층일 경우, 회의내용/장소/참석자 모두 포함되었으므로 승인.\n"
                "   - 1층, 2층 등으로 장소를 표기할 수 있음. 그럴 경우 장소로 인정됨.\n\n"
                "2. (비업무)개인시간_흡연 등 (휴게 O):\n"
                "   - 요청 사유가 '흡연', '화장실', '편의점', '카페', '병원', '은행', '통화' 등 업무와 관련없는 행위의 개인적인 이유이면 승인.\n"
                "   - 다른 유형의 요청 또는 카테고리가 맞지 않을 시 거절.\n\n"
                "3. (업무)기타업무 (휴게 X):\n"
                "   - 요청 사유가 명확하게 설명될 경우 승인.\n"
                "   - 명확하지 않을 경우 거절.\n\n"
                "4. (업무)출장, 이동, 외근 (휴게 X):\n"
                "   - 요청 사유에 장소와 내용이 명시되어 있어야 승인.\n"
                "   - 명시되지 않으면 거절.\n"
                "   - 외근 신청서 문서번호는 'AUTON'으로 시작하고 연속되는 숫자가 포함되어야 승인.\n"
                "   - 예) AUTON20240101\n\n"
                f"요청 종류: `{request_type}`\n"
                f"요청 사유: `{request_reason}`\n\n"
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