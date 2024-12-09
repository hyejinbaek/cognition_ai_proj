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
        {"role": "system", "content": "당신은 승인 또는 거절을 판단하는 어시스턴트입니다."},
        {
            "role": "user",
            "content": (
                "다음 입력 데이터를 보고 승인 또는 거절을 판단하세요.\n\n"
                "조건:\n"
                "- (업무)회의: 회의실 장소, 내용, 참석자가 모두 포함되면 승인, 그렇지 않으면 거절.\n"
                "- (업무)기타업무: 사유가 명확하면 승인, 그렇지 않으면 거절.\n\n"
                "예를 들어, 요청 종류가 (업무)출장,이동,외근 일 경우 2024 오토살롱위크 박람회 외근 라고 요청 사유를 작성했을 때 팀즈 공지 참고 후 전자결재 문서 번호 및 사유를 기입하지 않아서 거절이다."
                "즉, 요청 사유에 (업무)출장,이동,외근일 경우 외근 신청서 문서번호가 있어야 승인된다."
                "예를 들어, 요청 종류가 (업무)회의일 경우, 요청 사유가 1층 샘플 수령, 샘플 확인이라고 입력했을 때 기타 업무로 재요청 혹은 회의내용/회의장소/참석자를 포함하지 않았기 때문에 거절이다."
                f"요청 종류: {request_type}\n"
                f"요청 사유: {request_reason}\n\n"
                "결론:"
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