# 엑셀파일을 jsonl 파일로 변환

import pandas as pd
import json

# Load your Excel file
excel_file = "dataset/processed_sample.xlsx"  # Replace with the path to your file
df = pd.read_excel(excel_file)

# Prepare JSONL format
jsonl_data = []
for _, row in df.iterrows():
    # Combine '요청' and '요청 사유' as input and '상태' as output
    prompt = f"요청: {row['요청']}, 요청 사유: {row['요청 사유']}"
    completion = f"{row['상태']}: {row['승인권자 노트']}"
    jsonl_data.append({"prompt": prompt, "completion": completion})

# Save to JSONL file
jsonl_file = "output.jsonl"  # Replace with your desired output filename
with open(jsonl_file, "w", encoding="utf-8") as f:
    for item in jsonl_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"JSONL file saved to {jsonl_file}")
