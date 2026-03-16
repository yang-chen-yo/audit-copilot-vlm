import google.generativeai as genai
import json
import datetime as datetime
import os

API_KEY = "AIzaSyAf437Ar4fAtXaPmLst77TggVpvOMkuYQg" 
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.1, 
}

system_instruction = """
你是一個專業的「影像稽核規則轉換器」。請根據使用者的中文輸入，理解其場域與意圖，並將其轉換為嚴格的 JSON 格式。

輸出 JSON 的規範：
1. categories: 包含所有要偵測的英文名詞。
2. person_category: 從 categories 中選出代表「人」的標籤。
3. thresholds: 為 categories 中的每個物件設定偵測門檻 (0.1 ~ 0.9)。
   - 一般人或大型裝備設 0.3
   - 小物件 (如手套、眼鏡) 設 0.2
   - 顏色容易混淆或重要的裝備 (如反光背心) 設 0.5
4. compliance_rules: 定義穿戴規則。
   - target: 必須與 person_category 一致。
   - required_ppe: 該角色必須佩戴的裝備列表。

範例輸出：
{
  "categories": ["construction worker", "hard hat", "safety vest"],
  "person_category": "construction worker",
  "thresholds": {
    "construction worker": 0.3,
    "hard hat": 0.2,
    "safety vest": 0.5
  },
  "compliance_rules": [
    {
      "target": "construction worker",
      "required_ppe": ["hard hat", "safety vest"]
    }
  ]
}
"""


model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction = system_instruction
)

import os
import json

def translate_to_vlm_prompt(zh_text, output_filename):
    """將中文轉換為 VLM 專用的複雜 JSON 規則檔"""
    prompt = f"請將以下需求轉換為指定的 JSON 格式：{zh_text}"
    
    try:
        response = model.generate_content(prompt)
        translated_text = response.text.strip()

        try:
            parsed_data = json.loads(translated_text)
        except json.JSONDecodeError:
            clean_text = translated_text.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_text)
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_file_dir, "..", "templates")
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, output_filename) 
        
        print(f"檔案確實寫入至：{save_path}")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)

        return translated_text
        
    except Exception as e:
        print(f"API 呼叫失敗或解析錯誤: {e}")
        return "translation_error"

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        translate_to_vlm_prompt(sys.argv[1], sys.argv[2])
    else:
        print("❌ 參數不足！請輸入：python translate.py '中文指令' '檔名.json'")