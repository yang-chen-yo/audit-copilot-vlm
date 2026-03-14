import google.generativeai as genai
import json
import datetime as datetime
import os

API_KEY = "AIzaSyDdjBpnxsv2WbPOrtJgLvtH2EytPs2eayw" 
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.1, # 降低隨機性，確保每次翻譯都精確穩定
}

system_instruction = """["person", "hard hat", "safety vest"]
                    上面是我的輸出範例，你只需要根據我的中文輸入，找出我想要尋找名詞並且輸出上面的英文格式"""


model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction = system_instruction
)

import os
import json

def translate_to_vlm_prompt(zh_text, output_filename):
    """將中文轉換為 VLM 專用的英文 Prompt"""
    strict_prompt = f"OUTPUT ONLY THE ENGLISH TEXT, NOTHING ELSE: {zh_text}"
    try:
        response = model.generate_content(strict_prompt)
        translated_text = response.text.strip()

        try:
            # 讓 Python 把它當成真正的 List 解析
            parsed_list = json.loads(translated_text)
        except json.JSONDecodeError:
            # 防呆機制：萬一 Gemini 亂加了 Markdown (例如 ```json)
            clean_text = translated_text.replace("```json", "").replace("```", "").strip()
            parsed_list = json.loads(clean_text)
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_file_dir, "..", "templates")
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, output_filename) 
        print(f"檔案確實寫入至：{save_path}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"categories": parsed_list}, f, ensure_ascii=False, indent=4)

        return translated_text
        
    except Exception as e:
        print(f"API 呼叫失敗: {e}")
        return "translation_error"

# --- 測試區塊 ---
if __name__ == "__main__":
    import sys
    # sys.argv[1] 是第一個參數 (中文內容)
    # sys.argv[2] 是產出的檔名
    if len(sys.argv) >= 3:
        translate_to_vlm_prompt(sys.argv[1], sys.argv[2])
    else:
        print("❌ 參數不足！請輸入：python translate.py '中文指令' '檔名.json'")