import google.generativeai as genai
import json
import datetime as datetime
import os

API_KEY = "AIzaSyDiplXst3rS29AP5jsjBm4ONPrbU8vOFQ8" 
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

def translate_to_vlm_prompt(zh_text):
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
        
        target_dir = "test_records" 
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, "result.json")
        print(f"檔案確實寫入至：{save_path}")
        # 最精簡的 JSON 輸出：直接覆蓋/寫入到同一個 result.json 檔案
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"categories": parsed_list}, f, ensure_ascii=False, indent=4)

        return translated_text
        
    except Exception as e:
        print(f"API 呼叫失敗: {e}")
        return "translation_error"

# --- 測試區塊 ---
if __name__ == "__main__":
    test_labels = [
        "幫我檢查誰沒有戴安全帽", 
    ]
    
    for zh in test_labels:
        en = translate_to_vlm_prompt(zh)
        print(f"中: {zh}")
        print(f"英: {en}\n")