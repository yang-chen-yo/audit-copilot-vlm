import google.generativeai as genai
import os

API_KEY = "AIzaSyAzcy4LqfT-ELSfdpiqt2lZr3YlpdBvGMc" 
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.1, # 降低隨機性，確保每次翻譯都精確穩定
}

system_instruction = """你現在是一個沒有感情的程式碼翻譯元件，不是聊天機器人。
                    你的唯一任務是將中文翻譯成電腦視覺 (VLM) 專用的英文提示詞。
                    【最高指導原則】：
                    1. 絕對不允許輸出任何問候、解釋、引號、句號或 Markdown 格式。
                    2. 只能輸出翻譯後的純英文。"""


model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction = system_instruction
)

def translate_to_vlm_prompt(zh_text):
    """將中文轉換為 VLM 專用的英文 Prompt"""
    strict_prompt = f"Translate to English. OUTPUT ONLY THE ENGLISH TEXT, NOTHING ELSE: {zh_text}"
    try:
        response = model.generate_content(strict_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"API 呼叫失敗: {e}")
        return "translation_error"

# --- 測試區塊 ---
if __name__ == "__main__":
    test_labels = [
        "'幫我檢查誰沒有戴安全帽", 
        "人臉邊緣有合成痕跡", 
        "背景光影不自然"
    ]
    
    print("--- Gemini API 專家級翻譯測試 ---")
    for zh in test_labels:
        en = translate_to_vlm_prompt(zh)
        print(f"中: {zh}")
        print(f"英: {en}\n")