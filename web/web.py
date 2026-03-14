from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo_utils import translate
from werkzeug.utils import secure_filename
import time, subprocess

app = Flask(__name__, template_folder='.')
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(WEB_DIR, ".."))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/media/<path:folder>/<filename>')#取得output圖片路徑
def serve_media(folder, filename):
    media_dir = os.path.join(ROOT_DIR, 'static', folder)
    return send_from_directory(media_dir, filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '沒有檔案'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未選擇檔案'}), 400

    # 儲存檔案
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    
    return jsonify({
        'status': 'success',
        'file_path': save_path, # 絕對路徑，供後端使用
        'web_path': f'/static/uploads/{filename}' # 網頁顯示路徑
    })

@app.route('/api/process', methods=['POST'])
def handle_process():
    data = request.json
    image_name = data.get('image_name', '')
    user_input = data.get('text', '')

    image_full_path = os.path.join(UPLOAD_FOLDER, image_name)
    # 檢查圖片是否存在，避免 demo.py 崩潰
    if not os.path.exists(image_full_path):
        return jsonify({'status': 'error', 'message': f'找不到圖片檔案: {image_name}'}), 400
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_json_name = f"task_{timestamp}.json"

    translate_script = os.path.join(ROOT_DIR, "demo_utils", "translate.py")
    demo_script = os.path.join(ROOT_DIR, "demo.py")
    
    temp_json_path = os.path.join(ROOT_DIR, "templates", temp_json_name)
    
    try:
        # 步驟 A: 執行翻譯 
        # 傳入 user_input 和 我們定義好的 temp_json_name
        print(f"正在執行翻譯: {translate_script}")
        subprocess.run([sys.executable, translate_script, user_input, temp_json_name], check=True)
        
        # 步驟 B: 執行 AI 推論
        # 這裡的 --template 需要指向剛剛產生的 JSON 檔案路徑
        print(f"正在執行 AI 推論: {demo_script}")
        subprocess.run([
            sys.executable, 
            demo_script, 
            f"--template={temp_json_path}",
            f"--demo_image_name={image_full_path}"
        ], check=True)

        file_base_name = os.path.splitext(image_name)[0]
        current_model = data.get('model', 'resnet_50x4')
        model_suffix = current_model.replace("resnet_", "r")
        output_image_url = f"/media/output/{file_base_name}_{model_suffix}.jpg"

        summary_filename = f"{file_base_name}_{model_suffix}_summary.txt"
        summary_path = os.path.join(ROOT_DIR, 'static', 'output', summary_filename)
        audit_summary = ""
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                audit_summary = f.read()
        else:
            audit_summary = "無法讀取稽核摘要檔案。"
        
        return jsonify({
            'status': 'success',
            'message': '稽核完成',
            'file': temp_json_name,
            'output_url': output_image_url,
            'summary': audit_summary
        })

    except subprocess.CalledProcessError as e:
        print(f"腳本執行出錯，錯誤碼: {e.returncode}")
        return jsonify({'status': 'error', 'message': f'執行失敗，代碼: {e.returncode}'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)