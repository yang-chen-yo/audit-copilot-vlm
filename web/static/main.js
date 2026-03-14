
// 取得元素
const sendBtn = document.getElementById('send-btn');
const landingView = document.getElementById('landing-view');
const workspaceView = document.getElementById('workspace-view');
const reportContent = document.getElementById('report-content');
const fileUpload = document.getElementById('file-upload');
const fileNameDisplay = document.getElementById('file-name-display');
const fileTagContainer = document.getElementById('file-tag-container');

// 1. 當使用者選擇檔案時，更新上方的小標籤檔名
fileUpload.addEventListener('change', function(e) {
    if(e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
        fileTagContainer.style.display = 'inline-flex';
    } else {
        fileTagContainer.style.display = 'none';
    }
});

// 2. 點擊「↑」按鈕時，切換到工作區
sendBtn.addEventListener('click', async function() {
    const userInput = document.getElementById('user-input').value;
    const file = fileUpload.files[0]; // 取得輸入框文字
    if (!userInput) return alert("請輸入指令");
    if (!file) return alert("請先上傳圖片");

    // 1. 切換 UI 狀態
    landingView.classList.add('hidden');
    workspaceView.classList.remove('hidden');
    workspaceView.classList.add('active');
    reportContent.innerHTML = '<div class="loading">正在思考中...</div>';

    try {
        // --- 步驟 A: 上傳圖片 ---
        const formData = new FormData();
        formData.append('file', file);

        const uploadRes = await fetch('/api/upload', {
            method: 'POST',
            body: formData // 注意：傳送檔案不要設定 Content-Type header
        });
        const uploadData = await uploadRes.json();

        if (uploadData.status !== 'success') {
            throw new Error("圖片上傳失敗: " + uploadData.error);
        }

        // 取得伺服器存檔後的檔名 (從後端回傳中擷取)
        const savedFileName = uploadData.web_path.split('/').pop();

        // --- 步驟 B: 呼叫處理 API ---
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: userInput,
                image_name: savedFileName // 把檔名傳給後端 handle_process
            })
        });

        const result = await response.json();

        if (result.status === 'success') {
            reportContent.innerHTML = `
                <div style="white-space: pre-wrap; line-height: 1.6;">
                ${result.summary}
                </div>
                `;
            const resultImg = document.getElementById('result-img');
            if (resultImg) {
                // 加上 ?t=時間戳記 是為了防止瀏覽器快取 (Cache)，確保永遠顯示最新生成的圖片
                resultImg.src = result.output_url + '?t=' + new Date().getTime();
                resultImg.alt = "稽核完成的結果圖";
            }
        } else {
            reportContent.innerHTML = `發生錯誤：${result.message}`;
        }

    } catch (error) {
        console.error("Error:", error);
        reportContent.innerHTML = "處理失敗: " + error.message;
    }
});