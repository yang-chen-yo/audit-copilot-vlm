
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
sendBtn.addEventListener('click', function() {
    // 隱藏首頁
    landingView.classList.add('hidden');
    
    // 顯示工作區
    workspaceView.classList.remove('hidden');
    workspaceView.classList.add('active');

    // 模擬 AI 處理後的文字更新 (之後這裡會換成真實的 Flask API 回傳結果)
    reportContent.innerHTML = '分析完成！<br>系統明確偵測到：<br> - 9 件 helmet<br> - 14 件 vest';
});