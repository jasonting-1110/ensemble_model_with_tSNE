# 專案重點

這個專案包含多個 Notebook 與功能模組，重點程式細節請參考以下文件：
- [資料前處理](2024_8_13/dat_data_preprocessing)
- [模型訓練與優化](required_funcs/deal_with_resized_npy_ensemble.ipynb)
- [gradCAM熱力圖實作](required_funcs/CAM_generation.ipynb)
- [t-SNE 可視化界面實作](required_funcs/tSNE.py)
- [關鍵字搜尋](required_funcs/keysearch_try.js)

- ## 其他 Branch 資源
- [imgPreprocessing branch:資料前處理](https://github.com/jasonting-1110/ensemble_model_with_tSNE/tree/imgPreprocessing/2024_8_13/dat_data_preprocessing/840_OCT_loopTest.ipynb)
-1. 資料讀取與記憶體管理:一次讀取單一 frame.dat 檔，避免載入過多資料造成記憶體爆滿，並依照子資料夾排序與篩選，批次處理多個 frame。

-2. 背景扣除 (Background Subtraction):從多張背景檔 (BG/frame*.dat) 建立三維矩陣，計算平均背景影像與 A-line 平均值，降低雜訊，並將背景平均值套用到每個 frame，達到雜訊扣除效果。

-3. 像素校正 (Pixel Calibration):載入外部校正檔案 p_avg_1.txt，使用 Cubic Spline interpolation 對每條 A-line 進行像素重取樣，確保訊號對齊。

-4. 傅立葉轉換與零填補 (FFT & Zero-padding):對每個 A-line 進行 FFT，轉換到頻域，在頻域中進行 zero padding，再用 IFFT 還原到空間域，提升解析度。

-5. 訊號重組與區域提取:對內插後的訊號再次 FFT → 計算 dB 強度，重新排列頻譜，確保正確頻域結構，擷取關注區域 (像素範圍 450–1000)，獲取最終影像矩陣，對影像進行旋轉，符合分析需求。

-6. 結果輸出:每個 frame 輸出兩種檔案格式：.png（整數矩陣 → 檔案小，方便視覺化）.npy（浮點矩陣 → 保留完整數值資訊，並儲存於子資料夾中的 png/ 與 npy/ 資料夾。

-7. 效能監控:在處理每個資料夾與 frame 時，輸出進度資訊，記錄單一資料夾的運算時間，利於效能評估。
