# tsne_visualization.ipynb

這個 Notebook 用於將模型的特徵向量進行 t-SNE 降維，並產生互動式視覺化結果。

## 功能
- 載入預先訓練的特徵向量
- 使用 t-SNE 進行 2D/3D 降維
- 輸出互動式 HTML 圖表

## 輸入
- `features.pkl`：模型輸出的特徵檔

## 輸出
- `tsne_plot.html`：可互動的視覺化結果
