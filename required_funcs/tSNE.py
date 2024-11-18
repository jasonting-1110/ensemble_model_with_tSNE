import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
from sklearn.manifold import TSNE
from keras.models import Model
from datetime import datetime
from IPython.display import Image
import re
import base64  #将PNG图像解析为base64编码并将其插入到HTML中
import json
import shutil



#函式參數的輸入
#model:ensemble_model layer_name 'dense_5' test_data=[test_data, test_data]

#一個存取許多圖像的list
def count_files_in_folder(folder_path):
    return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])


def extract_info_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 0:
        part0 = parts[0]
    else:
        part0 = filename

    match = re.search(r'(supra\d+|NOR\d+)', filename) #\d+:後面接至少1個數字
    #if match:
    return f"{part0} {match.group(1)}" if match else filename  #兩行簡化為一行 
    #return filename

def extract_image_path_from_html(html):
    start_index = html.find('src="') + len('src="') #.find找出 src="字串在img標籤中的index  從src="中的"為起始點，取出圖像路徑
    end_index = html.find('"', start_index)
    image_path = html[start_index:end_index]
    return image_path

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# 使用base64编码的图像插入到HTML中
def create_html_with_image(encoded_string, true_label, predicted_label):
    html = f'<img src="data:image/png;base64,{encoded_string}"><br>True Label: {true_label}<br>Predicted Label: {predicted_label}'
    return html




def create_html_with_two_images(encoded_string, CAM_encoded_string, true_label_info, predicted_label):
    return (f'<div class="mpld3-tooltip"><p>True Label: {true_label_info}</p>'
            f'<p>Predicted Label: {predicted_label}</p>'
            f'<img src="data:image/png;base64,{CAM_encoded_string}" width="100" height="100">'
            f'<img src="data:image/png;base64,{encoded_string}" width="100" height="100"></div>')

#  <div class="image-container" data-keywords="{true_label_info}">
# </div>
#測試新改的函式:添加的屬性要用" "包住
def create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_label):
    return  f'''
   <div class="image-container" data-keywords="{true_label_info}">
        <img src="data:image/png;base64,{encoded_string}" width="200">     
        <img src="data:image/png;base64,{CAM_encoded_string}" width="200">
        <br>
        <span class="true-label" style="font-size: 30px;"> True Label:{true_label_info}</span> 
        <br>
        <span class="true-label" style="font-size: 30px;"> Predicted: {predicted_label} </span>
    </div>      
    '''

#搜尋引擎
##########################
def add_tooltips(folder, required_count):
    tooltips = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        if os.path.isfile(image_path):
            tooltip = f'<div class="file-name">{filename}</div><img src="{image_path}">'
            tooltips.append((filename, tooltip))
    
    while len(tooltips) < required_count:
        tooltips.extend(tooltips[:required_count - len(tooltips)])
    
    print(f"Generated {len(tooltips)} tooltips from folder: {folder}")
    return tooltips[:required_count]


##########################


#測試合併原圖及cam
#######################################################################################################


#要確保group_indices 在 tooltips 的索引范围内，你需要确保 tooltips 数组正确填充并且包含所有需要的图像数据
    
#預期目標:請把True label改為圖像在folder中的真實名稱->成功
#把錯分的點用紅框圈選出來，像是Predicted label為nor，卻顯示cal點/Predicted label為cal，卻顯示nor點

#把gradCAM圖結合進去 : 添加cal_CAM_folder & 添加nor_CAM_folder
#按一個鈕，切換成CAM/OCT 影像

########################################################################################################
   
def plot_tSNEBest(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder):

    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output) # 輸入原始模型，得到指定layer_name輸出，當作新模型
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1) # 使用指定層輸出對test_data進行預測，得到中間層輸出

    #利用reshape降維成2維數組
    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))
    
    #提取標籤：將 test_labels 從獨熱編碼轉換為類別索引，以獲取每個樣本的實際標籤
    layer_output_label = np.argmax(test_labels, axis=1)

    #創建包含t-SNE座標以及對應標籤的數據框(DataFrame)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    #按照label進行分組
    groups = df.groupby('label')


    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)

    #tooltips量要夠多!!
    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    # if len(tooltips) < intermediate_output.shape[0]:  # 這樣會被原本的test_data總量定死，如果有手動刪除瑕疵test_data，要用更新後的版本去跑
    #     raise ValueError("The number of tooltips is less than the number of data points. Ensure that the tooltips cover all data points.")

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    combined_labels = []
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    
    
    #groups: 按照label分組完的數據點
    for _,group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            all_colors.append(color)

            # 从 HTML 标签中提取图像路径
            image_path = extract_image_path_from_html(tooltip)

            # 编码图像为Base64
            encoded_string = encode_image_to_base64(image_path)

            # 生成HTML内容
            html_content = create_html_with_image(encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            #與圖例出現3個點的原因無關
            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                 
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
                 

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    #避免在循环内多次调用 ax.plot，改为在循环外调用一次 ax.scatter
    scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8,s=5) #s:size
    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=2, hoffset=2))

    legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)]  # [0],[0] represent xdata & ydata
    ax.legend(handles=legend_handles, loc='best')

    plt.title('tSNE')
    plt.tight_layout()
    plt.show()

    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    html_str_with_style=(f'<p style="font-size: 30px;">tSNE Accuracy: {tSNE_acc:.2f}</p>'
                        f'<p style="font-size: 30px;">nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>'
                        f'<p style="font-size: 30px;">cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>'
                        f'<p style="font-size: 30px;">Number of Cal points: {cal_count}</p>'
                        f'<p style="font-size: 30px;">Number of Nor points: {nor_count}</p>' 
                        f'<div style="display: flex; justify-content: center;"><div>{html_str}</div></div>'
                        )
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)

    



"""
2024/7/7 對claude下指令:
我使用plot_tSNEBest，分別對cal_folder, nor_folder 以及cal_CAM_folder, nor_CAM_folder進行處理，
兩者的生成原理相同的，只是每個資料點的對應圖片是不同的，也就是說，我執行
plot_tSNEBest(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder)和
plot_tSNEBest(model, layer_name, test_data, test_labels, save_dir, cal_CAM_folder, nor_CAM_folder)，
現在我想把兩者的圖片整合在同一個介面，請幫我另一個新的函式，把cal_folder, nor_folder,cal_CAM_folder, nor_CAM_folder列入參數中，
只要我執行新立的函式，便能同時實現 plot_tSNEBest(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder)
和plot_tSNEBest(model, layer_name, test_data, test_labels, save_dir, cal_CAM_folder, nor_CAM_folder) 的結果，
你身為一個演算法開發工程師，請根據我提供的函式定義，任意修改
"""
#把gradCAM圖結合進去->成功! 關鍵在於tooltips，以及對tooltips進行encode的函式create_html_with_two_images要多加參數
#在html添加搜尋欄，輸入true_label資訊，便能放大顯示出該資料點(類似電影情節搜索罪犯資料庫)
def plot_tSNE_combined(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, cal_CAM_folder, nor_CAM_folder):
    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    #因為原folder跟CAM_folder對應的資料點是相同的，不需要再對cal_CAM_folder,nor_CAM_folder進行count_files_in_folder
    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)


    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    #多CAM_tooltips
    cal_CAM_tooltips = add_tooltips(cal_CAM_folder, required_cal_count)
    nor_CAM_tooltips = add_tooltips(nor_CAM_folder, required_nor_count)
    CAM_tooltips = cal_CAM_tooltips + nor_CAM_tooltips
    CAM_tooltips = np.array(CAM_tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    combined_labels = []
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []

    for _ ,group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
                CAM_tooltip = CAM_tooltips[i][1] #多CAM_tooltip
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            all_colors.append(color)

            # Original image
            image_path = extract_image_path_from_html(tooltip)
            encoded_string = encode_image_to_base64(image_path)

            # CAM image
            CAM_image_path = extract_image_path_from_html(CAM_tooltip)
            CAM_encoded_string = encode_image_to_base64(CAM_image_path)

            # Generate HTML content with both images :多了CAM_encoded_string
            html_content = create_html_with_two_images(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
               

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8, s=5)
    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=2, hoffset=2)) 
    #實行滑鼠游標懸停，將tooltip connect到散佈圖scatter上，當滑鼠停在資料點時，會顯示combined_labels中對應的html內容

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)
    ]
    ax.legend(handles=legend_handles, loc='best')

    plt.title('tSNE')
    plt.tight_layout()
    plt.show()

    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    html_str_with_style = (f'<p>tSNE Accuracy: {tSNE_acc:.2f}</p>'
                           f'<p>nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>'
                           f'<p>cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>'
                           f'<p>Number of Cal points: {cal_count}</p>'
                           f'<p>Number of Nor points: {nor_count}</p>' 
                           f'<div style="display: flex; justify-content: center;"><div>{html_str}</div></div>' #內容置中
                          )
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE_combined.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)


#先把搜尋列表功能做出來，再想辦法與鼠標自動懸停功能結合(此code不變動，要優化就copy paste，用備份函式修改)
def keySearch_tSNE(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, cal_CAM_folder, nor_CAM_folder,search_term=None):
    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)
    #因為原folder跟CAM_folder對應的資料點是相同的，不需要再對cal_CAM_folder,nor_CAM_folder進行count_files_in_folder

    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    #多CAM_tooltips
    cal_CAM_tooltips = add_tooltips(cal_CAM_folder, required_cal_count)
    nor_CAM_tooltips = add_tooltips(nor_CAM_folder, required_nor_count)
    CAM_tooltips = cal_CAM_tooltips + nor_CAM_tooltips
    CAM_tooltips = np.array(CAM_tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    combined_labels = []
    file_names = []  # 新增：存儲文件名
   

    for  _ ,group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
                CAM_tooltip = CAM_tooltips[i][1] #多CAM_tooltip
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            all_colors.append(color)
            # file_names.append(filename)  # 新增：存儲文件名
            file_names.append(true_label_info)  # 新增：存儲文件名   true_label_info對應到js code中的var fileName

            

            # Original image
            image_path = extract_image_path_from_html(tooltip)
            encoded_string = encode_image_to_base64(image_path) #允許把二進制圖像嵌入到HTML

            # CAM image
            CAM_image_path = extract_image_path_from_html(CAM_tooltip)
            CAM_encoded_string = encode_image_to_base64(CAM_image_path)

            # 修改 create_html_with_image 函数调用，添加唯一标识符(多顯示point_i)
            #創建包含圖像的HTML
            # html_content = create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j],filename)
            #scatter_point_html = f'<circle data-point-index="{i}" ...>'  , file_names
            #scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8)
            html_content = create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            """
            在你之前提供的 create_html_images_ID 函數中，
            true_label_info 被用作一個 span 標籤的內容，而這個 span 標籤具有 class="true-label"。
            因此，在 JavaScript 代碼中，當使用 querySelector('.true-label') 時，
            它會選擇這個 span 標籤並獲取其文本內容，即 true_label_info
            """

            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                 
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
                 

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8)  # 添加 ids 参数->ax.scatter不接受!!

    #print(file_names):確認資訊正確



    

    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)
    ]
    ax.legend(handles=legend_handles, loc='best')

    plt.title('tSNE')
    plt.tight_layout()


    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    # 修改 html_str_with_style，添加搜索功能的 JavaScript 代码

    # 將文件名轉換為 JSON 格式的字符串:才能在js中使用python的數據
    file_names_json = json.dumps(file_names)

    html_str_with_style = (f'''
        <p>tSNE Accuracy: {tSNE_acc:.2f}</p>
        <p>nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>
        <p>cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>
        <p>Number of Cal points: {cal_count}</p>
        <p>Number of Nor points: {nor_count}</p>
        <input type="text" id="search-box" placeholder="Search for true_label_info">
        <div style="display: flex; justify-content: center;"><div>{html_str}</div></div>
        <script>
            var fileNames = {file_names_json};
            
            // Search functionality
            document.getElementById('search-box').addEventListener('input', function() {{ //如其名，透過id名找到搜索框
                var searchTerm = this.value.toLowerCase().trim();
                
                var searchResults = document.getElementById('search-results');

                //如果在搜索框旁边不存在带有 search-results id 的元素，代码会创建一个 ul 元素并赋予其 id='search-results'，然后将其插入到搜索框的下方
                if (!searchResults) {{
                    searchResults = document.createElement('ul');
                    searchResults.id = 'search-results';
                    this.parentNode.insertBefore(searchResults, this.nextSibling); // this為搜索框，确保搜索结果列表在搜索框之后显示
                }}
                searchResults.innerHTML = '';

                if (searchTerm === '') {{
                    return;
                }} 

                var matchedFileNames = [];
                for (var i = 0; i < fileNames.length; i++) {{
                    var fileName = fileNames[i].toLowerCase();
                    if (fileName.match(searchTerm)) {{  // match replaces include
                        matchedFileNames.push({{ index: i, name: fileNames[i] }});
                    }}
                }}

                matchedFileNames.forEach(function(match) {{
                    var li = document.createElement('li');
                    li.textContent = match.name;
                    li.onclick = function() {{
                        document.getElementById('search-box').value = match.name;
                       
                        searchResults.innerHTML = '';
                    }};
                    searchResults.appendChild(li);
                }});

                if (matchedFileNames.length === 1 && 
                matchedFileNames[0].name.toLowerCase() === searchTerm) {{
                    
                    searchResults.innerHTML = '';
                }}
            }});

           
          
     
        </script>
        ''')
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)

    #建立儲存GUI的目錄
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE_combined.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)

def keySearch_try(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, cal_CAM_folder, nor_CAM_folder):
    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)
    #因為原folder跟CAM_folder對應的資料點是相同的，不需要再對cal_CAM_folder,nor_CAM_folder進行count_files_in_folder

    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    #多CAM_tooltips
    cal_CAM_tooltips = add_tooltips(cal_CAM_folder, required_cal_count)
    nor_CAM_tooltips = add_tooltips(nor_CAM_folder, required_nor_count)
    CAM_tooltips = cal_CAM_tooltips + nor_CAM_tooltips
    CAM_tooltips = np.array(CAM_tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    combined_labels = []
    file_names = []  # 新增：存儲文件名
   

    for  _ ,group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
                CAM_tooltip = CAM_tooltips[i][1] #多CAM_tooltip
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            all_colors.append(color)
            # file_names.append(filename)  # 新增：存儲文件名
            file_names.append(true_label_info)  # 新增：存儲文件名   true_label_info對應到js code中的var fileName

            

            # Original image
            image_path = extract_image_path_from_html(tooltip)
            encoded_string = encode_image_to_base64(image_path) #允許把二進制圖像嵌入到HTML

            # CAM image
            CAM_image_path = extract_image_path_from_html(CAM_tooltip)
            CAM_encoded_string = encode_image_to_base64(CAM_image_path)

            # 修改 create_html_with_image 函数调用，添加唯一标识符(多顯示point_i)
            #創建包含圖像的HTML
            # html_content = create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j],filename)
            #scatter_point_html = f'<circle data-point-index="{i}" ...>'  , file_names
            #scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8)
            html_content = create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            """
            在你之前提供的 create_html_images_ID 函數中，
            true_label_info 被用作一個 span 標籤的內容，而這個 span 標籤具有 class="true-label"。
            因此，在 JavaScript 代碼中，當使用 querySelector('.true-label') 時，
            它會選擇這個 span 標籤並獲取其文本內容，即 true_label_info
            """

            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                 
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
                 

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8)  # 添加 ids 参数->ax.scatter不接受!!



    #print(file_names):確認資訊正確



    

    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10))

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)
    ]
    ax.legend(handles=legend_handles, loc='best')

    plt.title('tSNE')
    plt.tight_layout()


    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    # 修改 html_str_with_style，添加搜索功能的 JavaScript 代码

    # 將文件名轉換為 JSON 格式的字符串:才能在js中使用python的數據
    file_names_json = json.dumps(file_names)

    html_str_with_style = (f'''
        <p>tSNE Accuracy: {tSNE_acc:.2f}</p>
        <p>nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>
        <p>cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>
        <p>Number of Cal points: {cal_count}</p>
        <p>Number of Nor points: {nor_count}</p>
        <input type="text" id="search-box" onkeyup="searchFunction()" placeholder="Search for true_label_info">
        <div style="display: flex; justify-content: center;"><div>{html_str}</div></div>
        <script>
            var fileNames = {file_names_json};
            var combinedLabels = {json.dumps(combined_labels)};
            
            // Search functionality
            document.getElementById('search-box').addEventListener('input', function() {{ //如其名，透過id名找到搜索框
                var searchTerm = this.value.toLowerCase().trim();
                
                var searchResults = document.getElementById('search-results');

                //如果在搜索框旁边不存在带有 search-results id 的元素，代码会创建一个 ul 元素并赋予其 id='search-results'，然后将其插入到搜索框的下方
                if (!searchResults) {{
                    searchResults = document.createElement('ul');
                    searchResults.id = 'search-results';
                    this.parentNode.insertBefore(searchResults, this.nextSibling); // this為搜索框，确保搜索结果列表在搜索框之后显示
                }}
                searchResults.innerHTML = '';

                if (searchTerm === '') {{
                    return;
                }} 

                var matchedFileNames = [];
                for (var i = 0; i < fileNames.length; i++) {{
                    var fileName = fileNames[i].toLowerCase();
                    if (fileName.match(searchTerm)) {{  // match replaces include
                        matchedFileNames.push({{ index: i, name: fileNames[i] }});
                    }}
                }}

                matchedFileNames.forEach(function(match) {{
                    var li = document.createElement('li');
                    li.textContent = match.name;
                    //當用戶點擊列表時執行的函數
                    li.onclick = function() {{
                        // 將搜索框的值設置為被點擊的文件名
                        document.getElementById('search-box').value = match.name;
                       // 點擊的當下，會清空搜索結果列表，只留下搜索框打的關鍵字
                        searchResults.innerHTML = '';
                        //showTooltipAndHighlightPoint(match.index); 
                        //searchFunction(); 
                        
                    }};
                    searchResults.appendChild(li);
                }});

                if (matchedFileNames.length === 1 && 
                matchedFileNames[0].name.toLowerCase() === searchTerm) {{
                    searchResults.innerHTML = '';
                    //showTooltipAndHighlightPoint(matchedFileNames[0].index);
                    //searchFunction();
                    
                }}
            }});


            function searchFunction() {{
                var input, filter, gallery, keywords, i;
                input = document.getElementById('search-box');
                filter = input.value.toLowerCase();
                gallery = document.getElementsByClassName('image-container');  //image-container屬性在html_content中
 
                for (i = 0; i < gallery.length; i++) {{
                    keywords = gallery[i].getAttribute('data-keywords');
                    if (keywords.indexOf(filter) > -1) {{
                        gallery[i].style.display = "";  // 顯示匹配的圖像容器
                    }} else {{
                        gallery[i].style.display = "none";
                    }}
                }}           
            }}

            function searchFunction() {{
                var input, filter, gallery, i, labelInfo;
                input = document.getElementById('search-box');
                filter = input.value.toLowerCase();
                gallery = document.getElementById('gallery'); //改寫畫廊，讓它能夠顯示於html網頁
                
                // 清空当前画廊内容
                gallery.innerHTML = '';
                
                for (i = 0; i < combinedLabels.length; i++) {{
                    // 创建一个临时的 div 来解析 HTML 字符串
                    var tempDiv = document.createElement('div');
                    tempDiv.innerHTML = combinedLabels[i];
                    
                    // 获取 true label 信息
                    var trueLabelSpan = tempDiv.querySelector('.true-label');
                    if (trueLabelSpan) {{
                        labelInfo = trueLabelSpan.textContent.toLowerCase();
                        
                        // 检查是否匹配搜索词
                        if (labelInfo.indexOf(filter) > -1) {{
                            // 如果匹配，将这个 HTML 内容添加到画廊中
                            gallery.innerHTML += combinedLabels[i];
                        }}
                    }}
                }}
                
                // 如果没有匹配项，显示一条消息
                if (gallery.children.length === 0) {{
                    gallery.innerHTML = '<p>No matching images found.</p>';
                }}
            }}
                        

            

        </script>
        ''')
    
    #  // 突出顯示數據點
    #             var point = d3.selectAll('.mpld3-baseaxes .mpld3-path')  //d3:data driven document
    #                 .filter(function(d, i) {{ return i === index; }});
    #             if (!point.empty()) {{
    #                 point.node().dispatchEvent(new MouseEvent('mouseover'));
    #                 d3.selectAll('.mpld3-tooltip')
    #                     .style('opacity', 0);
    #                 d3.selectAll('.mpld3-tooltip')
    #                     .filter(function(d, i) {{ return i === index; }})
    #                     .style('opacity', 1);
    #             }}
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)

    #建立儲存GUI的目錄
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成包含搜索功能和画廊的完整 HTML
    # html_str_with_style = create_html_with_search_and_gallery(combined_labels)

    save_path = os.path.join(save_dir, 'tSNE_combined.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)


#幫我設計一個函數，當我在搜索框中輸入匹配某筆fileNames的關鍵字，就要在介面中顯示對應到combinedLabels，透過滑鼠自動懸停來實現(關鍵)
def keySearch_tooltip(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, cal_CAM_folder, nor_CAM_folder):
    batch_size = 128 #原64，調大看速度是否加快(並沒有明顯加快)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)
    #因為原folder跟CAM_folder對應的資料點是相同的，不需要再對cal_CAM_folder,nor_CAM_folder進行count_files_in_folder

    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    #多CAM_tooltips
    cal_CAM_tooltips = add_tooltips(cal_CAM_folder, required_cal_count)
    nor_CAM_tooltips = add_tooltips(nor_CAM_folder, required_nor_count)
    CAM_tooltips = cal_CAM_tooltips + nor_CAM_tooltips
    CAM_tooltips = np.array(CAM_tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    combined_labels = []
    file_names = []  # 新增：存儲文件名
   

    for  _ ,group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
                CAM_tooltip = CAM_tooltips[i][1] #多CAM_tooltip
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            all_colors.append(color)
            # file_names.append(filename)  # 新增：存儲文件名
            file_names.append(true_label_info)  # 新增：存儲文件名   true_label_info對應到js code中的var fileName

            

            # Original image
            image_path = extract_image_path_from_html(tooltip)
            encoded_string = encode_image_to_base64(image_path) #允許把二進制圖像嵌入到HTML

            # CAM image
            CAM_image_path = extract_image_path_from_html(CAM_tooltip)
            CAM_encoded_string = encode_image_to_base64(CAM_image_path)

          
            html_content = create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            """
            在你之前提供的 create_html_images_ID 函數中，
            true_label_info 被用作一個 span 標籤的內容，而這個 span 標籤具有 class="true-label"。
            因此，在 JavaScript 代碼中，當使用 querySelector('.true-label') 時，
            它會選擇這個 span 標籤並獲取其文本內容，即 true_label_info
            """

            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                 
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
                 

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8)  # 添加 ids 参数->ax.scatter不接受!!



    #print(file_names):確認資訊正確(存取了各種病理資料的名稱)

    #考慮把Tooltip丟入js，或許可以實現鼠標自動懸停匹配關鍵字的scatter
    Tooltip = plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10)
    

    plugins.connect(fig,Tooltip)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)
    ]
    ax.legend(handles=legend_handles, loc='best')

    plt.title('tSNE')
    plt.tight_layout()


    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    # 修改 html_str_with_style，添加搜索功能的 JavaScript 代码

    # 將文件名轉換為 JSON 格式的字符串:才能在js中使用python的數據
    file_names_json = json.dumps(file_names)
    combined_labels_json = json.dumps(combined_labels)

    html_str_with_style = (f'''
        <p style="font-size: 30px;">tSNE Accuracy: {tSNE_acc:.2f}</p>
        <p style="font-size: 30px;">nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>
        <p style="font-size: 30px;">cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>
        <p style="font-size: 30px;">Number of Cal points: {cal_count}</p>
        <p style="font-size: 30px;">Number of Nor points: {nor_count}</p>
        <input type="text" id="search-box" onkeyup="searchFunction()" placeholder="Search for true_label_info">
        <div style="display: flex; justify-content: center;"><div>{html_str}</div></div>
        <script>
           var fileNames = {file_names_json};
           var combinedLabels = {combined_labels_json};

            function setupSearchAndTooltip() {{
                var searchBox = document.getElementById('search-box');
                var searchResults = document.getElementById('search-results');
                var tooltip = document.getElementById('tooltip');
                
                if (!tooltip) {{
                    tooltip = document.createElement('div');  //沒有tooltip就自己創建!!->關鍵!
                    tooltip.id = 'tooltip';
                    tooltip.style.position = 'absolute';
                    tooltip.style.display = 'none';
                    tooltip.style.background = 'white';
                    tooltip.style.border = '1px solid black';
                    tooltip.style.padding = '5px';
                    tooltip.style.zIndex = '1000';
                    document.body.appendChild(tooltip);
                }}

                searchBox.addEventListener('input', function() {{
                    var searchTerm = this.value.toLowerCase().trim();
                    
                    if (!searchResults) {{
                        searchResults = document.createElement('ul');
                        searchResults.id = 'search-results'; 
                        this.parentNode.insertBefore(searchResults, this.nextSibling);
                    }}
                    searchResults.innerHTML = '';
                    
                    if (searchTerm === '') {{
                        return;
                    }}
                    
                    var matchedFileNames = [];
                    for (var i = 0; i < fileNames.length; i++) {{
                        var fileName = fileNames[i].toLowerCase();
                        if (fileName.includes(searchTerm)) {{
                            matchedFileNames.push({{ index: i, name: fileNames[i] }});
                        }}
                    }}
                    
                    matchedFileNames.forEach(function(match) {{
                        var li = document.createElement('li');  //沒有就自己創建!!
                        li.textContent = match.name;
                        li.style.cursor = 'pointer';
                        
                        li.onmouseover = function(event) {{
                            tooltip.innerHTML = combinedLabels[match.index]; 
                            tooltip.style.display = 'block';
                            tooltip.style.left = event.pageX + 10 + 'px';
                            tooltip.style.top = event.pageY + 10 + 'px';
                        }};
                        
                        li.onmouseout = function() {{
                            tooltip.style.display = 'none';
                        }};
                        
                        li.onclick = function() {{
                            searchBox.value = match.name;
                            searchResults.innerHTML = '';
                            // 可以在這裡添加其他點擊後的操作
                        }};
                        
                        searchResults.appendChild(li);
                    }});
                }});

                // 點擊頁面其他地方時隱藏搜索結果
                document.addEventListener('click', function(event) {{
                    if (event.target !== searchBox && event.target !== searchResults) {{
                        searchResults.innerHTML = '';
                    }}
                }});
            }}

    // 調用設置函數
    setupSearchAndTooltip();   
        </script>
        ''')
    

    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)

    #建立儲存GUI的目錄
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    save_path = os.path.join(save_dir, 'tSNE_combined.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)

#把js code另外存取並引入，方便閱讀
def keySearch_optimize(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, cal_CAM_folder, nor_CAM_folder):
    batch_size = 128 #原64，調大看速度是否加快(並沒有明顯加快)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)
    #因為原folder跟CAM_folder對應的資料點是相同的，不需要再對cal_CAM_folder,nor_CAM_folder進行count_files_in_folder

    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    #多CAM_tooltips
    cal_CAM_tooltips = add_tooltips(cal_CAM_folder, required_cal_count)
    nor_CAM_tooltips = add_tooltips(nor_CAM_folder, required_nor_count)
    CAM_tooltips = cal_CAM_tooltips + nor_CAM_tooltips
    CAM_tooltips = np.array(CAM_tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    combined_labels = []
    file_names = []  # 新增：存儲文件名
   

    for  _ ,group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
                CAM_tooltip = CAM_tooltips[i][1] #多CAM_tooltip
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            all_colors.append(color)
            # file_names.append(filename)  # 新增：存儲文件名
            file_names.append(true_label_info)  # 新增：存儲文件名   true_label_info對應到js code中的var fileName

            

            # Original image
            image_path = extract_image_path_from_html(tooltip)
            encoded_string = encode_image_to_base64(image_path) #允許把二進制圖像嵌入到HTML

            # CAM image
            CAM_image_path = extract_image_path_from_html(CAM_tooltip)
            CAM_encoded_string = encode_image_to_base64(CAM_image_path)

          
            html_content = create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            """
            在你之前提供的 create_html_images_ID 函數中，
            true_label_info 被用作一個 span 標籤的內容，而這個 span 標籤具有 class="true-label"。
            因此，在 JavaScript 代碼中，當使用 querySelector('.true-label') 時，
            它會選擇這個 span 標籤並獲取其文本內容，即 true_label_info
            """

            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                 
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
                 

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    scatter = ax.scatter(all_x, all_y, c=all_colors, marker='o', alpha=0.8)  # 添加 ids 参数->ax.scatter不接受!!



    #print(file_names):確認資訊正確(存取了各種病理資料的名稱)

    #考慮把Tooltip丟入js，或許可以實現鼠標自動懸停匹配關鍵字的scatter
    Tooltip = plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10)
    

    plugins.connect(fig,Tooltip)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)
    ]
    ax.legend(handles=legend_handles, loc='best')

    plt.title('tSNE')
    plt.tight_layout()


    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    # 修改 html_str_with_style，添加搜索功能的 JavaScript 代码

    # 將文件名轉換為 JSON 格式的字符串:才能在js中使用python的數據
    file_names_json = json.dumps(file_names)
    combined_labels_json = json.dumps(combined_labels)

    #將fileNames和combinedLabels在全局作用域定義，使其可以在外部js中被訪問
    #修改keysearch.js的路徑:D:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\reqiured_funcs
    html_str_with_style = (f'''
        <p style="font-size: 30px;">tSNE Accuracy: {tSNE_acc:.2f}</p>
        <p style="font-size: 30px;">nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>
        <p style="font-size: 30px;">cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>
        <p style="font-size: 30px;">Number of Cal points: {cal_count}</p>
        <p style="font-size: 30px;">Number of Nor points: {nor_count}</p>
        <input type="text" id="search-box" onkeyup="searchFunction()" placeholder="Search for true_label_info">
        <div style="display: flex; justify-content: center;"><div>{html_str}</div></div>
        <script>
            var fileNames = {file_names_json};
            var combinedLabels = {combined_labels_json};
        </script>
        <script src ="keysearch.js"></script>
        ''')
   
    # 存取到當前目錄中
    js_file_path = r'E:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\required_funcs\keysearch.js'
    # 確認 save_path 已正確設置
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)

    # 確認保存目錄已存在，若不存在則創建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 將 keysearch.js 複製到保存目錄中，跟tSNE_combined.html在同路徑才能引入
    shutil.copy(js_file_path, save_dir)

    save_path = os.path.join(save_dir, 'tSNEperp.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)


























