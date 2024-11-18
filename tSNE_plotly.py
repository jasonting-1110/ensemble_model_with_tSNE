import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3  #0.5.10版
from mpld3 import plugins
from sklearn.manifold import TSNE
from keras.models import Model
from datetime import datetime
from PIL import Image
import re
import base64  #将PNG图像解析为base64编码并将其插入到HTML中
import json
import shutil
# import sys
import webbrowser
import gc

import plotly
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

import gzip

# SSG（Static Site Generation） 和 CSR（Client-Side Rendering） 渲染
# HTML 文件是靜態生成的（SSG）
#在頁面上渲染圖表和工具提示是通過客戶端 JavaScript 進行的（CSR）

"""
考慮對模型輸出量化(能夠大幅降低內存)->尚未嘗試過，不知是否可行
import numpy as np

def quantize_output(output, bit_depth=8):
    scale = (2 ** bit_depth - 1)
    min_val = np.min(output)
    max_val = np.max(output)
    
    # 將數據縮放到 [0, scale]
    quantized_output = np.clip(((output - min_val) / (max_val - min_val)) * scale, 0, scale)
    
    # 將數據轉換為整數
    quantized_output = np.round(quantized_output).astype(np.uint8)
    
    return quantized_output

# 使用量化模型進行預測
intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

# 量化輸出
quantized_intermediate_output = quantize_output(intermediate_output, bit_depth=8)

"""




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

    match = re.search(r'(c\d+|n\d+)', filename) #\d+:後面接至少1個數字  #實際要用的data
    return f"{part0} {match.group(1)}" if match else filename  #兩行簡化為一行 
   


#從html的角度出發，去發掘想匯入的圖片(必須將圖檔與html存取於相同跟目錄test_png才能讓該函式發揮作用)
def extract_relative_path_from_html(html):
    img_tag_pattern = re.compile(r'<img\s+src="([^"]+)"', re.IGNORECASE)
    matches = img_tag_pattern.findall(html)
    
    for match in matches:
        if 'test_png' in match:
            # 找到最後一個 'test_png' 的位置
            start_index = match.rfind('test_png')
            if start_index != -1:
                # 提取從 'test_png' 到結尾的路徑
                relative_path = match[start_index:]
                # 只保留 'test_png' 之後的部分
                relative_path = relative_path.split('test_png/')[-1]
                # 添加 '../../' 以從 tSNE/2024-08-27 目錄往上二級返回 test_png
                relative_path = '../../' + relative_path
                relative_path.replace('\\', '/')
    
    return relative_path

#更加靈活，不會受特定路徑結構的限制
def get_relative_path(html_path, image_path):
    # 获取共同路径
    common_path = os.path.commonpath([html_path, image_path])
    # 计算从 HTML 到共同路径的相对路径
    html_to_common = os.path.relpath(common_path, os.path.dirname(html_path))
    # 计算从共同路径到图片的相对路径
    common_to_image = os.path.relpath(image_path, common_path)
    # 组合相对路径
    return os.path.join(html_to_common, common_to_image).replace(os.sep, '/')


# def extract_relative_path_from_html(html):
#     # 正則表達式來提取所有 <img src="..."> 的 src 屬性
#     img_tag_pattern = re.compile(r'<img\s+src="([^"]+)"', re.IGNORECASE)
#     matches = img_tag_pattern.findall(html)
#      # '<img'：匹配 <img，即HTML中的圖像標籤的開頭部分。這部分是固定的，沒有變化。
#     # \s+：匹配一個或多個空白字符（如空格、Tab 等）。這裡用來匹配 <img 和 src=" 之間可能存在的一個或多個空格。
#     # (b) src="
#     # 'src="'：匹配 src="，這是HTML <img> 標籤中用來指定圖片路徑的屬性。
#     # (c) ([^"]+)
#     # ([^"]+)：這是一個捕獲群組，用來匹配 src=" 之後的圖片路徑
#     # IGNORECASE:忽略大小寫
#     valid_paths = []
#     for match in matches:
#         # 檢查是否包含 'test_png' 並提取該部分路徑
#         if 'test_png' in match:
#             # 確保找到最後一個 'test_png' 的位置(跟find()不同)
#             start_index = match.rfind('test_png')
#             if start_index != -1:
#                 # 提取從 'test_png' 到結尾的路徑
#                 relative_path = match[start_index:]
#                 # valid_paths.append(relative_path.replace('\\', '/'))
#                 relative_path.replace('\\', '/')
    
#     return relative_path

#透過二進制將數據轉換為txt(會增加33%的數據量): 只須返回base64編碼字串，<img>標籤由create_html_images_ID提供
def encode_image_to_base64(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return  encoded_string  
        
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except Exception as e: 
        print(f"Error encoding image {image_path}: {e}")
        return None




def create_html_with_two_images(encoded_string, CAM_encoded_string, true_label_info, predicted_label):
    return (f'<div class="mpld3-tooltip"><p>True Label: {true_label_info}</p>'
            f'<p>Predicted Label: {predicted_label}</p>'
            f'<img src="data:image/png;base64,{CAM_encoded_string}" width="100" height="100">'
            f'<img src="data:image/png;base64,{encoded_string}" width="100" height="100"></div>')

#測試新改的函式:添加的屬性要用" "包住
#把圖調小(看是否能成功顯示html)
def create_html_images_ID(encoded_string, CAM_encoded_string, true_label_info, predicted_label):
    return  f'''
   <div class="image-container" data-keywords="{true_label_info}">
        <img src="data:image/png;base64,{encoded_string}" width="50">     
        <img src="data:image/png;base64,{CAM_encoded_string}" width="50">
        <br>
        <span class="true-label" style="font-size: 30px;"> True Label:{true_label_info}</span> 
        <br>
        <span class="true-label" style="font-size: 30px;"> Predicted: {predicted_label} </span>
    </div>      
    '''

def add_tooltips(folder, required_count):
    tooltips = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        if os.path.isfile(image_path):
            tooltip = f'<div class="file-name">{filename}</div><img src="{image_path}">'
            tooltips.append((filename, tooltip))  # 由此得知data type
    
    while len(tooltips) < required_count:
        tooltips.extend(tooltips[:required_count - len(tooltips)])
    
    print(f"Generated {len(tooltips)} tooltips from folder: {folder}")
    return tooltips[:required_count]  # return包含tuple的list




#測試合併原圖及cam
#######################################################################################################


#要確保group_indices 在 tooltips 的索引范围内，你需要确保 tooltips 数组正确填充并且包含所有需要的图像数据
    
#預期目標:請把True label改為圖像在folder中的真實名稱->成功
#把錯分的點用紅框圈選出來，像是Predicted label為nor，卻顯示cal點/Predicted label為cal，卻顯示nor點

#把gradCAM圖結合進去 : 添加cal_CAM_folder & 添加nor_CAM_folder
#按一個鈕，切換成CAM/OCT 影像

########################################################################################################

def plot_test(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder):

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
    # print(tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    #print(layer_output_label_predict) 生成許多0 & 1 之 label matrix
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    cal_cls_error_count = nor_cls_error_count = 0
    cal_count = 0
    nor_count = 0

    # 用于存储正常和错误分类点的信息
    x_values = []
    y_values = []
    colors = []
    symbols = []
    text_labels = []


    # 建立两个列表，一个用于存储Base64编码，另一个用于存储标记信息
    image_paths = []
    labels = []
        
    # 存儲錯誤分類點的數據
    cal_misread_as_nor_x = []
    cal_misread_as_nor_y = []
    nor_misread_as_cal_x = []
    nor_misread_as_cal_y = []


    # 遍历每个组
    for _, group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):  #(j,i) = (index,value of group_indices)
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename = tooltips[i][0]
                tooltip = tooltips[i][1]
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)  # 根据前缀命名调整函数:為string
            


            color = 'blue' if predicted_labels_group[j] == 'cal' else 'green'
            symbol = 'circle'  # 默认形状

            # 根据条件设置错误分类点的颜色和形状
            if ('c' in true_label_info and predicted_labels_group[j] == 'nor'):
                color = 'red'
                cal_cls_error_count += 1
                cal_misread_as_nor_x.append(group.x.iloc[j]) 
                cal_misread_as_nor_y.append(group.y.iloc[j])
               
            elif ('n' in true_label_info and predicted_labels_group[j] == 'cal'):
                color = 'rgb(255, 102, 102)' #淺紅
                #symbol = 'cross' #原本為x
                nor_cls_error_count += 1
                nor_misread_as_cal_x.append(group.x.iloc[j])
                nor_misread_as_cal_y.append(group.y.iloc[j])
               


            x_values.append(group.x.iloc[j])
            y_values.append(group.y.iloc[j])
            colors.append(color)
            symbols.append(symbol)
            

            # 从 HTML 标签中提取图像路径
            image_path = extract_relative_path_from_html(tooltip) # 為list!


            sample_label = f'<br>True Label: {true_label_info}<br>Predicted Label: {predicted_labels_group[j]}'
           
            text_labels.append(sample_label)
            image_paths.append(image_path)

            #當資料量過大時，使用Base64編碼的影像可能會導致網頁性能下降，甚至無法正確顯示圖像。
            # 這是因為每個Base64編碼的影像都是一個非常長的字串，當有大量數據時，這些字串可能會佔用大量記憶體和網頁渲染資源

            #使用相對路徑解決看看!
            # 優勢：
            # 減少HTML文件大小：相對路徑方式不會將圖片數據嵌入HTML，僅包含路徑信息，因此HTML文件非常小。
            # 降低內存消耗：相對路徑圖片文件存儲在磁碟或服務器上，瀏覽器在需要時載入，不會一次性消耗大量內存。
            # 靈活性：可以輕鬆管理、更新圖片文件而不需要更改HTML文件的內容。
            # 注意事項：
            # 相對路徑：確保HTML文件和圖片之間的相對路徑是正確的，這樣瀏覽器才能找到並顯示圖片。
            # 圖片存儲位置：所有圖片必須存放在與HTML文件相對應的目錄中，或者在網頁中指定正確的URL。



            # 更新计数
            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    # print(filename) #02974_n1_frame751.png
        print(tooltip) # <div class="file-name">02974_n1_frame751.png</div><img src=該圖片路徑>
        print(image_paths) #檔案總管中的圖片路徑
    # print(img,sample_label) #確定有label資訊，只是被擠到base64編碼後面
    

    
        
    # 创建 Plotly 图形
    fig = go.Figure()

    # 因為每個點有懸停功能，所以要與Nor/Cal的標示分開處理，不同的點對應不同顏色)
    #記得隱藏trace
    #問題可能出在此，導致原始圖像無法顯示
    #是因為scatter無法用來顯示圖像，
    fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='markers',
    marker=dict(
        size=5,  # 對應 Matplotlib 中的 s=5
        color=colors,  # 對應 Matplotlib 中的 c=all_colors
        opacity=0.8,  # 對應 Matplotlib 中的 alpha=0.8
        symbol='circle',  # 對應 Matplotlib 中的 marker='o'
    ),
    text=text_labels,
    customdata=image_paths,  #只接受list,array,series

    # customdata=image_encoded_strings,  # 用于存储 Base64 编码字符串
    # text=labels,

    #實現鼠標懸停功能
    #該屬性只能用來顯示文本->圖像要用別的屬性!(盡量不用base64，否則占用記憶體)
    hovertemplate="""
    <b>%{text}</b><br><br>
    <img src= "%{customdata}" style="width:100px;height:100px;"><br>
    <extra></extra>
    """,  
        
    # visible=False:想辦隱藏多餘的trace 0!
    showlegend=False  # 隐藏图例中的条目

    ))

    

    # 添加图例
    fig.add_trace(go.Scatter(
        x=[None],  # 不显示数据点
        y=[None],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',  # 颜色
            symbol='circle'
        ),
        name='Cal'  # 图例中的标签
    ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            symbol='circle'
        ),
        name='Nor'
    ))

    # 添加 Cal misread as Nor 點
    fig.add_trace(go.Scatter(
        x=cal_misread_as_nor_x,
        y=cal_misread_as_nor_y,
        mode='markers',
        marker=dict(color='red', symbol='cross', size=10),
        name='Cal misread as Nor'
    ))

    # 添加 Nor misread as Cal 點
    fig.add_trace(go.Scatter(
        x=nor_misread_as_cal_x,
        y=nor_misread_as_cal_y,
        mode='markers',
        marker=dict(color='rgb(255, 102, 102)', symbol='cross', size=10),
        name='Nor misread as Cal'
    ))

    # 更新布局:圖像調小，才不會佔據<p>標籤內容
    fig.update_layout(
        title='t-SNE Plot',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend=dict(
            title='Legend',
            itemsizing='constant'
        ),
        width=1000,  # 图形宽度
        height=750  # 图形高度
    )

    # 显示图形於notebook介面
    fig.show()

    


    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count 
    tSNE_acc = (total_points - cls_error_count) / total_points


    # 將圖形轉換為 HTML 字符串，避免版本相容性問題
    html_str = to_html(fig, full_html=False, include_plotlyjs='cdn') #指向plotly.js之cdn版(但需要連網才能加載圖)，'cdn'改為True更棒(但文件較大)


    

    html_str_with_style=(f"""
    <html>
    <head>
        <title>tSNE Plot</title>

    </head>
    <body>
    <h1>tSNE Plot</h1>
    <p style="font-size: 30px;">tSNE Accuracy: {tSNE_acc:.2f}</p>'
    <p style="font-size: 30px;">nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>'
    <p style="font-size: 30px;">cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>'
    <p style="font-size: 30px;">Number of Cal points: {cal_count}</p>'
    <p style="font-size: 30px;">Number of Nor points: {nor_count}</p>'
    <div style="display: flex; justify-content: center;"><div>{html_str}</div></div>'
    </body>
    </html>
                    """)                                                   
    
     
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE.html')
    with open(save_path, 'w') as file:  
        file.write(html_str_with_style)


    # 自动打开 HTML 文件
    webbrowser.open(f'file://{os.path.abspath(save_path)}')





#用此code檢查tooltips參數，因為base64編碼疑似有誤
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
            # encoded_string = encode_image_to_base64(image_path)
            encoded_string = compress_image(image_path)

            # CAM image
            CAM_image_path = extract_image_path_from_html(CAM_tooltip)
            # CAM_encoded_string = encode_image_to_base64(CAM_image_path)
            CAM_encoded_string = compress_image(CAM_image_path)

            # Generate HTML content with both images :多了CAM_encoded_string
            html_content = create_html_with_two_images(encoded_string, CAM_encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)


            if ('c' in true_label_info and predicted_labels_group[j] == 'nor'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                
            elif ('n' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
               

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    print(filename)
    print(tooltip)
    print(image_path)

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
    html_str_with_style = (f'<p style="font-size: 30px;>tSNE Accuracy: {tSNE_acc:.2f}</p>'
                           f'<p style="font-size: 30px;>nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>'
                           f'<p style="font-size: 30px;>cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>'
                           f'<p style="font-size: 30px;>Number of Cal points: {cal_count}</p>'
                           f'<p style="font-size: 30px;>Number of Nor points: {nor_count}</p>' 
                           f'<div style="display: flex; justify-content: center;"><div>{html_str}</div></div>' #內容置中
                          )
    
    # 使用 gzip 壓縮 HTML 內容
    compressed_html_str = gzip.compress(html_str_with_style.encode('utf-8'))

    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE_combined.html')
    # with open(save_path, 'w') as file:
    #     file.write(html_str_with_style) #錯在最後一行!

    with open(save_path, 'wb') as file: # 以二進位模式才能打開壓縮後的byte檔
        file.write(compressed_html_str) 

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

            # if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'): 
            #     ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
            #     cal_cls_error_count += 1
                 
            # elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
            #     ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
            #     nor_cls_error_count += 1
                 

            # if predicted_labels_group[j] == 'cal':
            #     cal_count += 1
            # else:
            #     nor_count += 1

            if ('c' in true_label_info and predicted_labels_group[j] == 'n'): 
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                cal_cls_error_count += 1
                 
            elif ('n' in true_label_info and predicted_labels_group[j] == 'c'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', color='red', s=10)
                nor_cls_error_count += 1
                 

            if predicted_labels_group[j] == 'c':
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
    js_file_path = r'E:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\reqiured_funcs\keysearch.js'
    # 確認 save_path 已正確設置
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)

    # 確認保存目錄已存在，若不存在則創建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 將 keysearch.js 複製到保存目錄中，跟tSNE_combined.html在同路徑才能引入
    shutil.copy(js_file_path, save_dir)

    save_path = os.path.join(save_dir, 'tSNE_combined.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)

#測試此函式
#####################################################################
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from plotly.io import to_html

def get_relative_path(image_path, base_dir):
    # 規範化路徑，確保分隔符一致
    image_path = os.path.normpath(image_path)
    base_dir = os.path.normpath(base_dir)

    # 確保兩個路徑位於同一磁碟分區
    if os.path.splitdrive(image_path)[0] == os.path.splitdrive(base_dir)[0]:
        # 在同一磁碟分區，返回相對路徑
        return os.path.relpath(image_path, base_dir)
    else:
        # 否則返回絕對路徑
        return image_path

def plot_SUCK(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder):
    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    # 降維成2維數組
    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))
    
    # 提取標籤
    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = count_files_in_folder(cal_folder)
    required_nor_count = count_files_in_folder(nor_folder)

    # 生成 tooltips
    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = np.array(cal_tooltips + nor_tooltips)

    prediction = model.predict(test_data)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in np.argmax(prediction, axis=1)]

    cal_cls_error_count = nor_cls_error_count = 0
    cal_count = nor_count = 0
    x_values, y_values, colors, symbols, text_labels, image_paths = [], [], [], [], [], []
    cal_misread_as_nor_x, cal_misread_as_nor_y = [], []
    nor_misread_as_cal_x, nor_misread_as_cal_y = [], []

    for _, group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue
            try:
                filename, tooltip = tooltips[i]   #tooltips為tuple組成的list(觀察add_tooltips的回傳值)
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)

            color = 'blue' if predicted_labels_group[j] == 'cal' else 'green'
            symbol = 'circle'

            if 'c' in true_label_info and predicted_labels_group[j] == 'nor':
                color = 'red'
                cal_cls_error_count += 1
                cal_misread_as_nor_x.append(group.x.iloc[j])
                cal_misread_as_nor_y.append(group.y.iloc[j])
            elif 'n' in true_label_info and predicted_labels_group[j] == 'cal':
                color = 'rgb(255, 102, 102)'
                nor_cls_error_count += 1
                nor_misread_as_cal_x.append(group.x.iloc[j])
                nor_misread_as_cal_y.append(group.y.iloc[j])

            x_values.append(group.x.iloc[j])
            y_values.append(group.y.iloc[j])
            colors.append(color)
            symbols.append(symbol)

            # 提取相對路徑
            image_path = extract_relative_path_from_html(tooltip)
            relative_path = get_relative_path(save_dir, image_path[0])  # 使用相對路徑

            sample_label = f'<br>True Label: {true_label_info}<br>Predicted Label: {predicted_labels_group[j]}'
            text_labels.append(sample_label)
            image_paths.append(relative_path)

            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    # 创建 Plotly 图形
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            opacity=0.8,
            symbol='circle',
        ),
        text=text_labels,
        customdata=image_paths,  # 使用相對路徑
        hovertemplate="""
        <b>%{text}</b><br><br>
        <img src="%{customdata}" style="width:100px;height:100px;"><br>
        <extra></extra>
        """,
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='blue', symbol='circle'),
        name='Cal'
    ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='Nor'
    ))

    fig.add_trace(go.Scatter(
        x=cal_misread_as_nor_x,
        y=cal_misread_as_nor_y,
        mode='markers',
        marker=dict(color='red', symbol='cross', size=10),
        name='Cal misread as Nor'
    ))

    fig.add_trace(go.Scatter(
        x=nor_misread_as_cal_x,
        y=nor_misread_as_cal_y,
        mode='markers',
        marker=dict(color='rgb(255, 102, 102)', symbol='cross', size=10),
        name='Nor misread as Cal'
    ))

    fig.update_layout(
        title='t-SNE Plot',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend=dict(title='Legend', itemsizing='constant'),
        width=1000,
        height=750
    )

    fig.show()

    total_points = len(df)
    cls_error_count = nor_cls_error_count + cal_cls_error_count
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = to_html(fig, full_html=False, include_plotlyjs='cdn')

    html_str_with_style = f"""
    <html>
    <head>
        <title>tSNE Plot</title>
    </head>
    <body>
    <h1>tSNE Plot</h1>
    <p style="font-size: 30px;">tSNE Accuracy: {tSNE_acc:.2f}</p>
    <p style="font-size: 30px;">nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>
    <p style="font-size: 30px;">cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>
    <p style="font-size: 30px;">Number of Cal points: {cal_count}</p>
    <p style="font-size: 30px;">Number of Nor points: {nor_count}</p>
    <div style="display: flex; justify-content: center;"><div>{html_str}</div></div>
    </body>
    </html>
    """

    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)

























