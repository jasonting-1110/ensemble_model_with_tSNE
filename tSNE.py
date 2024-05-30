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

#函式參數的輸入
#model:ensemble_model layer_name 'dense_5' test_data=[test_data, test_data]

#一個存取許多圖像的list
#def add_tooltips(folder):
def add_tooltips(folder,required_count):    
    tooltips = []
# Iterate over cal_folder
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        if os.path.isfile(image_path):
            tooltips.append((filename,f'<img src="{image_path}">'))
            #tooltips.append((filename,f'<img src="{image_path}" width="402" height="292">')) width="402" height="292"影響路徑讀取至html
    #print("Length of tooltips:", len(tooltips))

    # 如果tooltips数量不足，进行补充
    while len(tooltips) < required_count:
        tooltips.extend(tooltips[:required_count - len(tooltips)])
    
    # 打印生成的tooltips数量
    print(f"Generated {len(tooltips)} tooltips from folder: {folder}")

    return tooltips[:required_count]

    #return tooltips


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
    start_index = html.find('src="') + len('src="') #.find找出 src="字串在img標籤中的index  從src="中的"為起始點
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

#要確保group_indices 在 tooltips 的索引范围内，你需要确保 tooltips 数组正确填充并且包含所有需要的图像数据
    
#預期目標:請把True label改為圖像在folder中的真實名稱->成功
#把錯分的點用紅框圈選出來，像是Predicted label為nor，卻顯示cal點/Predicted label為cal，卻顯示nor點

def plot_tSNE_best(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, label_name=['Nor', 'Cal']):

    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = 1976
    required_nor_count = 1874
    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    if len(tooltips) < intermediate_output.shape[0]:
        raise ValueError("The number of tooltips is less than the number of data points. Ensure that the tooltips cover all data points.")

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    combined_labels = []
    plotted_labels = {}
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    

    for label, group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        label_name = 'Cal' if label == 1 else 'Nor'

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

            # # 读取图像并进行base64编码
            # with open(tooltip, "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # # 使用base64编码的图像插入到HTML中
            # combined_labels.append(f'<img src="data:image/png;base64,{encoded_string}"><br>True Label: {true_label_info}<br>Predicted Label: {predicted_labels_group[j]}')

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
    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10))

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
    html_str_with_style = (f'<p>tSNE Accuracy: {tSNE_acc:.2f}</p>'
                           f'<p>nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>'
                           f'<p>cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>'
                           f'<p>Number of Cal points: {cal_count}</p>'
                           f'<p>Number of Nor points: {nor_count}</p>'   
                           f'<div style="display: flex; justify-content: center;"><div>{html_str}</div></div>')

    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)



def plot_tSNE_tuning(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, label_name=['Nor', 'Cal']):

    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = 1976
    required_nor_count = 1874
    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    if len(tooltips) < intermediate_output.shape[0]:
        raise ValueError("The number of tooltips is less than the number of data points. Ensure that the tooltips cover all data points.")

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    combined_labels = []
    plotted_labels = {}
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    all_colors = []
    

    for label, group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        label_name = 'Cal' if label == 1 else 'Nor'

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

            # # 读取图像并进行base64编码
            # with open(tooltip, "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            # # 使用base64编码的图像插入到HTML中
            # combined_labels.append(f'<img src="data:image/png;base64,{encoded_string}"><br>True Label: {true_label_info}<br>Predicted Label: {predicted_labels_group[j]}')

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
    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10))

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
    html_str_with_style = (f'<p>tSNE Accuracy: {tSNE_acc:.2f}</p>'
                           f'<p>nor_cls_error_point: {nor_cls_error_count}/{total_points}</p>'
                           f'<p>cal_cls_error_point: {cal_cls_error_count}/{total_points}</p>'
                           f'<p>Number of Cal points: {cal_count}</p>'
                           f'<p>Number of Nor points: {nor_count}</p>'   
                           f'<div style="display: flex; justify-content: center;"><div>{html_str}</div></div>')

    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)



def plot_tSNE_opt(model, layer_name, test_data, test_labels, save_dir, cal_folder, nor_folder, label_name=['Nor', 'Cal']):
    batch_size = 64
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

    Y = TSNE(n_components=2, init='random', random_state=0, perplexity=10, verbose=1).fit_transform(
        intermediate_output.reshape(intermediate_output.shape[0], -1))

    layer_output_label = np.argmax(test_labels, axis=1)
    df = pd.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
    groups = df.groupby('label')

    required_cal_count = 1976
    required_nor_count = 1874
    cal_tooltips = add_tooltips(cal_folder, required_cal_count)
    nor_tooltips = add_tooltips(nor_folder, required_nor_count)
    tooltips = cal_tooltips + nor_tooltips
    tooltips = np.array(tooltips)

    if len(tooltips) < intermediate_output.shape[0]: #intermediate_output.shape[0]: 1st dim(sample_num in test_data)
        raise ValueError("The number of tooltips is less than the number of data points. Ensure that the tooltips cover all data points.")

    prediction = model.predict(test_data)
    layer_output_label_predict = np.argmax(prediction, axis=1)
    layer_output_label_predict = ['nor' if label == 0 else 'cal' for label in layer_output_label_predict]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.margins(0.05)

    cal_cls_error_count = nor_cls_error_count = 0
    combined_labels = []
    #plotted_labels = {}
    cal_count = 0
    nor_count = 0

    all_x = []
    all_y = []
    #all_colors = []


    # 創建兩個空的列表，用於存儲 'cal' 和 'nor' 類別的資料點
    cal_points = []
    nor_points = []

    for label, group in groups:
        group_indices = group.index.tolist()
        predicted_labels_group = np.array([layer_output_label_predict[i] for i in group_indices])

        label_name = 'Cal' if label == 1 else 'Nor'

        for j, i in enumerate(group_indices):
            if i >= len(tooltips) or i < 0:
                continue

            # for i, (filename, path) in enumerate(tooltips):
            #     print(i,(tooltips[i][0], tooltips[i][1])) 確認 tooltips[i][0] tooltips[i][1]真的對應filename以及path
            try:
                filename = tooltips[i][0]           
                tooltip = tooltips[i][1] #對應圖檔路徑
            except (KeyError, IndexError):
                print(f"Invalid index at position {i}, skipping.")
                continue

            true_label_info = extract_info_from_filename(filename)
            color = 'b' if predicted_labels_group[j] == 'cal' else 'g'

            if predicted_labels_group[j] == 'cal':
                cal_points.append((group.x.iloc[j], group.y.iloc[j]))
            else:
                nor_points.append((group.x.iloc[j], group.y.iloc[j]))

            all_x.append(group.x.iloc[j])
            all_y.append(group.y.iloc[j])
            #all_colors.append(color)

             # 从 HTML 标签中提取图像路径
            image_path = extract_image_path_from_html(tooltip)

            # 编码图像为Base64
            encoded_string = encode_image_to_base64(image_path)

            # 生成HTML内容
            html_content = create_html_with_image(encoded_string, true_label_info, predicted_labels_group[j])
            combined_labels.append(html_content)

            #combined_labels.append(f'<img src=r"{tooltip}"><br>True Label: {true_label_info}<br>Predicted Label: {predicted_labels_group[j]}')
        
            #原本cal nor圖例會被覆蓋->3個點出現的原因與此無關!
            if ('supra' in true_label_info and predicted_labels_group[j] == 'nor'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', edgecolor='red', facecolor='none', s=20)
                cal_cls_error_count += 1
            elif ('NOR' in true_label_info and predicted_labels_group[j] == 'cal'):
                ax.scatter(group.x.iloc[j], group.y.iloc[j], marker='x', edgecolor='red', facecolor='none', s=20)
                nor_cls_error_count += 1

            

    


            if predicted_labels_group[j] == 'cal':
                cal_count += 1
            else:
                nor_count += 1

    color_map = {'cal': 'b', 'nor': 'g'}
    scatter_colors = [color_map[label] for label in predicted_labels_group]
    scatter = ax.scatter(all_x, all_y, c=scatter_colors, marker='o', alpha=0.8,s=30) #s:size

        # 在圖上添加 'cal' 和 'nor' 類別的點
    # cal_scatter = ax.scatter(*zip(*cal_points), color='b', marker='o', alpha=0.8, s=30, label='Cal')  
    # nor_scatter = ax.scatter(*zip(*nor_points), color='g', marker='o', alpha=0.8, s=30, label='Nor')
    # """
    # *cal_points :把cal_points拆成許多tuple
    # zip(*cal_points) : 分別把x,y座標都zip成一個tuple,形成包含兩個大tuple的list
    # *zip(*cal_points) : unpack list,產出x,y的兩個大tuple
    # """
    # #scatter=cal_scatter + nor_scatter PathCollection can't add

    plugins.connect(fig, plugins.PointHTMLTooltip(scatter, labels=combined_labels, voffset=10, hoffset=10))

    #設置圖例
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label='Cal', markerfacecolor='b', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Nor', markerfacecolor='g', markersize=10)]
    ax.legend(handles=legend_handles, loc='best')
    #plt.legend()
    plt.title('tSNE')
    plt.tight_layout()
    plt.show()


    cls_error_count=cal_cls_error_count + nor_cls_error_count

    total_points = len(df)
    tSNE_acc = (total_points - cls_error_count) / total_points

    html_str = mpld3.fig_to_html(fig)
    html_str_with_style = (f'<p>tSNE Accuracy: {tSNE_acc:.2f}</p>'
                           f'<p>nor_cls_error_points: {nor_cls_error_count}/{total_points}</p>'  #還是有些NOR錯分為cal
                           f'<p>cal_cls_error_points: {cal_cls_error_count}/{total_points}</p>'
                           f'<p>Number of Cal points: {cal_count}</p>'
                           f'<p>Number of Nor points: {nor_count}</p>'   
                           f'<div style="display: flex; justify-content: center;"><div>{html_str}</div></div>')

    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'tSNE.html')
    with open(save_path, 'w') as file:
        file.write(html_str_with_style)





