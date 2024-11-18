
/*var fileNames = {file_names_json};  //not defined(如何讓js引入python中的file_names_json)
var combinedLabels = {combined_labels_json}; */

 function setupSearchAndTooltip() {{
    
    /* 使用全局變量(即使沒有這兩個變量事先定義於此
    外部js檔，code仍能成功執行，
    在瀏覽器環境中，所有全局變數都是 window 對象的屬性。
    當你在 HTML 文件中通過 window.fileNames 定義變數時，
    它們會自動成為全局變數，可以在任何地方通過 window 對象訪問)*/

    //var fileNames = window.fileNames;  
    //var combinedLabels = window.combinedLabels;
     //console.log('Setting up search and tooltip'); //Debug info
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

     //可以考慮用arrow func => 簡化code
     searchBox.addEventListener('input', function() {{
         console.log('Input event triggered'); // Debug
         var searchTerm = this.value.toLowerCase().trim();  // searchTerm = this.value: 在searchBox的輸入
         
        // 當用戶在searchBox(this.)輸入資訊時，如果searchResults不存在，就動態創建<ul>，並插入到searchBox後面
        if (!searchResults) {
            searchResults = document.createElement('ul'); //動態創建新<ul>
            searchResults.id = 'search-results'; 
            this.parentNode.insertBefore(searchResults, this.nextSibling);  //將新建<ul>插入到searchBox後面
        }

        // 在操作之前检查 searchResults 是否有效(searchResults 在某些情况下是 null，因此当你尝试设置 innerHTML 属性时就会导致 TypeError)
        if (searchResults) {
            searchResults.innerHTML = ''; // 只有在 searchResults 存在的情况下才清空内容
        } else {
            console.error('searchResults is not defined'); // Debug info
            return; // 如果没有 searchResults，直接返回
        }
         
         if (searchTerm === '') {{
             return;
         }}
         
        /* 過濾匹配的文件名 (matchedFileNames)： 
        代碼會遍歷 fileNames陣列(也就是true_label_info)，
        檢查每個文件名是否包含 searchTerm（用戶輸入的關鍵詞）。
        匹配到的文件名會被加入 matchedFileNames 陣列中 */
         var matchedFileNames = [];
         for (var i = 0; i < fileNames.length; i++) {{
             var fileName = fileNames[i].toLowerCase();
             if (fileName.includes(searchTerm)) {{
                 matchedFileNames.push({ index: i, name: fileNames[i] });
             }}
         }}
         
         //在searchResults中顯示匹配結果  match: matchedFileNames中的某個元素顯示為文本內容
         //將所有匹配結果以清單<li>列出
         //由於fileNames為list，感覺可以直接遍歷! 出錯!!
            // fileNames.forEach(function(match) {{
            //     var li = document.createElement('li');  
            //     li.textContent = match.name; // 將匹配的文件名顯示為<li>的文本   match.name: fileNames[i]
            //     li.style.cursor = 'pointer';
        
         //在searchResults中顯示匹配結果  match: matchedFileNames中的某個元素顯示為文本內容
         //將所有匹配結果以清單<li>列出
         //當用戶在搜索框中輸入內容時，並不是所有的文件名都應該顯示在結果列表中。你需要根據用戶輸入的內容過濾 fileNames，只顯示那些符合搜索條件的文件名
             matchedFileNames.forEach(function(match) {{
             var li = document.createElement('li');  
             li.textContent = match.name; // 將匹配的文件名顯示為<li>的文本   match.name: fileNames[i]
             li.style.cursor = 'pointer';
             
            
             // 懸停時顯示對應的 combinedLabels tooltip
             li.onmouseover = function(event) {{
                 tooltip.innerHTML = combinedLabels[match.index]; 
                 tooltip.style.display = 'block';
                 tooltip.style.left = event.pageX + 10 + 'px';
                 tooltip.style.top = event.pageY + 10 + 'px';
             }};
             
             // 離開匹配項時，隱藏對應的 tooltip
             li.onmouseout = function() {{
                 tooltip.style.display = 'none';
             }};
             
             // 點擊時將文件名填充到 searchBox
             li.onclick = function() {{
                 searchBox.value = match.name;
                 searchResults.innerHTML = '';
                 // 可以在這裡添加其他點擊後的操作
             }};
             
             //將<li>元素添加到 searchResults（一個 HTML <ul> 容器）中，用來生成一個列表
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




/*  為什麼需要先使用 matchedFileNames 這樣的中間變量來存儲匹配結果，而不能直接遍歷 fileNames 列表。

原因是：
過濾文件名：當用戶在搜索框中輸入內容時，並不是所有的文件名都應該顯示在結果列表中。你需要根據用戶輸入的內容過濾 fileNames，只顯示那些符合搜索條件的文件名。因此，先過濾出匹配的文件名是必要的。

例如，當用戶輸入 "example" 時，你希望只顯示包含 "example" 的文件名，而不是所有的文件名。
匹配結果的臨時存儲： matchedFileNames 是臨時存儲所有符合搜索條件的文件名。這樣可以讓你清晰地分離出「過濾」和「顯示」的邏輯：

第一個步驟是過濾符合條件的文件名。
第二個步驟是將這些過濾後的文件名顯示出來。

/* 

searchBox
當用戶在 searchBox 中輸入文字時，觸發 input 事件，根據輸入的 searchTerm 過濾 fileNames

matchedFileNames
從 fileNames 中篩選出包含 searchTerm 的匹配項，生成 matchedFileNames

searchResults
動態創建 <ul> 元素，作為結果顯示的容器，位於 searchBox 的下方。
迭代 matchedFileNames，為每個匹配項創建 <li>，並將其添加到 searchResults。 

接著從此製作ppt
Tooltip
當用戶懸停在某個匹配的 <li> 項時，顯示對應的 combinedLabels 作為工具提示。
根據鼠標的位置動態顯示工具提示的位置，並根據 combinedLabels 數組顯示對應的內容。

Event Handling
當用戶懸停或點擊匹配結果時，動態顯示工具提示或將選中的文件名填入 searchBox，並隱藏搜索結果

[User Input] ----(Input Event)----> [Search Box (searchBox)]
                                               |
                                               v
                            [Filtered File Names (matchedFileNames)]
                                               |
                          For Each Matched File Name:
                                               |
                                               v
          [Create <li> element with match.name] -----> [Search Results List (searchResults)]
                                               |
                                               v
                   Hover Event (onmouseover) -----> [Tooltip Display (combinedLabels)]
                                               |
                                               v
             Click Event (onclick) ----> [Fill searchBox with match.name and clear searchResults]


*/