
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
         tooltip.style.display = 'none'; //最初被隱藏，只有在滑鼠懸停在搜索結果上時才會顯示工具提示
         tooltip.style.background = 'white';
         tooltip.style.border = '1px solid black';
         tooltip.style.padding = '5px';
         tooltip.style.zIndex = '1000';
         document.body.appendChild(tooltip);
     }}

     //當用戶在搜索框中輸入內容時，觸發 input 事件
     searchBox.addEventListener('input', function() {{
         console.log('Input event triggered'); // Debug
         var searchTerm = this.value.toLowerCase().trim();

         //檢查是否存在 searchResults 列表元素，如果不存在就創建一個 ul 列表，並插入到搜索框的後面
         if (!searchResults) {{
             searchResults = document.createElement('ul');
             searchResults.id = 'search-results'; 
             this.parentNode.insertBefore(searchResults, this.nextSibling);
         }}
         //清空 searchResults 的內容，以便重新顯示匹配結果
         searchResults.innerHTML = '';
         
         if (searchTerm === '') {{
             return;
         }}

         /*遍歷 fileNames 列表，將每個文件名轉換為小寫，並檢查它是否包含 searchTerm。
           如果匹配，將文件名和其索引加入到 matchedFileNames 陣列中*/
         var matchedFileNames = [];
         for (var i = 0; i < fileNames.length; i++) {{
             var fileName = fileNames[i].toLowerCase();
             if (fileName.includes(searchTerm)) {{
                 matchedFileNames.push({ index: i, name: fileNames[i] });
             }}
         }}

         /*為每個匹配的文件名創建一個 li 元素，並將其文本內容設置為文件名。
           設置 li 的樣式，讓它在滑鼠懸停時呈現為可點擊狀態*/
         matchedFileNames.forEach(function(match) {{
             var li = document.createElement('li');  //沒有就自己創建!!
             li.textContent = match.name; //根據 matchedFileNames.push({ index: i, name: fileNames[i] })中的name屬性找到fileNames[i]
             li.style.cursor = 'pointer';

             /*當滑鼠懸停在 li 上時，顯示工具提示 (tooltip)。
               工具提示的內容是對應的 combinedLabels，並且根據
               滑鼠的位置來動態設置工具提示的位置，
               透過在li中的各種操作，把matchedFileNames中的元素與tooltip連結*/
             li.onmouseover = function(event) {{
                 tooltip.innerHTML = combinedLabels[match.index]; 
                 tooltip.style.display = 'block';
                 tooltip.style.left = event.pageX + 10 + 'px';
                 tooltip.style.top = event.pageY + 10 + 'px';
             }};
             
             li.onmouseout = function() {{
                 tooltip.style.display = 'none';
             }};

             //當用戶點擊某個搜索結果時，將該文件名填入搜索框，並清空搜索結果列表。
             li.onclick = function() {{
                 searchBox.value = match.name; //對應li.textContent = match.name
                 searchResults.innerHTML = '';
                 
             }};
             
             searchResults.appendChild(li);
         }});
     }});

     // 點擊頁面搜索框和搜索結果之外的其他地方時隱藏搜索結果
     document.addEventListener('click', function(event) {{
         if (event.target !== searchBox && event.target !== searchResults) {{
             searchResults.innerHTML = '';
         }}
     }});
 }}

// 調用設置函數
setupSearchAndTooltip();   
