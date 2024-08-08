
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

     searchBox.addEventListener('input', function() {{
         console.log('Input event triggered'); // Debug
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
                 matchedFileNames.push({ index: i, name: fileNames[i] });
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