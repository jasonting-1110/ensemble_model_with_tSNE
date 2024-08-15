
#將電腦檔案總管的檔案上傳到github
##進入專案folder
cd ensemble_model_with_tSNE
##新增並進入subfolder
cd data_preprocessing
mkdir data_preprocessing
##移動檔案
move 840_OCT_3_v2.ipynb 840_OCT_loopTest.ipynb p_avg_1.txt resize_normalization.ipynb data_preprocessing\
##添加 提交變更
git add .
git commit -m "Moved files to data_preprocessing folder"
##推送檔案到GitHub的master分支
git push origin master 
