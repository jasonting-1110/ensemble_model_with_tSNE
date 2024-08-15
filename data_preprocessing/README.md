
#將電腦檔案總管的檔案上傳到github
cd ensemble_model_with_tSNE
mkdir data_preprocessing
move 840_OCT_3_v2.ipynb 840_OCT_loopTest.ipynb p_avg_1.txt resize_normalization.ipynb data_preprocessing\
git status
git commit -m "Moved files to data_preprocessing folder"
git push origin master #推送檔案到GitHub的master分支
