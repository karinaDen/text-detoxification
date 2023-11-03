dataset for baseline was taken from https://www.kaggle.com/datasets/nicapotato/bad-bad-words


steps for solution:
0) create train and eval datasets
1) create delite baseline :
    1.1) write download dataset form Kaggle, unzip it, and represent into txt file
    1.2) delite all matched words with badwords.txt from eval dataset 
2) try to parallel train gpt-2 model
3) try to train T5 model as it was in the lab
4) compare results