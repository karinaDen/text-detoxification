<h1 align="center">Text detoxification Assignment 
<h1 align="center">Innipolis [F23] PML&DL course</h1> 

___

## 🎓 Student 
**Name**: Karina Denisova 
<br/>
**Email**: k.denisova@innopolis.university
<br/>
**Group numer**: BS21-DS-01

## 🗒 Project description 

Project aims to address the challenge of transforming text with toxic style into text with a neutral style while preserving the same underlying meaning. 


## 🤾‍♀️ Run the project

1) Clone the repository
2) Install requirements
```
pip install -r requirements.txt
```
3) For dataset creation run:
```
python src/data/make_dataset.py
```
4) For baseline model run:
```
python src/models/baseline/predict_baseline.py 
```
4.1) For baseline evaluation run:
```
python src/metrics/metrics.py --inputs data/interim/translation.txt --preds data/interim/baseline.txt
```
5) For training T5-small model run:
```
python src/models/T5/T5_model_train.py
```
5.1) For T5-small model evaluation run:
```
python src/metrics/metrics.py --inputs data/interim/test_translation.txt --preds data/interim/result.txt
```
5.2) To download T5-small model weights run:
```
python src/data/load_weights.py
```


Also you can run all modelt and thesting with examples from the [`notebooks`](https://github.com/karinaDen/text-detoxification/tree/main/notebooks) folder.

