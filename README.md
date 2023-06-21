# KDD-CUP-2023-solution
In KDD Cup '23, the method we used received 6th place in Task 1 and 4th place in Task 2.
# Requirements:
The requirements of experiment environment are listed in requirements.txt.
# Overview
The method we used has two steps, including recall and rerank.The overall flow is shown in the figure.
![image](https://github.com/karrich/KDD-CUP-2023-solution/assets/57396778/8c65c963-9673-4725-b1df-d7114b1716ae)
# Steps to reproduce the results
## 1. Data preprocessing
1. `task1 data process.ipynb`  
2. `task2 data process.ipynb`  
3. `title embedding.ipynb`  
4. nn model/`nn data process.ipynb`  
## 2. train mlp model
1. `init id embedding.ipynb`  
2. `training.ipynb` 
## 3. Recall
For Task 1:  
1. rule recall/`rule recall-task1.ipynb`    
2. nn model/`nn recall-task1.ipynb`  
    
For Task 2:  
1. rule recall/`rule recall-task2.ipynb`  
2. nn model/`nn recall-task2.ipynb`
## 4. Generate features for Rerank
For Task 1: `xgb-task1/pro feature.ipynb`  
For Task 2: `xgb-task2/pro feature.ipynb`
## 5. Train the xgboost model for Rerank and Inference
For Task 1: `xgb-task1/training.ipynb`  
For Task 1: `xgb-task2/training.ipynb`


