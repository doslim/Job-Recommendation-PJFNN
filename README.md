# Job Recommendation PJFN
 This repository contains several baseline models and PJFNN (Person-Job Fit Neural Network from the paper: [Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning](https://dl.acm.org/doi/abs/10.1145/3234465)) on the datasets from [Job Recommendation Challenge](https://www.kaggle.com/c/job-recommendation).


## Requirements and Structures
The following packages are required.
- torch-1.11.0
- nltk-3.6.5
- gensim-3.8.3

The structure of our project is as follows.
- code: contain all the code in the form of jupyter notebooks.
    -  build\_dataset.ipynb: a set of functions to build a small dataset from the original dataset.
    -  baseline.ipynb: build and evaluate baseline models on the person-job fit task (binary classification).
    -  ranking.ipynb: build and evaluate baseline models on the job recommendation task.
    -  pjfnn.ipynb: build and evaluate modified PJFNN on the two tasks.
- report.pdf: a brief report (in Chinese) of the details of our implementations and implementation results.

## Usage
To run our codes, you should first download the datasets from [Kaggle](https://www.kaggle.com/c/job-recommendation). And rewrite the data path in our codes (mostly in build\_dataset.ipynb).

## Results

We conduct our experiments on a subset of the original dataset, where we set the ```WindowID=6```. Since the data only have the application records (positive samples), we carry out randomly negative sampling to build a dataset such that the ratio of postive and negative samples are 1:1. More details can be found in the build\_dataset.ipynb.

We consider the following two tasks:
- Person-Job fit: the inputs are in the forms like ```(job, person)```, and the outputs are binary labels indicating that whether the people are suitable for the jobs.
- Job Recommendation: the models are required to rank 20 randomly selected jobs for the given person.

We use classic metrics for binary classification in the person-job fit task and hit rate in the recommendation task.

### Person-Job Fit
|      | Accuracy | Precision | Recall | F1-score | AUC |
| :---- | ----: | ----: | ----: | ----: | ----: |
| Linear Regression | 0.544 |	0.546 | 0.522 | 0.534 | 0.549 |
| Logistic Regression | 0.534 |	0.536 |	0.510 | 0.523 | 0.550 |
| Naive Bayes | 0.514 | 0.516 | 0.446 | 0.479 | 0.531 |
| Decision Tree | 0.605 | 0.609 | 0.590 | 0.599 | 0.631 | 
| Random Forest | 0.637 | 0.634	| 0.647	| 0.640	| 0.702 |
| AdaBoost | 0.526 | 0.529 | 0.465 | 0.495 | 0.535 |
| GBDT | 0.629 | 0.630 | 0.624 | 0.628 | 0.663 |
| XGBoost | 0.607 | 0.606 | 0.615 | 0.610 | 0.614 |
| PJFNN-m | 0.660 | 0.654 | 0.679 | 0.667 | 0.715 |

### Hit Rate@N
|      | N=1 | N=5 | N=10| N=20 | 
| :---- | ----: | ----: | ----: | ----: | 
| Linear Regression | 0.008	| 0.100	| 0.158	| 0.288 |
| Logistic Regression | 0.008 | 0.092 | 0.142 | 0.300 |
| Naive Bayes | 0.015 | 0.085 | 0.208 | 0.369 |
| Decision Tree | 0.031 | 0.081 | 0.162	| 0.304 |
| Random Forest | 0.027	| 0.123	| 0.227	| 0.431 |
| AdaBoost | 0.019 | 0.123 | 0.212 | 0.327 |
| GBDT | 0.038 | 0.150 | 0.235 | 0.404 |
| XGBoost | 0.027 | 0.104 | 0.196 | 0.377 |
| PJFNN-m | 0.042 | 0.150 | 0.262 | 0.446 |
