# Project for 'DS-GA 1003 Machine Learning'
 ## Toxic Comment Classification Challenge

### Requirements
 - python3
 - [pytorch](https://pytorch.org/)
 - [fastText](https://github.com/facebookresearch/fastText)
 
Run the following commands to install other packages:
<br>
`pip3 install -r requirements.txt`
<br>
`pip3 install -U spacy`

### Data

Download the data from the [Toxic Comment Classification Challenge webpage](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/).

### Usage
Navigate to the [src](https://github.com/ranamihir/ml_project/tree/master/src) folder.

Machine Learning models: Run the following commands (back to back):
 - `python3 preprocessing.py`
 - `python3 models.py`

fastText models: Run the [fasttext](https://github.com/ranamihir/ml_project/blob/master/src/fasttext.ipynb) notebook.

Deep Learning models: Run the [deeplearning](https://github.com/ranamihir/ml_project/blob/master/src/deeplearning.ipynb) notebook. and [deeplearning2](https://github.com/ranamihir/ml_project/blob/master/src/deeplearning2.ipynb) notebooks.

### Results
All vectorized n-grams, AUC-ROC summary dataframes, predictions and probabilities will be dumped in the `pickle_objects/` folder.

Models and ROC curve plots will be dumped in the folders `pickle_objects/models/` and `plots/` (or `pickle_objects/models_features/` and `plots_features/` if you choose to use extra features -- see [models.py](https://github.com/ranamihir/ml_project/blob/master/src/models.py)).
