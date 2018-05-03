import pandas as pd
import numpy as np
import random
import json
import pickle
from os import path, makedirs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

# Set random seed
random.seed(1337)

def pretty_print(data_dict):
    try:
        print(json.dumps(data_dict, indent=4))
    except TypeError:
        print(data_dict)
    except Exception as e:
        print(str(e))

def load_data(data_path='../data/', is_clean=0, is_os=0):
    clean = '_clean' if is_clean else ''
    os = '_os' if is_os else ''
    data_sets = {}
    data_cols = [
        'data', 'X_train', 'X_val', 'X_train_val', 'X_test', \
        'y_train', 'y_val', 'y_train_val', 'y_test'
    ]
    
    for i, col in enumerate(data_cols):
        data_cols[i] = col + clean + os
    
    for col in data_cols:
        data_sets[col] = pickle.load(open(data_path+'{}.pkl'.format(col),'rb'))
    
    return data_sets

is_clean, is_os = 1, 0
clean, os = '_clean', ''
pickle_path = '../pickle_objects/'
model_path = pickle_path + 'models/'

data_sets = load_data('../data/', is_clean=1, is_os=0)

target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print(data_sets['data'+clean+os].head(10))

print(data_sets['X_train'+clean+os].head())

# Print value counts for each target
for target in target_cols:
    print('{}:'.format(target))
    for data in data_sets:
        if 'y' in data:
            value_counts = data_sets[data][target].value_counts()
            print('{}: {:.3f}%\t'.format(data, 100*value_counts[1]/sum(value_counts)), end='')
    print('\n')

num_feats = 1000
n_grams = 2
ngram_range = list(map(lambda x: x+1,range(n_grams)))

def load_ngrams(data, num_feats, ngram_range, pickle_path='../pickle_objects/', is_clean=1, is_os=0):
    clean = '_clean' if is_clean else ''
    os = '_os' if is_os else ''
    ngrams = {}
    vec_params = {'analyzer': 'word', 'lowercase': True,'max_features': num_feats, 'ngram_range': ngram_range}
    
    for vec in ['countvec', 'tfidf']:
        # Load vectorizer if present
        file_name = '{}{}_{}_ngrams_{}{}.pkl'.format(pickle_path, vec, num_feats, n_grams, clean+os)
        if path.isfile(file_name):
            ngrams[vec] = pickle.load(open(file_name, 'rb'))
        else:
            # Fit, store, and load vectorizer
            ngrams_vec = CountVectorizer(**vec_params) if vec == 'countvec' else TfidfVectorizer(**vec_params)
            ngrams_vec.fit(data['comment_text'])
            ngrams[vec] = ngrams_vec
            if not path.exists(pickle_path):
                makedirs(pickle_path)
            pickle.dump(ngrams_vec, open(file_name, 'wb'))
    return ngrams

ngrams = load_ngrams(data_sets['X_train'+clean+os], num_feats, ngram_range, \
                     pickle_path, is_clean, is_os)

def transform_to_ngrams(data_set, data_cols, ngrams):
    for data in data_cols:
        for vec in ['countvec', 'tfidf']:
            data_sets[data+'_'+vec] = ngrams[vec].transform(data_sets[data]['comment_text'])
    return data_sets

data_sets = transform_to_ngrams(data_sets, ['X_train'+clean+os, 'X_val'+clean+os, \
                                            'X_train_val'+clean+os, 'X_test'+clean+os], ngrams)

def normalize_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def fit_model(base_model, X, y, param_grid, target_cols, scoring='roc_auc', cv=None):
    if cv:
        models, mean_val_scores, params = {}, [], []
        for target in target_cols:
            model_target = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring, refit=False)
            model_target.fit(X, y[target])
            mean_val_scores.append(model_target.cv_results_['mean_test_score'])
            if not params:
                params = model_target.cv_results_['params']
        mean_val_scores = np.mean(np.array(mean_val_scores), axis=0)
        best_param_idx = np.argmax(mean_val_scores)
        models['best_params_'], models['best_mean_score_'] = \
        params[best_param_idx], mean_val_scores[best_param_idx]
    else:
        models = {}
        for target in target_cols:
            base_model.set_params(**param_grid).fit(X, y[target])
            models[target] = base_model
    return models

def dump_models(model, X, model_name, target_cols, model_path='../pickle_objects/models/', cv=None):
    if not path.exists(model_path):
        makedirs(model_path)
    file_name = '{}{}_{}.pkl'.format(model_path, model_name, X)
    if not path.isfile(file_name):
        pickle.dump(model, open(file_name, 'wb'))

def fit_all_models(data_sets, data_cols, model_list, param_grids, \
                   target_cols, model_path='../pickle_path/models/', cv=None):
    best_models, best_params, best_scores = {}, {}, {}
    X, y = data_cols
    for model in model_list:
        best_models[model] = fit_model(model_list[model], data_sets[X], \
                                          data_sets[y], param_grids[model], \
                                          target_cols, 'roc_auc', cv)
        if cv:
            print('Model = {}'.format(model))
            pretty_print(best_models[model])
        else:
            dump_models(best_models[model], X, model, target_cols, model_path, cv)
    return best_models

model_list = {
    'bnb': BernoulliNB(),
#     'gnb': GaussianNB(),
    'lrl1': LogisticRegression(penalty='l1'),
    'lrl2': LogisticRegression(penalty='l2'),
    'rf': RandomForestClassifier(),
    'xgb': XGBClassifier(),
    'svm': SVC(kernel='linear')
}
    
param_grids = {
    'bnb': {},
#     'gnb': {},
    'lrl1': {'C': np.concatenate((np.reciprocal(np.arange(1, 13, 3)), \
                                  np.logspace(1, 6, num=6, endpoint=True, base=10)))},
    'lrl2': {'C': np.concatenate((np.reciprocal(np.arange(1, 13, 3)), \
                                  np.logspace(1, 6, num=6, endpoint=True, base=10)))},
    'rf': {
        'n_estimators': np.arange(50, 250, 50),
        'max_features': ['auto', 'log2'],
        'max_depth': np.arange(3, 13, 2)
    },
    'xgb': {'n_estimators': [1]},
    'svm': {'C': np.concatenate((np.arange(1, 13, 3), np.logspace(1, 6, num=6, endpoint=True, base=10)))},
}

# Set predefined split for CV
# 0 corresponds to val, -1 to train
val_fold = [-1]*len(data_sets['X_train'+clean+os]) + [0]*len(data_sets['X_val'+clean+os])
predefined_split = PredefinedSplit(test_fold=val_fold)

# Find best hyperparameter settings for each data set for each model
vec = 'countvec'
data_cols = ('X_train_val'+clean+os+'_'+vec, 'y_train_val'+clean+os)
best_models = fit_all_models(data_sets, data_cols, model_list, param_grids, \
                             target_cols, model_path, cv=predefined_split)

pretty_print(best_models)

def refit_best_model(data_sets, model_list, data_cols, best_models, \
                     target_cols, pickle_path='../pickle_path/'):
    # Refit all models with all data sets with best hyperparameters
    best_params = {}
    X, y = data_cols
    for model in model_list:
        best_params[model] = best_models[model]['best_params_']
    best_refitted_models = fit_all_models(data_sets, data_cols, model_list, \
                                          best_params, target_cols, model_path, cv=None)
    return best_refitted_models

best_refitted_models = refit_best_model(data_sets, model_list, data_cols, \
                                        best_models, target_cols, model_path)

def predict_labels_and_probas(fitted_models, model_list, X, target_cols):
    probabilities = {}
    for model in model_list:
        probabilities[model] = {}
        for target in target_cols:
            probabilities[model][target] = fitted_models[model][target].predict_proba(X)[:,1]
#         probabilities[model] = np.squeeze(np.asarray(probabilities[model].todense()))
    return probabilities

probabilities = predict_labels_and_probas(best_refitted_models, model_list, \
                                          data_sets['X_test'+clean+os+'_'+vec], target_cols)

def plot_model_roc_curves(y_test, probabilities, model_list, target_cols, vec='countvec', \
                          model_or_target='model', plots_path='../plots/'):
    aucs = {}
    # Plot by model
    if model_or_target == 'model':
        for model in model_list:
            aucs[model] = {}
            plt.figure(figsize=(10,8))
            for target in target_cols:
                fpr, tpr, threshold = roc_curve(y_test[target], probabilities[model][target])
                auc_value = auc(fpr, tpr)
                aucs[model][target] = auc_value
                plt.plot(fpr, tpr, label='{}: {:0.5f}'.format('auc_'+target, auc_value))
            plt.xlabel('fpr')
            plt.ylabel('tpr')
            plt.title('ROC Curve for {} with {}'.format(model, vec))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            plt.savefig(plots_path+'roc_'+model+'_'+vec+'.jpg')
            plt.close('all')
    # Plot by target column
    elif model_or_target == 'target':
        for target in target_cols:
            aucs[target] = {}
            plt.figure(figsize=(10,8))
            for model in model_list:
                fpr, tpr, threshold = roc_curve(y_test[target], probabilities[model][target])
                auc_value = auc(fpr, tpr)
                aucs[target][model] = auc_value
                plt.plot(fpr, tpr, label='{}: {:0.5f}'.format('auc_'+model, auc_value))
            plt.xlabel('fpr')
            plt.ylabel('tpr')
            plt.title('ROC Curve for {} with {}'.format(target, vec))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            plt.savefig(plots_path+'roc_'+target+'_'+vec+'.jpg')
            plt.close('all')
    else:
        raise TypeError("Parameter 'model_or_target' must be one of 'model' or 'target'.")
    return aucs

def get_mean_auc(aucs, model_list=None, target_cols=None, model_or_target='model'):
    mean_aucs = {}
    # Compute mean auc by model
    columns = model_list if model_or_target == 'model' else target_cols
    for col in columns:
        mean_aucs[col] = np.mean(list(aucs[col].values()))
    return mean_aucs

model_or_target = 'target'
aucs = plot_model_roc_curves(data_sets['y_test'+clean+os], probabilities, \
                      model_list, target_cols, vec, model_or_target, '../plots/')

pretty_print(aucs)

mean_aucs = get_mean_auc(aucs, model_list, target_cols, model_or_target)

pretty_print(mean_aucs)

def get_aucs_df(aucs, model_list, target_cols, model_or_target='model'):
    aucs_df = pd.DataFrame.from_dict(aucs)
    aucs_df['mean'] = np.mean(aucs_df, axis=1)
    aucs_df.loc['mean'] = np.mean(aucs_df, axis=0)
    return aucs_df

aucs_df = get_aucs_df(aucs, model_list, target_cols, model_or_target)

pickle.dump(aucs_df, open('{}aucs_df.pkl'.format(pickle_path), 'wb'))

print(aucs_df)