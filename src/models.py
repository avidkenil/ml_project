import pandas as pd
import numpy as np
import random
import json
import pickle
import joblib
from os import path, makedirs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


# Pretty print JSON objects
def pretty_print(data_dict):
    try:
        print(json.dumps(data_dict, indent=4))
    except TypeError:
        print(data_dict)
    except Exception as e:
        print(str(e))

def create_paths(path_list):
    print('Creating all folder paths... ', end='', flush=True)
    for folder_path in path_list:
        if not path.exists(folder_path):
            makedirs(folder_path)
    print('Done.')

# Load all data sets
def load_data(data_path='../data/', is_clean=0, is_os=0):
    print('Loading data... ', end='', flush=True)
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

    print('Done.')
    return data_sets

# Load all ngrams data if present, otherwise fit on data and dump them
def load_ngrams(data_sets, data_col, num_feats, ngrams, vectorizers, pickle_path='../pickle_objects/', is_clean=1, is_os=0):
    clean = '_clean' if is_clean else ''
    os = '_os' if is_os else ''
    data_col += clean+os
    ngrams_data = {}
    ngram_range = list(map(lambda x: x+1,range(ngrams)))
    vec_params = {'analyzer': 'word', 'lowercase': True,'max_features': num_feats, 'ngram_range': ngram_range}
    
    for vec in vectorizers:
        # Load vectorizer if present
        file_name = '{}{}_ngrams_{}_{}_{}.pkl'.format(pickle_path, vec, data_col, num_feats, ngrams)
        if path.isfile(file_name):
            print('Loading {} ngrams... '.format(vec), end='', flush=True)
            ngrams_data[vec] = pickle.load(open(file_name, 'rb'))
            print('Done.')
        else:
            # Fit, store, and load vectorizer
            print('{} ngrams not found. Fitting and dumping them... '.format(vec), end='', flush=True)
            ngrams_vec = CountVectorizer(**vec_params) if vec == 'countvec' else TfidfVectorizer(**vec_params)
            ngrams_vec.fit(data_sets[data_col]['comment_text'])
            ngrams_data[vec] = ngrams_vec
            pickle.dump(ngrams_vec, open(file_name, 'wb'))
            print('Done.')
    return ngrams_data

# Transform data on fitted ngrams data
def transform_to_ngrams(data_sets, data_cols, ngrams_data, vectorizers):
    print('Transforming data to ngrams... ', end='', flush=True)
    for data in data_cols:
        for vec in vectorizers:
            data_sets[data+'_'+vec] = ngrams_data[vec].transform(data_sets[data]['comment_text'])
    print('Done.')
    return data_sets

# Dump all models of a type fitted on all target columns
def dump_models(model, X, model_name, target_cols, model_path='../pickle_objects/models/'):
    for target in target_cols:
        file_name = '{}{}_{}_{}.sav'.format(model_path, model_name, X, target)
        if not path.isfile(file_name):
            print('\t\tDumping {} fitted on {}... '.format(model_name, target), end='', flush=True)
            joblib.dump(model[target], open(file_name, 'wb'))
            print('Done.')
        else:
            print('\t\tDid not dump {} fitted on {}: File already exists in "{}".' \
                  .format(model_name, target, file_name))

# Fit a model on all target columns after performing grid search or with best parameters
def fit_model(base_model, X, y, param_grid, target_cols, scoring='roc_auc', cv=None):
    if cv:
        models, mean_val_scores, params = {}, [], []
        for target in target_cols:
            print('\t\tRunning for {}... '.format(target), end='', flush=True)
            model_target = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring, n_jobs=4, refit=False)
            model_target.fit(X, y[target])
            print('Done.')
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
            model = clone(base_model)
            print('\t\tRunning for {}... '.format(target), end='', flush=True)
            model.set_params(**param_grid).fit(X, y[target])
            models[target] = model
            print('Done.')
    return models

# Fit all models on all target columns or dump the refitted ones with best parameters
def fit_all_models(data_sets, data_cols, model_list, param_grids, target_cols, \
                   model_path='../pickle_path/models/', cv=None):
    best_models, best_params, best_scores = {}, {}, {}
    X, y = data_cols
    for model in model_list:
        print('\tRunning {}...'.format(model))
        best_models[model] = fit_model(model_list[model], data_sets[X], data_sets[y], \
                                       param_grids[model], target_cols, 'roc_auc', cv)
        print('\tDone.')
        if cv:
            pretty_print(best_models[model])
        else:
            print('\tDumping {}... '.format(model))
            dump_models(best_models[model], X, model, target_cols, model_path)
            print('\tDone.')
    return best_models

# Refit all models on all target columns with best parameters
def refit_best_models(data_sets, model_list, data_cols, best_models, \
                      target_cols, model_path='../pickle_path/models/'):
    # Refit all models with all data sets with best hyperparameters
    print('Refitting with best parameters...')
    best_params = {}
    for model in model_list:
        best_params[model] = best_models[model]['best_params_']
    best_refitted_models = fit_all_models(data_sets, data_cols, model_list, best_params, \
                                          target_cols, model_path, cv=None)
    print('Done.')
    return best_refitted_models

# Load all models of a type fitted on all target columns
def load_models(model_name, X, model_path='../pickle_objects/models/'):
    model = {}
    for target in target_cols:
        file_name = '{}{}_{}_{}.sav'.format(model_path, model_name, X, target)
        if path.isfile(file_name):
            print('\tLoading {} fitted on {}... '.format(model_name, target), end='', flush=True)
            model[target] = joblib.load(open(file_name, 'rb'))
            print('Done.')
        else:
            print('\tDid not load {} fitted on {}: File not found in "{}".' \
                  .format(model_name, target, file_name))
    return model

# Load all models fitted on all target columns
def load_all_models(model_list, data_cols, target_cols, model_path='../pickle_path/models/'):
    print('Loading models fitted with best parameters...')
    best_models = {}
    X, y = data_cols
    for model_name in model_list:
        best_models[model_name] = load_models(model_name, X, model_path)
    print('Done.')
    return best_models

# Predict labels and probabilities for all models for all target columns
def predict_labels_and_probas(fitted_models, model_list, X, target_cols):
    probabilities, predictions = {}, {}
    for model in model_list:
        print('\tPredicting labels and probabilities for {}...'.format(model))
        probabilities[model], predictions[model] = {}, {}
        for target in target_cols:
            print('\t\tPredicting for {}... '.format(target), end='', flush=True)
            probabilities[model][target] = fitted_models[model][target].predict_proba(X)[:,1]
            predictions[model][target] = fitted_models[model][target].predict(X)
            print('Done.')
        print('\tDone.')
    return probabilities, predictions

# Plot ROC curves for all models / target columns
def plot_model_roc_curves(y_test, probabilities, model_list, target_cols, vec='countvec', \
                          plot_type='model', plots_path='../plots/'):
    aucs = {}
    # Plot by model
    if plot_type == 'model':
        for model in model_list:
            print('\tPlotting ROC curve for {}... '.format(model), end='', flush=True)
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
            plt.tight_layout()
            plt.savefig(plots_path+'roc_'+model+'_'+vec+'.jpg')
            plt.close('all')
            print('Done.')
    # Plot by target column
    elif plot_type == 'target':
        for target in target_cols:
            print('\tPlotting ROC curve for {}... '.format(target), end='', flush=True)
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
            plt.tight_layout()
            plt.savefig(plots_path+'roc_'+target+'_'+vec+'.jpg')
            plt.close('all')
            print('Done.')
    else:
        raise TypeError("Parameter 'plot_type' must be one of 'model' or 'target'.")
    return aucs

# Generate mean column-wise AUC for all models
def get_mean_auc(aucs, model_list=None, target_cols=None, plot_type='model'):
    print('\tComputing mean AUCs... ', end='', flush=True)
    mean_aucs = {}
    # Compute mean auc by model
    columns = model_list if plot_type == 'model' else target_cols
    for col in columns:
        mean_aucs[col] = np.mean(list(aucs[col].values()))
    print('Done.')
    return mean_aucs

# Generate a summary AUCs dataframe for all models vs. all target columns
def get_aucs_df(aucs, model_list, target_cols, plot_type='model'):
    print('\tGenerating AUCs DataFrame... ', end='', flush=True)
    aucs_df = pd.DataFrame.from_dict(aucs)
    aucs_df['mean'] = np.mean(aucs_df, axis=1)
    aucs_df.loc['mean'] = np.mean(aucs_df, axis=0)
    print('Done.')
    return aucs_df

# Plot all ROC curves, dump all mean column-wise AUCs, generate summary AUCs dataframe, and return final predictions
def plot_and_dump_results(data_sets, best_refitted_models, model_list, vec, target_cols, plot_type='model', \
                          is_clean=1, is_os=0, plots_path='../plots/', pickle_path='../pickle_objects/'):
    clean = '_clean' if is_clean else ''
    os = '_os' if is_os else ''

    probabilities, predictions = predict_labels_and_probas(best_refitted_models[vec], model_list, \
                                                           data_sets['X_test'+clean+os+'_'+vec], target_cols)
    
    aucs = plot_model_roc_curves(data_sets['y_test'+clean+os], probabilities, model_list, \
                                 target_cols, vec, plot_type, plots_path)

    pretty_print(aucs)

    mean_aucs = get_mean_auc(aucs, model_list, target_cols, plot_type)

    pretty_print(mean_aucs)

    aucs_df = get_aucs_df(aucs, model_list, target_cols, plot_type)
    
    print('\tAUCs DataFrame for {}:'.format(vec))
    print(aucs_df)

    print('\tDumping AUCs DataFrame... ', end='', flush=True)
    pickle.dump(aucs_df, open('{}aucs_{}.pkl'.format(pickle_path, vec), 'wb'))
    print('Done.\n')

    if plot_type == 'model':
        return predictions

def main():
    # Set random seed
    random.seed(1337)
    
    # Specify whether to use cleaned data or not
    is_clean, is_os = 1, 0
    clean, os = '_clean', ''

    # Set all folder paths
    data_path = '../data/'
    pickle_path = '../pickle_objects2/'
    model_path = pickle_path + 'models/'
    plots_path = '../plots2/'

    # Specify initial variables
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    vectorizers = ['countvec', 'tfidf']
    plot_types = ['model', 'target']

    create_paths([data_path, pickle_path, model_path, plots_path])

    # Load all data sets
    data_sets = load_data(data_path, is_clean=is_clean, is_os=is_os)

    print(data_sets['data'+clean+os].head(10))

    print(data_sets['X_train'+clean+os].head())

    # Set ngram variables
    num_feats = 1000
    ngrams = 2

    # Load ngrams fitted on X_train (+clean+os)
    ngrams_data = load_ngrams(data_sets, 'X_train', num_feats, ngrams, vectorizers, pickle_path, is_clean, is_os)

    # Transform X_train_val (+clean+os) to ngrams
    data_sets = transform_to_ngrams(data_sets, ['X_train_val'+clean+os], ngrams_data, vectorizers)

    # List all models to be run
    model_list = {
        'bnb': BernoulliNB(),
        'lrl1': LogisticRegression(penalty='l1'),
        'lrl2': LogisticRegression(penalty='l2'),
        'rf': RandomForestClassifier(),
        'xgb': XGBClassifier(),
    #     'svm': SVC(kernel='linear')
    }
    
    # Specify corresponding parameters for GridSearchCV
    param_grids = {
        'bnb': {},
        'lrl1': {'C': np.concatenate((np.reciprocal(np.arange(1., 13., 3.)), \
                                      np.logspace(1., 6., num=6, endpoint=True, base=10)))},
        'lrl2': {'C': np.concatenate((np.reciprocal(np.arange(1., 13., 3.)), \
                                      np.logspace(1., 6., num=6, endpoint=True, base=10)))},
        'rf': {
            'n_estimators': np.arange(50, 250, 50),
            'max_features': ['auto', 'log2'],
            'max_depth': np.arange(3, 13, 2)
        },
        'xgb': {
            'n_estimators': np.arange(50, 250, 50),
            'max_depth': np.arange(3, 13, 2),
            'learning_rate': [1e-1, 1e-3, 1e-5],
            'reg_lambda': [1, 10, 1e-1]
        },
    #     'svm': {'C': np.concatenate((np.arange(1, 13, 3), np.logspace(1, 6, num=6, endpoint=True, base=10)))},
    }

    # Set predefined split for CV
    # 0 corresponds to val, -1 to train
    val_fold = [-1]*len(data_sets['X_train'+clean+os]) + [0]*len(data_sets['X_val'+clean+os])
    predefined_split = PredefinedSplit(test_fold=val_fold)

    # Find best hyperparameter settings for each data set for each model
    best_models = {}
    for vec in vectorizers:
        print('Running for {}...'.format(vec))
        data_cols = ('X_train_val'+clean+os+'_'+vec, 'y_train_val'+clean+os)
        best_models[vec] = fit_all_models(data_sets, data_cols, model_list, param_grids, \
                                          target_cols, model_path, cv=predefined_split)
        print('\n')

    # Load ngrams fitted on X_train_val (+clean+os)
    ngrams_data = load_ngrams(data_sets, 'X_train_val', num_feats, ngrams, vectorizers, pickle_path, is_clean, is_os)

    # Transform X_train_val and X_test (+clean+os) to ngrams
    data_sets = transform_to_ngrams(data_sets, ['X_train_val'+clean+os, 'X_test'+clean+os], \
                                    ngrams_data, vectorizers)

    # Refit all models with best parameters on ngrams fitted on X_train_val (+clean+os)
    best_refitted_models = {}
    for vec in vectorizers:
        print('\nRunning for {}...'.format(vec))
        data_cols = ('X_train_val'+clean+os+'_'+vec, 'y_train_val'+clean+os)
        best_refitted_models[vec] = refit_best_models(data_sets, model_list, data_cols, \
                                                      best_models[vec], target_cols, model_path)
        print('\n')

    predictions = {} # Store predictions for all models

    # Plot all ROC curves, dump all mean column-wise AUCs, generate summary AUCs dataframe, and get final predictions
    for plot_type in plot_types:
        for vec in vectorizers:
            print('Generating results for {}...'.format(vec))
            if plot_type == 'model':
                predictions[vec] = plot_and_dump_results(data_sets, best_refitted_models, model_list, vec, target_cols, \
                                                         plot_type, is_clean, is_os, plots_path, pickle_path)
            else:
                plot_and_dump_results(data_sets, best_refitted_models, model_list, vec, target_cols, \
                                      plot_type, is_clean, is_os, plots_path, pickle_path)
            print('Done.')

    # Dump all final predictions
    pickle.dump(predictions, open(pickle_path+'predictions.pkl', 'wb'))

if __name__ == "__main__":
    main()