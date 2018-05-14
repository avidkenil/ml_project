import pandas as pd
import numpy as np
import random
import pickle
import os

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def dump_data(data_sets, data_path='../data/', force=False):
    if not os.path.exists(data_path):
        print('Creating {}... '.format(data_path), end='', flush=True)
        os.makedirs(data_path)
        print('Done.')
    for data in data_sets:
        file_path = data_path+'{}.pkl'.format(data)
        if force or not os.path.isfile(file_path):
            print('Dumping {}... '.format(file_path), end='', flush=True)
            pickle.dump(data_sets[data], open(file_path, 'wb'))
            print('Done.')
        else:
            print('Did not dump {}: File already exists.'.format(file_path))

def split_data(data, target_cols, clean='_clean'):
    # Split data into train (0.8), validation (0.2), and test (0.2) sets
    X_train_val, X_test, y_train_val, y_test = \
    train_test_split(data.drop(target_cols, axis=1), data[target_cols], test_size=0.2, random_state=1337)

    X_train, X_val, y_train, y_val = \
    train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1337)
    
    # Order (separate) train+val set
    train_idx, val_idx = X_train.index.values, X_val.index.values
    X_train_val = X_train_val.loc[train_idx].append(X_train_val.loc[val_idx])
    y_train_val = y_train_val.loc[train_idx].append(y_train_val.loc[val_idx])
    
    data_sets = {
        'X_train': X_train,
        'X_val': X_val,
        'X_train_val': X_train_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_train_val': y_train_val,
        'y_test': y_test
    }
    
    data_sets = {data+clean: data_sets[data] for data in data_sets}

    return data_sets

def oversample_data(data_sets, oversampling_cols, oversampling_target_cols, clean='_clean'):
    for (X, y) in oversampling_cols:
        X, y = X+clean, y+clean
        X_os, y_os = X+'_os', y+'_os'
        total_rows = data_sets[y].shape[0]
        oversampling_indices = []
        for col in oversampling_target_cols:
            count = data_sets[y][data_sets[y][col] == 1].shape[0]
            while count/total_rows < 0.05:
                indices_1 = data_sets[y][data_sets[y][col] == 1].index.values
                rand_index = np.random.choice(indices_1)
                oversampling_indices.append(rand_index)
                count += 1
        
        data_sets[X_os] = data_sets[X].append(data_sets[X].loc[oversampling_indices])
        data_sets[y_os] = data_sets[y].append(data_sets[X].loc[oversampling_indices])
        data_sets[X_os], data_sets[y_os] = shuffle(data_sets[X_os], data_sets[y_os])
    
    return data_sets

def prepare_stopwords():
    NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere","no",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

    stopwords = STOP_WORDS.copy()
    for word in STOP_WORDS:
        if word in NEGATE:
            stopwords.remove(word)

    return stopwords

def spacy_tokenizer(parser, sentence, stopwords, punctuations):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return ' '.join(tokens)

def clean_data(data, col, parser, stopwords, punctuations):
    clean_text = []
    for text in data[col]:
        clean_text.append(spacy_tokenizer(parser, text, stopwords, punctuations))
    return clean_text

def main():
    # Load data
    print('Loading data... ', end='', flush=True)
    data_path = '../data/'
    data = pd.read_csv(data_path+'data.csv')
    print('Done.')

    print(data.head(10))

    # Print value counts for all target classes
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for col in target_cols:
        print(data[col].value_counts())

    # Set parameter indicating processing for uncleaned data
    clean = ''

    # Split data
    print('Splitting raw data into train-val-test sets... ', end='', flush=True)
    data_sets = split_data(data, target_cols, clean)
    print('Done.')

    # Oversample data
    print('Oversampling raw data... ', end='', flush=True)
    oversampling_target_cols = ['severe_toxic', 'threat', 'identity_hate']
    data_sets = oversample_data(data_sets, [('X_train', 'y_train'), ('X_train_val', 'y_train_val')], \
                                oversampling_target_cols, clean)
    print('Done.')

    # Dump data
    dump_data(data_sets, data_path)

    # All preprocessing that follows is for cleaned data
    clean = '_clean'

    # Load spacy parser and punctuations
    spacy_en = spacy.load('en')
    parser = English()
    punctuations = string.punctuation

    # Remove negations from stopword list
    print('Preparing stopwords... ', end='', flush=True)
    stopwords = prepare_stopwords()
    print('Done.')

    # Clean data
    print('Cleaning stopwords... ', end='', flush=True)
    data['comment_text'] = clean_data(data, 'comment_text', parser, stopwords, punctuations)
    print('Done.')

    # Dump cleaned data
    dump_data({'data_clean': data}, data_path)

    print(data.head())

    print(data.isnull().sum().sum())

    # Split cleaned data
    print('Splitting cleaned data into train-val-test sets... ', end='', flush=True)
    data_sets = split_data(data, target_cols, clean)
    print('Done.')

    # Oversample cleaned data
    print('Oversampling cleaned data... ', end='', flush=True)
    oversampling_target_cols = ['severe_toxic', 'threat', 'identity_hate']
    data_sets = oversample_data(data_sets, [('X_train', 'y_train'), ('X_train_val', 'y_train_val')], \
                                oversampling_target_cols, clean)
    print('Done.')

    # Dump data
    dump_data(data_sets, data_path)

if __name__ == "__main__":
    main()
