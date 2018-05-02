import torch
import torch.nn as nn
from torch.autograd import Variable

import collections as col
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_word_srs(text_data):
    word_counter = col.Counter()
    for line in text_data:
        for word in line.split():
            word_counter[word] += 1
    return pd.Series(word_counter).sort_values(ascending=False)


SPECIAL_SYMBOLS_ID = \
    PAD_ID, OOV_ID, EOS_ID, SOS_ID = \
    0, 1, 2, 3
SPECIAL_SYMBOLS_STR = \
    PAD_STR, OOV_STR, EOS_STR, SOS_STR = \
    "<PAD>", "<OOV>", "<EOS>", "<SOS>"
NUM_SPECIAL = 4


class Dictionary:
    def __init__(self, words):
        assert isinstance(words, list)
        self.id2word = [None] + words
        self.word2id = {word: 1 + i for i, word in enumerate(words)}

    def sentence2ids(self, sentence, eos=False, sos=False):
        if isinstance(sentence, str):
            tokens = sentence.split()
        else:
            tokens = sentence

        ids = [
            NUM_SPECIAL + self.word2id[word] - 1
            if word in self.word2id else OOV_ID
            for word in tokens
        ]

        if sos:
            ids = [SOS_ID] + ids
        if eos:
            ids = ids + [EOS_ID]
        return ids

    def sentences2ids(self, sentences, eos=False, sos=False):
        ids = [
            self.sentence2ids(sentence, eos=eos, sos=sos)
            for sentence in sentences
        ]
        lengths = [len(id_ls) for id_ls in ids]

        # Find max
        max_length = max(lengths)

        # Pad
        ids = [
            id_ls + [PAD_ID] * (max_length - len(id_ls))
            for id_ls in ids
        ]
        return ids, lengths

    def ids2sentence(self, ids):
        return ' '.join([
            '<OOV>' if i == OOV_ID else self.id2word[i - NUM_SPECIAL + 1]
            for i in ids
            if i != EOS_ID and i != PAD_ID and i != SOS_ID
        ])

    def ids2sentences(self, ids):
        return [self.ids2sentence(i) for i in ids]

    def rawids2sentence(self, ids):
        return ' '.join([
            SPECIAL_SYMBOLS_STR[i]
            if i < NUM_SPECIAL
            else self.id2word[i - NUM_SPECIAL + 1]
            for i in ids
        ])

    def rawids2sentences(self, ids):
        return [self.rawids2sentence(i) for i in ids]

    def size(self):
        return len(self.id2word) - 1


def read_embeddings(word_vector_path, vocabulary, max_read=None):
    # Unlike undreamt, I'm assuming we have a vocabulary

    vocabulary_set = set(vocabulary)

    embeddings_vectors_ls = []
    embeddings_words_ls = []

    with open(word_vector_path, "r") as f:
        for i, line in enumerate(f):
            word, vec = line.split(' ', 1)
            if word in vocabulary_set:
                embeddings_vectors_ls.append(np.fromstring(vec, sep=' '))
                embeddings_words_ls.append(word)
            if max_read is not None and i > max_read \
                    or len(embeddings_words_ls) == len(vocabulary):
                break

    embeddings_matrix = np.array(
        [np.zeros(embeddings_vectors_ls[-1].shape)] + embeddings_vectors_ls
    )
    embeddings = nn.Embedding(
        embeddings_matrix.shape[0],
        embeddings_matrix.shape[1],
        padding_idx=0,
    )
    embeddings.weight.data.copy_(torch.from_numpy(embeddings_matrix))
    return embeddings, Dictionary(embeddings_words_ls)


class RNNEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_classes,
                 bidirectional=False, layers=1, dropout=0):
        super(RNNEncoder, self).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even '
                             'for bidirectional encoders')
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size // self.directions
        self.special_embeddings = nn.Embedding(
            NUM_SPECIAL+1, embedding_size, padding_idx=0)
        self.rnn = nn.GRU(
            embedding_size,
            self.hidden_size,
            bidirectional=bidirectional,
            num_layers=layers,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size, output_classes)

    def forward(self, enc_input, lengths, word_embeddings, hidden):
        true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
        sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
        enc_input = torch.stack([enc_input[i, :] for i in true2sorted], dim=1)
        lengths = [lengths[i] for i in true2sorted]

        assert "LongTensor" in enc_input.data.type()
        embeddings = self.embed_ids(
            ids=enc_input,
            word_embeddings=word_embeddings,
        )

        embeddings = \
            nn.utils.rnn.pack_padded_sequence(embeddings, lengths)
        output, hidden = self.rnn(embeddings, hidden)
        if self.bidirectional:
            hidden = torch.stack([
                torch.cat((hidden[2*i], hidden[2*i+1]), dim=1)
                for i in range(self.layers)
            ])
        hidden = torch.stack([hidden[:, i, :] for i in sorted2true], dim=1)
        raw_output = self.linear(hidden.squeeze(0))
        return raw_output

    def initial_hidden(self, batch_size):
        return Variable(torch.zeros(
            self.layers*self.directions, batch_size, self.hidden_size
        ), requires_grad=False)

    def embed_ids(self, ids, word_embeddings, include_special=True):
        embeddings = word_embeddings(word_ids(ids))
        if include_special:
            embeddings += self.special_embeddings(special_ids(ids))
        return embeddings


def special_ids(ids_tensor):
    return ids_tensor * (ids_tensor < NUM_SPECIAL).long()


def word_ids(ids_tensor):
    return (ids_tensor - NUM_SPECIAL + 1) * (ids_tensor >= NUM_SPECIAL).long()


class CorpusReader:
    def __init__(self, x_df, y_df):
        """
        Reads in fixed order.
        No cache for now. Should be easy to upgrade later
        """
        self.x_df = x_df
        self.y_df = y_df

    def sample_batch(self, batch_size):
        row_ids = np.random.randint(
            low=0, high=len(self.x_df), size=batch_size)
        text_ls = []
        y_ls = []
        for i in row_ids:
            text_ls.append(self.x_df.iloc[i]["comment_text"])
            y_ls.append(self.y_df.iloc[i].values)
        return (
            text_ls,
            np.array(y_ls)
        )

    def iterate(self, batch_size, shuffle=True):
        row_ids = np.arange(len(self.x_df))
        if shuffle:
            np.random.shuffle(row_ids)
        batch_start = 0
        while batch_start < len(self.x_df):
            batch_row_ids = row_ids[
                            batch_start:batch_start + batch_size]
            actual_batch_size = len(batch_row_ids)
            text_ls = self.x_df.iloc[batch_row_ids]["comment_text"].tolist()
            y_ls = np.array(self.y_df.iloc[batch_row_ids])
            yield (
                int(batch_start / batch_size),
                actual_batch_size,
                text_ls,
                y_ls
            )
            batch_start += actual_batch_size


def get_device_func():
    if torch.has_cudnn:
        return lambda _: _.cuda() if _ is not None else None
    else:
        return lambda _: _.cpu() if _ is not None else None


def validate(corpus, model, word_embeddings,
             max_batch_size, dictionary, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    for i, batch_size, x_batch, y_batch in \
            corpus.iterate(max_batch_size):
        ids_batch, lengths = dictionary.sentences2ids(x_batch, eos=True)
        ids_batch = device(Variable(
            torch.LongTensor(ids_batch), volatile=True,
        ))
        y_batch = device(Variable(torch.LongTensor(y_batch), volatile=True))
        hidden = device(model.initial_hidden(batch_size))
        pred_batch = model(
            enc_input=ids_batch,
            lengths=lengths,
            word_embeddings=word_embeddings,
            hidden=hidden
        )
        loss = criterion(
            pred_batch.view(-1),
            y_batch.contiguous().view(-1).float(),
        )
        total_loss += loss.data[0] * batch_size
    return total_loss / len(corpus.x_df)


def inference(corpus, model, word_embeddings,
              max_batch_size, dictionary, device, to_prob=True):
    model.eval()
    preds = []
    for i, batch_size, x_batch, y_batch in \
            corpus.iterate(max_batch_size, shuffle=False):
        ids_batch, lengths = dictionary.sentences2ids(x_batch, eos=True)
        ids_batch = device(Variable(
            torch.LongTensor(ids_batch), volatile=True,
        ))
        hidden = device(model.initial_hidden(batch_size))
        pred_batch = model(
            enc_input=ids_batch,
            lengths=lengths,
            word_embeddings=word_embeddings,
            hidden=hidden
        )
        preds.append(pred_batch.data.cpu().numpy())
    preds = np.vstack(preds)
    if to_prob:
        return torch.sigmoid(torch.Tensor(preds)).numpy()
    else:
        return preds


def train_model(
        param_dict, device, full_word_srs,
        train_corpus, val_corpus,
        batch_size, log_step):
    word_list = full_word_srs[:param_dict["top_k_words"]].index.tolist()
    word_embeddings, dictionary = read_embeddings(
        param_dict["glove_path"],
        vocabulary=word_list,
    )
    word_embeddings = device(word_embeddings)
    word_embeddings.weight.requires_grad = False
    model = device(RNNEncoder(
        embedding_size=word_embeddings.embedding_dim,
        hidden_size=param_dict["hidden_size"],
        output_classes=6,
        bidirectional=True,
        layers=1,
        dropout=param_dict["dropout_prob"],
    ))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param_dict["learning_rate"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss_log = []
    for epoch in range(param_dict["n_epochs"]):
        for i, batch_size, x_batch, y_batch in \
                train_corpus.iterate(batch_size):
            model.train()
            optimizer.zero_grad()
            ids_batch, lengths = dictionary.sentences2ids(x_batch, eos=True)
            ids_batch = device(Variable(torch.LongTensor(ids_batch)))
            y_batch = device(Variable(torch.LongTensor(y_batch)))
            hidden = device(model.initial_hidden(batch_size))
            pred_batch = model(
                enc_input=ids_batch,
                lengths=lengths,
                word_embeddings=word_embeddings,
                hidden=hidden
            )
            loss = criterion(
                pred_batch.view(-1),
                y_batch.contiguous().view(-1).float(),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
            if log_step > 0 and i % log_step == 0:
                val_loss = validate(
                    val_corpus, model, word_embeddings,
                    batch_size, dictionary, device,
                )
                print(f"EPOCH {epoch}@{i}: {val_loss}, {dt.datetime.now()}")
                val_loss_log.append((epoch, i, val_loss))
        val_loss = validate(
            val_corpus, model, word_embeddings,
            batch_size, dictionary, device,
        )
        print(f"EPOCH {epoch}: {val_loss}, {dt.datetime.now()}")
        val_loss_log.append((epoch, i, val_loss))

    return model, word_embeddings, dictionary, val_loss_log


def get_auc(true_y, prob):
    auc_ls = []
    assert true_y.shape == prob.shape
    for i in range(true_y.shape[1]):
        fpr, tpr, threshold = roc_curve(true_y.iloc[:, i], prob[:, i])
        auc_value = auc(fpr, tpr)
        auc_ls.append(auc_value)
    return auc_ls


def plot_roc(true_y, prob, target_cols):
    auc_ls = []
    plt.figure(figsize=(10, 8))
    for i, column in enumerate(target_cols):
        fpr, tpr, threshold = roc_curve(true_y.iloc[:, i], prob[:, i])
        auc_value = auc(fpr, tpr)
        auc_ls.append(auc_value)
        plt.plot(fpr, tpr, label=f'auc_{column}: {auc_value:0.5f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.show()
