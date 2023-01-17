import string
import regex as re
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tokenizers import BertWordPieceTokenizer


def preprocess(text):
    # Remove integers
    text = re.sub(r'\d+', '', text)

    # remove newlines as \r and \n
    text = re.sub(r'\r', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    return text


def encode_text_and_labels(df, max_num_words, pre_or_post='post', subword=False):
    # create a tokenizer
    if subword:
        t = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True
        )

        t.train_from_iterator(df['text'])
        vocab_size = t.get_vocab_size()
        # integer encode the documents
        encoded_list = t.encode_batch(df['text'])
        encoded_docs = [x.ids for x in encoded_list]
        # pad documents to be as long as the longest sequence in the dataset
        max_length = max([len(x) for x in encoded_docs])
    else:
        t = Tokenizer(num_words=max_num_words, oov_token='<unk>')
        t.fit_on_texts(df['text'])
        vocab_size = len(t.word_index) + 1
        # integer encode the documents
        encoded_docs = t.texts_to_sequences(df['text'])
        # pad documents to be as long as the longest sequence in the dataset
        max_length = df['text'].apply(lambda x: len(x.split(' '))).max()

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding=pre_or_post)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['artist'])
    # binary encode
    onehot_encoded = to_categorical(integer_encoded)
    return padded_docs, onehot_encoded, vocab_size, max_length, t


def load_and_preprocess_data(path, max_num_words=None, pre_or_post='post', subword=False):
    """
    Load the data and preprocess it
    :param path: path to the data
    :return: preprocessed data in the form of a pandas dataframe. The first item returned is the data,
    the second is the labels, the third is the vocabulary size, and the fourth is the maximum length of a sequence
    """
    df = pd.read_csv(path)

    df = df.groupby('artist').filter(lambda x: len(x) > 100)

    df['text'] = df['text'].apply(preprocess)

    # Identify the rows that contain duplicated text in the 'song' column
    no_covers = ~df['song'].duplicated()

    # Filter the DataFrame to include only the rows with unique text
    df = df[no_covers]

    # prepare text data for a recurrent network
    return encode_text_and_labels(df, max_num_words, pre_or_post, subword)
