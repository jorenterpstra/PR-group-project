import string
import regex as re
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


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


def encode_text_and_labels(df):
    t = Tokenizer()
    t.fit_on_texts(df['text'])
    # keep only words that appear more than min_df times
    # keep only words that appear more than min_df times
    word_docs = {word: freq for word, freq in t.word_docs.items() if freq > 1}

    # create new tokenizer
    t_filtered = Tokenizer()

    # fit t_filtered
    t_filtered.word_index = {word: i + 1 for i, (word, freq) in enumerate(word_docs.items())}
    t_filtered.word_counts = word_docs

    vocab_size = len(t_filtered.word_index) + 1
    # integer encode the documents
    encoded_docs = t_filtered.texts_to_sequences(df['text'])
    # pad documents to be as long as the longest sequence in the dataset
    max_length = df['text'].apply(lambda x: len(x.split(' '))).max()
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['artist'])
    # binary encode
    onehot_encoded = to_categorical(integer_encoded)
    return padded_docs, onehot_encoded, vocab_size, max_length

def load_and_preprocess_data(path, in_colab=False):
    """
    Load the data and preprocess it, expect runtime of 20 seconds.
    :param path: path to the data
    :return: preprocessed data in the form of a pandas dataframe. The first item returned is the data,
    the second is the labels, the third is the vocabulary size, and the fourth is the maximum length of a sequence
    """
    df = pd.read_csv(path)
      
    # remove artist with fewer than 30 songs
    df = df.groupby('artist').filter(lambda x: len(x) > 100)

    df['text'] = df['text'].apply(preprocess)

    # Identify the rows that contain duplicated text in the 'song' column
    no_covers = ~df['song'].duplicated()

    # Filter the DataFrame to include only the rows with unique text
    df = df[no_covers]

    # prepare text data for a recurrent network
    return encode_text_and_labels(df)

