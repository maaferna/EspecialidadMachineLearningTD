# src/utils/preprocessing_text.py
from typing import Tuple, Optional
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets.imdb import get_word_index

def load_imdb_sequences(
    num_words: int = 20000,
    maxlen: int = 200,
    return_word_index: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Optional[dict]]:
    """
    Carga IMDb de tf.keras.datasets.imdb, limita el vocabulario a num_words y
    paddea a longitud fija maxlen. Las etiquetas quedan en {0,1}.
    Retorna: (x_train, y_train), (x_test, y_test), word_index (opcional)
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    x_train = pad_sequences(x_train, maxlen=maxlen, padding="pre", truncating="pre")
    x_test  = pad_sequences(x_test,  maxlen=maxlen, padding="pre", truncating="pre")
    if return_word_index:
        
        return (x_train, y_train), (x_test, y_test), get_word_index()
    return (x_train, y_train), (x_test, y_test), None
