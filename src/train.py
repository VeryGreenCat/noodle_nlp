import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import Seq2Seq
from preprocess import load_data, normalize_text, tokenize, remove_stopwords # get func from preprocess.py
import pickle

# 1. Load data
X_texts, y_texts = load_data("data/noodle_data.csv")

# 2. Tokenize input/output
X_tokens = [" ".join(tokenize(x)) for x in X_texts]
y_tokens = ["<start> " + " ".join(tokenize(y)) + " <end>" for y in y_texts]

# 3. Fit tokenizer
tokenizer = Tokenizer(filters="", oov_token="<unk>")
tokenizer.fit_on_texts(X_tokens + y_tokens)

X_seq = tokenizer.texts_to_sequences(X_tokens)
y_seq = tokenizer.texts_to_sequences(y_tokens)

# 4. Pad sequences
X_pad = pad_sequences(X_seq, padding="post")
y_pad = pad_sequences(y_seq, padding="post")

# Make max_seq_len for model = max of encoder and decoder
max_seq_len = max(X_pad.shape[1], y_pad.shape[1])

# Re-pad sequences to same length
X_pad = pad_sequences(X_seq, maxlen=max_seq_len, padding="post")
y_pad = pad_sequences(y_seq, maxlen=max_seq_len, padding="post")

# 5. Build model
vocab_size = len(tokenizer.word_index) + 1
model = Seq2Seq(vocab_size, max_seq_len)

# 6. Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 7. Prepare decoder input & target (teacher forcing)
decoder_inp = np.zeros_like(y_pad)
decoder_inp[:, 1:] = y_pad[:, :-1]
decoder_inp[:, 0] = tokenizer.word_index["<start>"]
decoder_tar = y_pad  # keep full length

# 8. Train
model.fit([X_pad, decoder_inp], decoder_tar, batch_size=32, epochs=10)

# 9. Save model
model.save("model.h5")



with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)