import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

from model import Seq2Seq
from preprocess import load_data, normalize_text, tokenize, remove_stopwords

# 1. Load data
X_texts, y_texts = load_data("D:/Code/NLP/noodle_nlp/data/orders_with_typos.csv")

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

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size=0.1, random_state=42)

# Prepare decoder input/target
def make_decoder_input(y_pad, tokenizer):
    decoder_inp = np.zeros_like(y_pad)
    decoder_inp[:, 1:] = y_pad[:, :-1]
    decoder_inp[:, 0] = tokenizer.word_index["<start>"]
    return decoder_inp

decoder_inp_train = make_decoder_input(y_train, tokenizer)
decoder_inp_test = make_decoder_input(y_test, tokenizer)

# 6. Build model
vocab_size = len(tokenizer.word_index) + 1
model = Seq2Seq(vocab_size, max_seq_len)

# 7. Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# 8. Train
history = model.fit(
    [X_train, decoder_inp_train],
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.1
)

# 9. Evaluate
test_loss = model.evaluate([X_test, decoder_inp_test], y_test)
print("Test loss:", test_loss)

# 10. Save model + tokenizer
model.save("model1.h5")
with open("tokenizer1.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

