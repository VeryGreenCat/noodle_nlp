import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import Seq2Seq
from preprocess import load_data, preprocess_sentence

# 1. Load data
X_str, y_str = load_data("data/orders.csv")

# 2. Clean data
X_tokens = [preprocess_sentence(x, is_output=False) for x in X_str]
y_tokens = [preprocess_sentence(y, is_output=True) for y in y_str]

# 3. Encode text to integer sequences
tokenizer = Tokenizer(filters="", oov_token="<unk>")
tokenizer.fit_on_texts(X_tokens + y_tokens)
X_seq = tokenizer.texts_to_sequences(X_tokens)
y_seq = tokenizer.texts_to_sequences(y_tokens)

# 4. Padding sequences because RNN needs fixed length input
max_seq_len = max(len(x) for x in X_seq + y_seq)
X_pad = pad_sequences(X_seq, maxlen=max_seq_len, padding="post")
y_pad = pad_sequences(y_seq, maxlen=max_seq_len, padding="post")

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size=0.1, random_state=42)

# 7. teacher forcing
def make_decoder_input(y_pad, tokenizer):
    decoder_inp = np.zeros_like(y_pad)  # create same shape, filled with 0
    decoder_inp[:, 1:] = y_pad[:, :-1]  # shift everything right by 1
    decoder_inp[:, 0] = tokenizer.word_index["<start>"] # force first token to <start>
    return decoder_inp

decoder_inp_train = make_decoder_input(y_train, tokenizer)
decoder_inp_test = make_decoder_input(y_test, tokenizer)

# Build model
vocab_size = len(tokenizer.word_index) + 1
model = Seq2Seq(vocab_size, max_seq_len)

# Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train
history = model.fit(
    [X_train, decoder_inp_train],
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.1
)

# Evaluate
test_loss = model.evaluate([X_test, decoder_inp_test], y_test)
print("Test loss:", test_loss)

# Save model + tokenizer
model.save("model2.h5")
with open("tokenizer2.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
