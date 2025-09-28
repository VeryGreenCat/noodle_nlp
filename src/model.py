# import tensorflow as tf
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.models import Model

# # Seq2Seq encoder-decoder with LSTM
# class Seq2Seq(Model):
#     def __init__(self, vocab_size, embedding_dim=128, hidden_units=256):
#         super().__init__()
#         self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
#         self.encoder_lstm = LSTM(hidden_units, return_state=True)
#         self.decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
#         self.fc = Dense(vocab_size, activation="softmax")

#     def call(self, encoder_input, decoder_input):
#         # Encoder
#         encoder_emb = self.embedding(encoder_input)
#         _, state_h, state_c = self.encoder_lstm(encoder_emb)

#         # Decoder
#         decoder_emb = self.embedding(decoder_input)
#         decoder_output, _, _ = self.decoder_lstm(decoder_emb, initial_state=[state_h, state_c])

#         return self.fc(decoder_output)




# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# encoder_inputs = Input(shape=(20,))
# encoder_emb = Embedding(input_dim=50, output_dim=64)(encoder_inputs)
# encoder_lstm, state_h, state_c = LSTM(64, return_state=True)(encoder_emb)
# encoder_states = [state_h, state_c]

# decoder_inputs = Input(shape=(20,))
# decoder_emb = Embedding(input_dim=50, output_dim=64)(decoder_inputs)
# decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)
# decoder_dense = Dense(50, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)

# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

def Seq2Seq(vocab_size, max_seq_len):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_len,))
    enc_emb = Embedding(vocab_size, 64)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(64, return_state=True)(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_seq_len,))
    dec_emb = Embedding(vocab_size, 64)(decoder_inputs)
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
