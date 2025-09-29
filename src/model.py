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
