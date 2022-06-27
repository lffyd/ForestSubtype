import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from tensorflow import keras

def AE(input_size,Encoder_hidden_size,hidden_size,Decoder_hidden_size,output_size):

    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_size,)),
        layers.Dense(Encoder_hidden_size[0], activation='relu'),
        layers.Dense(Encoder_hidden_size[1], activation='relu'),
        layers.Dense(Encoder_hidden_size[2], activation='relu'),
        layers.Dense(Encoder_hidden_size[3], activation='relu'),
        layers.Dense(Encoder_hidden_size[4], activation='relu'),
        layers.Dense(Encoder_hidden_size[5], activation='relu'),
        # layers.Dense(Encoder_hidden_size[6], activation='relu'),
        layers.Dense(hidden_size,activation='relu'),
        layers.Dense(Decoder_hidden_size[0], activation='relu'),
        layers.Dense(Decoder_hidden_size[1], activation='relu'),
        layers.Dense(Decoder_hidden_size[2], activation='relu'),
        layers.Dense(Decoder_hidden_size[3], activation='relu'),
        layers.Dense(Decoder_hidden_size[4], activation='relu'),
        layers.Dense(Decoder_hidden_size[5], activation='relu'),
        # layers.Dense(Decoder_hidden_size[6], activation='relu'),
        layers.Dense(output_size)
    ])

    return model