import json
import pandas as pd
import tensorflow as tf
import numpy as np
import urllib

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
sentences = []
labels = []
with open("sarcasm.json") as f:
    data = json.load(f)
for i in data:
  sentences.append(i["headline"]), labels.append(i["is_sarcastic"])

d1=pd.DataFrame({"text":sentences,"target":labels})

from sklearn.model_selection import train_test_split
train_data, validation_data, train_labels, validation_labels = train_test_split(d1["text"].to_numpy(),
                                                                                    d1["target"].to_numpy(),
                                                                                    test_size=0.1,
                                                                                   shuffle=False)

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
max_length = 15
max_vocab_length = vocab_size
text_vector = TextVectorization(max_tokens=vocab_size,
                                    output_mode="int",
                                    output_sequence_length=max_length,
                                    )
text_vector.adapt(train_data)

embedding = tf.keras.layers.Embedding(input_dim=max_vocab_length,
                                          output_dim=128,
                                          embeddings_initializer='uniform',
                                          input_length=max_length)

inputs = tf.keras.layers.Input(shape=(1,), name="input_layer", dtype="string")
x = text_vector(inputs)
x = embedding(x)
x = tf.keras.layers.LSTM(64, activation='tanh')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
model.fit(train_data,train_labels,epochs=10,validation_data=(validation_data,validation_labels))

model.save('sarcastic')



