import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

#data loader
with open("data/data.json", "r", encoding="utf-8") as file:
    data = json.load(file)
questions = [item['question'] for item in data]
reponses = ["<start> " + item['reponse'] + " <end>" for item in data]

#tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + reponses)

# Convertir le texte en séquences
questions_seq = tokenizer.texts_to_sequences(questions)
reponses_seq = tokenizer.texts_to_sequences(reponses)

# Longueur maximale pour le padding
max_length = max(len(seq) for seq in questions_seq + reponses_seq)

# Appliquer le padding
questions_padded = pad_sequences(questions_seq, maxlen=max_length, padding="post")
reponses_padded = pad_sequences(reponses_seq, maxlen=max_length, padding="post")

# Préparer les données d'entraînement pour qu'elles soient alignées "mot par mot"
X = questions_padded
y = reponses_padded  # Gardez les séquences complètes dans y pour générer du texte.

#Paramètres  modèle
vocab_size = len(tokenizer.word_index) + 1  # Taille du vocabulaire

#model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    Dense(vocab_size, activation="softmax")
])


#compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# training modèle
y_shifted = np.expand_dims(np.roll(y, -1, axis=1), -1)
model.fit(X, y_shifted, epochs=1000, batch_size=32)

#save model
model.save("model/trained_model.h5")
with open("model/tokenizer.pkl", "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
