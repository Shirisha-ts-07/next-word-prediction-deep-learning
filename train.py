import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# -------------------------------
# 1. Load dataset
# -------------------------------

with open("dataset/text.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

print("Dataset loaded successfully.")
print("Number of characters:", len(text))


# -------------------------------
# 2. Tokenization
# -------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Vocabulary size:", total_words)


# -------------------------------
# 3. Create sequences
# -------------------------------

input_sequences = []

for line in text.split("\n"):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        sequence = token_list[:i + 1]
        input_sequences.append(sequence)


print("Total sequences:", len(input_sequences))


# -------------------------------
# 4. Padding
# -------------------------------

max_sequence_len = max(
    len(sequence) for sequence in input_sequences
)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )
)

print("Maximum sequence length:", max_sequence_len)


# -------------------------------
# 5. Create X and y
# -------------------------------

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(
    y,
    num_classes=total_words
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# -------------------------------
# 6. Build LSTM model
# -------------------------------

model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=128
    ),

    LSTM(150),

    Dropout(0.2),

    Dense(
        total_words,
        activation="softmax"
    )
])


# -------------------------------
# 7. Compile model
# -------------------------------

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


print("\nModel architecture:")
model.summary()


# -------------------------------
# -------------------------------

print("\nStarting training...\n")

model.fit(
    X,
    y,
    epochs=100,
    batch_size=32,
    verbose=1
)


# -------------------------------
# 9. Save model
# -------------------------------

os.makedirs("model", exist_ok=True)

model.save("model/next_word_model.keras")

print("\n================================")
print("Model trained successfully!")
print("Model saved to:")
print("model/next_word_model.keras")
print("================================")