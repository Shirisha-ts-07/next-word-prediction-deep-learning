import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------

with open("dataset/text.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

print("Dataset loaded successfully.")

# -----------------------------------------
# 2. Tokenization
# -----------------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Vocabulary size:", total_words)

# -----------------------------------------
# 3. Create Input Sequences
# -----------------------------------------

input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        sequence = token_list[:i + 1]
        input_sequences.append(sequence)

max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(
    y,
    num_classes=total_words
)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------------------
# 4. Function to Create Model
# -----------------------------------------

def create_model(model_type):

    model = Sequential()

    model.add(
        Embedding(
            input_dim=total_words,
            output_dim=128
        )
    )

    if model_type == "RNN":
        model.add(SimpleRNN(150))

    elif model_type == "LSTM":
        model.add(LSTM(150))

    elif model_type == "GRU":
        model.add(GRU(150))

    model.add(
        Dense(
            total_words,
            activation="softmax"
        )
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


# -----------------------------------------
# 5. Train RNN
# -----------------------------------------

print("\n==============================")
print("Training Simple RNN")
print("==============================")

rnn_model = create_model("RNN")

rnn_history = rnn_model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    verbose=1
)


# -----------------------------------------
# 6. Train LSTM
# -----------------------------------------

print("\n==============================")
print("Training LSTM")
print("==============================")

lstm_model = create_model("LSTM")

lstm_history = lstm_model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    verbose=1
)


# -----------------------------------------
# 7. Train GRU
# -----------------------------------------

print("\n==============================")
print("Training GRU")
print("==============================")

gru_model = create_model("GRU")

gru_history = gru_model.fit(
    X,
    y,
    epochs=30,
    batch_size=32,
    verbose=1
)


# -----------------------------------------
# 8. Display Results
# -----------------------------------------

rnn_accuracy = rnn_history.history["accuracy"][-1]
lstm_accuracy = lstm_history.history["accuracy"][-1]
gru_accuracy = gru_history.history["accuracy"][-1]

print("\n======================================")
print("FINAL MODEL ACCURACY")
print("======================================")

print("Simple RNN Accuracy :", round(rnn_accuracy * 100, 2), "%")
print("LSTM Accuracy       :", round(lstm_accuracy * 100, 2), "%")
print("GRU Accuracy        :", round(gru_accuracy * 100, 2), "%")

print("======================================")

import matplotlib.pyplot as plt

# -----------------------------------------
# Accuracy Graph
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    rnn_history.history["accuracy"],
    label="Simple RNN"
)

plt.plot(
    lstm_history.history["accuracy"],
    label="LSTM"
)

plt.plot(
    gru_history.history["accuracy"],
    label="GRU"
)

plt.title("RNN vs LSTM vs GRU - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_comparison.png")
plt.close()


# -----------------------------------------
# Loss Graph
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    rnn_history.history["loss"],
    label="Simple RNN"
)

plt.plot(
    lstm_history.history["loss"],
    label="LSTM"
)

plt.plot( 
    gru_history.history["loss"],
    label="GRU"
)

plt.title("RNN vs LSTM vs GRU - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_comparison.png")
plt.close()

print("\nGraphs saved successfully!")