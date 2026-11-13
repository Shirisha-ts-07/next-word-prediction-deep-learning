import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------------
# 1. Load trained model
# -----------------------------------

model = load_model("model/next_word_model.keras")


# -----------------------------------
# 2. Load dataset
# -----------------------------------

with open("dataset/text.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()


# -----------------------------------
# 3. Create tokenizer
# -----------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# -----------------------------------
# 4. Find maximum sequence length
# -----------------------------------

sequences = []

for line in text.split("\n"):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        sequences.append(token_list[:i + 1])


max_sequence_len = max(len(seq) for seq in sequences)


# -----------------------------------
# 5. Predict next word
# -----------------------------------

def predict_next_word(seed_text):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predictions = model.predict(
        token_list,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_word = index_to_word.get(
        predicted_index,
        ""
    )

    return predicted_word


# -----------------------------------
# 6. Generate multiple words
# -----------------------------------

def generate_text(seed_text, number_of_words):

    generated_text = seed_text

    for _ in range(number_of_words):

        next_word = predict_next_word(generated_text)

        if next_word == "":
            break

        generated_text += " " + next_word

    return generated_text


# -----------------------------------
# 7. User input
# -----------------------------------

while True:

    print("\n--------------------------------")

    user_input = input(
        "Enter a sentence (type 'exit' to stop): "
    )

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    result = generate_text(
        user_input,
        5
    )

    print("\nGenerated text:")
    print(result)