from hashlib import sha256
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Keep project files discoverable even when Streamlit is launched elsewhere.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "next_word_model.keras"
DATASET_PATH = PROJECT_DIR / "dataset" / "text.txt"

st.set_page_config(
    page_title="Next Word Prediction Using Deep Learning",
    page_icon="N",
    layout="wide",
)

# A restrained palette and a small amount of custom styling keep the page focused.
st.markdown(
    """
    <style>
        :root {
            --ink: #172b3a;
            --muted: #61717d;
            --accent: #087e78;
            --line: #dce5e7;
            --surface: #f4f8f7;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 92% 2%, rgba(8, 126, 120, 0.09), transparent 28rem),
                #fbfcfa;
            color: var(--ink);
        }
        .block-container { max-width: 1100px; padding-top: 2.4rem; }
        .project-kicker {
            color: var(--accent); font-size: 0.78rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem;
        }
        .project-intro {
            color: var(--muted); font-size: 1.08rem; max-width: 760px;
            line-height: 1.65; margin: 0.4rem 0 1.5rem;
        }
        .process-step {
            background: var(--surface); border-top: 3px solid var(--accent);
            padding: 0.8rem 0.65rem; min-height: 4.2rem; text-align: center;
            color: var(--ink); font-weight: 650;
        }
        div[data-testid="stMetric"] {
            background: var(--surface); border: 1px solid var(--line);
            padding: 0.9rem 1rem; border-radius: 6px;
        }
        div[data-testid="stMetricValue"] { color: var(--accent); }
        .tech-list { color: var(--muted); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the LSTM model...")
def load_prediction_assets():
    """Load the trained model and rebuild its tokenizer from the training text."""
    model = load_model(MODEL_PATH)

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        text = file.read().lower()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    index_to_word = {
        index: word for word, index in tokenizer.word_index.items()
    }

    # Match the padded input length used by the existing training setup.
    sequences = []
    training_sentence_sequences = set()
    for line in text.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        if token_list:
            training_sentence_sequences.add(tuple(token_list))
        for index in range(1, len(token_list)):
            sequences.append(token_list[:index + 1])

    max_sequence_len = max(len(sequence) for sequence in sequences)
    return model, tokenizer, index_to_word, max_sequence_len, training_sentence_sequences


def predict_probabilities(texts, model, tokenizer, max_sequence_len):
    """Get next-word probabilities for one or more partial sentences."""
    token_lists = [tokenizer.texts_to_sequences([text])[0] for text in texts]
    padded_tokens = pad_sequences(
        token_lists,
        maxlen=max_sequence_len - 1,
        padding="pre",
    )
    return model.predict(padded_tokens, verbose=0)


def get_corpus_next_word_ids(token_ids, training_sentence_sequences):
    """Find next words observed after this exact token prefix in the dataset."""
    prefix = tuple(token_ids)
    if not prefix:
        return None

    next_word_ids = {
        sentence[len(prefix)]
        for sentence in training_sentence_sequences
        if len(sentence) > len(prefix) and sentence[:len(prefix)] == prefix
    }
    return next_word_ids or None


def get_blocked_next_word_ids(token_ids):
    """Block immediate repeats and continuations that repeat an existing trigram."""
    if not token_ids:
        return set()

    blocked_ids = {token_ids[-1]}
    if len(token_ids) >= 3:
        current_context = tuple(token_ids[-2:])
        for index in range(len(token_ids) - 2):
            if tuple(token_ids[index:index + 2]) == current_context:
                blocked_ids.add(token_ids[index + 2])
    return blocked_ids


def get_top_k_candidates(
    probabilities,
    index_to_word,
    allowed_word_ids=None,
    blocked_word_ids=None,
    top_k=12,
):
    """Return high-probability words supported by the optional filters."""
    blocked_word_ids = blocked_word_ids or set()
    candidates = []

    for index in np.argsort(probabilities)[::-1]:
        word_id = int(index)
        word = index_to_word.get(int(index), "")
        probability = float(probabilities[index])
        is_allowed = allowed_word_ids is None or word_id in allowed_word_ids
        if word and probability > 0 and is_allowed and word_id not in blocked_word_ids:
            candidates.append((word, probability))
        if len(candidates) == top_k:
            break

    return candidates


def sampling_weights(candidates, temperature=0.55):
    """Convert candidate probabilities into a temperature-scaled distribution."""
    scores = np.array(
        [np.log(probability) / temperature for _, probability in candidates],
        dtype=float,
    )
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


def generate_lstm_suggestions(
    seed_text,
    number_of_suggestions,
    words_per_suggestion,
    model,
    tokenizer,
    index_to_word,
    max_sequence_len,
    training_sentence_sequences,
    top_k=12,
):
    """Generate distinct continuations while preserving the exact user prefix."""
    input_tokens = tokenizer.texts_to_sequences([seed_text])[0]

    # Give each suggestion a different likely first word to avoid near-duplicates.
    first_probabilities = predict_probabilities(
        [seed_text], model, tokenizer, max_sequence_len
    )[0]
    first_supported_ids = get_corpus_next_word_ids(
        input_tokens, training_sentence_sequences
    )
    first_candidates = get_top_k_candidates(
        first_probabilities,
        index_to_word,
        allowed_word_ids=first_supported_ids,
        blocked_word_ids=get_blocked_next_word_ids(input_tokens),
        top_k=top_k,
    )
    if not first_candidates and first_supported_ids:
        first_candidates = get_top_k_candidates(
            first_probabilities,
            index_to_word,
            blocked_word_ids=get_blocked_next_word_ids(input_tokens),
            top_k=top_k,
        )
    if not first_candidates:
        return []

    seed_material = (
        f"{seed_text.casefold()}:{number_of_suggestions}:{words_per_suggestion}"
    )
    random_seed = int.from_bytes(sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(random_seed)
    first_choices = rng.choice(
        len(first_candidates),
        size=number_of_suggestions,
        replace=number_of_suggestions > len(first_candidates),
        p=sampling_weights(first_candidates),
    )
    suggestions = [
        {
            "text": seed_text + " " + first_candidates[choice][0],
            "token_ids": input_tokens + [
                tokenizer.word_index[first_candidates[choice][0]]
            ],
            "complete": False,
        }
        for choice in first_choices
    ]

    # Predict the next position for all active suggestions together.
    for _ in range(words_per_suggestion - 1):
        active_suggestions = [
            suggestion for suggestion in suggestions if not suggestion["complete"]
        ]
        if not active_suggestions:
            break

        probabilities_by_suggestion = predict_probabilities(
            [suggestion["text"] for suggestion in active_suggestions],
            model,
            tokenizer,
            max_sequence_len,
        )

        for suggestion, probabilities in zip(active_suggestions, probabilities_by_suggestion):
            supported_ids = get_corpus_next_word_ids(
                suggestion["token_ids"], training_sentence_sequences
            )
            blocked_ids = get_blocked_next_word_ids(suggestion["token_ids"])
            candidates = get_top_k_candidates(
                probabilities,
                index_to_word,
                allowed_word_ids=supported_ids,
                blocked_word_ids=blocked_ids,
                top_k=top_k,
            )
            if not candidates and supported_ids:
                candidates = get_top_k_candidates(
                    probabilities,
                    index_to_word,
                    blocked_word_ids=blocked_ids,
                    top_k=top_k,
                )
            if not candidates:
                suggestion["complete"] = True
                continue

            word = rng.choice(
                [candidate[0] for candidate in candidates],
                p=sampling_weights(candidates),
            )
            suggestion["text"] += " " + word
            suggestion["token_ids"].append(tokenizer.word_index[word])
            if tuple(suggestion["token_ids"]) in training_sentence_sequences:
                suggestion["complete"] = True

    # Keep a final duplicate guard in case the model has a very small vocabulary.
    unique_suggestions = []
    seen_suggestions = set()
    for suggestion in suggestions:
        normalized_text = " ".join(suggestion["text"].split()).casefold()
        if normalized_text not in seen_suggestions:
            seen_suggestions.add(normalized_text)
            unique_suggestions.append(suggestion["text"])

    return unique_suggestions


def get_ranked_corpus_suggestions(
    seed_text,
    input_tokens,
    number_of_suggestions,
    words_per_suggestion,
    model,
    max_sequence_len,
    training_sentence_sequences,
    index_to_word,
):
    """Rank observed sentence continuations by their LSTM next-word probabilities."""
    prefix = tuple(input_tokens)
    if not prefix:
        return []

    candidates = []
    seen_text = set()
    for sentence in training_sentence_sequences:
        if len(sentence) <= len(prefix) or sentence[:len(prefix)] != prefix:
            continue

        continuation = sentence[len(prefix):len(prefix) + words_per_suggestion]
        if not continuation:
            continue

        text = seed_text + " " + " ".join(
            index_to_word[word_id] for word_id in continuation
        )
        normalized_text = " ".join(text.split()).casefold()
        if normalized_text not in seen_text:
            seen_text.add(normalized_text)
            candidates.append({
                "text": text,
                "continuation": continuation,
                "score": 0.0,
            })

    if not candidates:
        return []

    # Score every next word in batches so ranking remains quick for the small corpus.
    for offset in range(words_per_suggestion):
        active_candidates = [
            candidate for candidate in candidates
            if len(candidate["continuation"]) > offset
        ]
        if not active_candidates:
            break

        contexts = [
            list(prefix) + list(candidate["continuation"][:offset])
            for candidate in active_candidates
        ]
        padded_contexts = pad_sequences(
            contexts,
            maxlen=max_sequence_len - 1,
            padding="pre",
        )
        probabilities_by_candidate = model.predict(padded_contexts, verbose=0)

        for candidate, probabilities in zip(
            active_candidates, probabilities_by_candidate
        ):
            next_word_id = candidate["continuation"][offset]
            candidate["score"] += np.log(max(float(probabilities[next_word_id]), 1e-12))

    candidates.sort(
        key=lambda candidate: (
            -candidate["score"] / len(candidate["continuation"]),
            candidate["text"].casefold(),
        )
    )
    return [candidate["text"] for candidate in candidates[:number_of_suggestions]]


def generate_suggestions(
    seed_text,
    number_of_suggestions,
    words_per_suggestion,
    model,
    tokenizer,
    index_to_word,
    max_sequence_len,
    training_sentence_sequences,
):
    """Prefer LSTM-scored corpus continuations, then fill from top-k sampling."""
    input_tokens = tokenizer.texts_to_sequences([seed_text])[0]
    suggestions = get_ranked_corpus_suggestions(
        seed_text,
        input_tokens,
        number_of_suggestions,
        words_per_suggestion,
        model,
        max_sequence_len,
        training_sentence_sequences,
        index_to_word,
    )
    if len(suggestions) >= number_of_suggestions:
        return suggestions

    sampled_suggestions = generate_lstm_suggestions(
        seed_text,
        number_of_suggestions,
        words_per_suggestion,
        model,
        tokenizer,
        index_to_word,
        max_sequence_len,
        training_sentence_sequences,
    )
    seen_text = {suggestion.casefold() for suggestion in suggestions}
    for suggestion in sampled_suggestions:
        if suggestion.casefold() not in seen_text:
            seen_text.add(suggestion.casefold())
            suggestions.append(suggestion)
        if len(suggestions) == number_of_suggestions:
            break

    return suggestions


st.markdown('<div class="project-kicker">Deep learning mini project</div>', unsafe_allow_html=True)
st.title("Next Word Prediction Using Deep Learning")
st.markdown(
    '<p class="project-intro">Explore how a trained Long Short-Term Memory (LSTM) '
    'network uses the words in a sentence to predict what may come next.</p>',
    unsafe_allow_html=True,
)

try:
    model, tokenizer, index_to_word, max_sequence_len, training_sentence_sequences = (
        load_prediction_assets()
    )
except (OSError, ValueError) as error:
    st.error(f"The model or tokenizer dataset could not be loaded: {error}")
    st.stop()

st.subheader("Try the predictor")
user_input = st.text_input(
    "Seed sentence",
    placeholder="For example: machine learning",
)
control_columns = st.columns(2)
with control_columns[0]:
    number_of_suggestions = st.selectbox(
        "Number of suggestions",
        options=(3, 5, 10),
        index=1,
    )
with control_columns[1]:
    words_per_suggestion = st.slider(
        "Words per suggestion",
        min_value=1,
        max_value=10,
        value=6,
    )

st.caption(
    f"Training text: {len(training_sentence_sequences)} sentences · "
    f"{len(tokenizer.word_index):,} vocabulary words · "
    f"{max_sequence_len - 1} tokens of model context"
)

if user_input.strip():
    known_tokens = tokenizer.texts_to_sequences([user_input])[0]
    if not known_tokens:
        st.info(
            "None of the seed words are in the training vocabulary. Suggestions "
            "will use the model's general learned word patterns, while keeping "
            "your original seed at the beginning."
        )
    elif get_corpus_next_word_ids(known_tokens, training_sentence_sequences):
        st.caption(
            "The LSTM scores continuations found after this seed in the training text."
        )
    else:
        st.caption(
            "No matching training prefix was found; suggestions use the LSTM's learned vocabulary."
        )
    with st.spinner("Generating suggestions..."):
        suggestions = generate_suggestions(
            user_input,
            number_of_suggestions,
            words_per_suggestion,
            model,
            tokenizer,
            index_to_word,
            max_sequence_len,
            training_sentence_sequences,
        )
    if suggestions:
        st.markdown("#### Suggested Sentences")
        suggestion_columns = st.columns(2)
        for index, suggestion in enumerate(suggestions, start=1):
            with suggestion_columns[(index - 1) % len(suggestion_columns)]:
                with st.container(border=True):
                    st.caption(f"SUGGESTION {index:02d}")
                    st.write(suggestion)
        st.caption(
            "Suggestions reflect patterns in this relatively small training "
            "dataset and may be incomplete or less natural for some inputs."
        )
        if len(suggestions) < number_of_suggestions:
            st.info(
                f"The model produced {len(suggestions)} unique suggestions "
                "from its available next-word candidates."
            )
    else:
        st.warning("The model could not find a next-word candidate for that input.")

st.divider()
st.subheader("About the Project")
st.write(
    "This project demonstrates next-word generation with a deep learning model. "
    "The saved LSTM learns patterns in the supplied AI, machine learning, and NLP text. "
    "Its suggestions are examples of learned continuations, not guaranteed facts or "
    "arbitrary natural language."
)

st.subheader("How It Works")
steps = st.columns(5)
for column, step in zip(
    steps,
    ("Text", "Tokenization", "Sequence Generation", "LSTM", "Prediction"),
):
    column.markdown(f'<div class="process-step">{step}</div>', unsafe_allow_html=True)

st.subheader("Model Comparison")
metric_columns = st.columns(3)
for column, name, accuracy in zip(
    metric_columns,
    ("Simple RNN", "LSTM", "GRU"),
    ("86.39%", "79.90%", "85.05%"),
):
    column.metric(name, accuracy, help="Accuracy reported for this project comparison.")

accuracy_chart = PROJECT_DIR / "accuracy_comparison.png"
loss_chart = PROJECT_DIR / "loss_comparison.png"
available_charts = [
    ("Accuracy Comparison", accuracy_chart),
    ("Loss Comparison", loss_chart),
]
available_charts = [(caption, path) for caption, path in available_charts if path.is_file()]
if available_charts:
    chart_columns = st.columns(len(available_charts))
    for column, (caption, path) in zip(chart_columns, available_charts):
        column.image(str(path), caption=caption, width="stretch")

st.subheader("Technologies")
st.markdown(
    '<p class="tech-list">Python &nbsp; · &nbsp; TensorFlow &nbsp; · &nbsp; '
    'Keras &nbsp; · &nbsp; NumPy &nbsp; · &nbsp; Streamlit</p>',
    unsafe_allow_html=True,
)