import os
import json
import re
import numpy as np
import streamlit as st  # type: ignore
from joblib import load

import matplotlib.pyplot as plt
from datasets import load_dataset

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Page config + constants
st.set_page_config(page_title="Sentiment Analysis Demo", page_icon="🎬", layout="wide")

MODEL_PATH = os.path.join("models", "imdb_best_model.joblib")
METRICS_JSON = os.path.join("data", "imdb_metrics.json")
VISUALS_DIR = "visuals"
MODELS_DIR = "models"

OPTIONAL_MODEL_FILES = {
    "Best (auto)": "imdb_best_model.joblib",
    "LogReg": "imdb_logreg.joblib",
    "Naive Bayes": "imdb_naive_bayes.joblib",
    "Linear SVM (calibrated)": "imdb_linear_svm.joblib",
}

# Example banks (5 per category) + cycling
EXAMPLE_BANKS = {
    "Sarcasm 🙄": [
        "Wow… what a masterpiece… I totally didn't fall asleep twice 🙄",
        "Truly life-changing. I will never get those two hours back.",
        "10/10, would definitely not recommend to my worst enemy.",
        "I loved it so much I checked how much time was left every 5 minutes.",
        "Incredible. I especially enjoyed the part where nothing happened.",
    ],
    "Mixed feelings 🤔": [
        "The acting was great, but the story made no sense and the pacing was weird.",
        "Some scenes were amazing, but overall it felt messy and too long.",
        "I liked the characters, but the plot was all over the place.",
        "Beautiful visuals, weak writing. I'm conflicted.",
        "Not bad, not great. It had moments, but it didn't fully land.",
    ],
    "Very positive 😄": [
        "Absolutely loved it. Great performances, emotional story, and a beautiful ending.",
        "A genuinely fantastic movie. Funny, smart, and surprisingly moving.",
        "This was such a satisfying watch. I'd rewatch it in a heartbeat.",
        "Brilliant. The writing, directing, and soundtrack all worked perfectly for me.",
        "One of the best films I've seen this year. Highly recommend.",
    ],
    "Very negative 😡": [
        "Boring, messy, and way too long. I kept waiting for it to get good and it never did.",
        "Painfully predictable and not even fun-bad. Just bad.",
        "The dialogue was cringe and the plot was a mess.",
        "It tried so hard to be deep, but it was just confusing and dull.",
        "I couldn't connect to anything here. It felt like wasted potential.",
    ],
    "Low-information 🪣": [
        "and and and and and and and and and",
        "was was was was was was was was was",
        "the the the the the the the the the",
        "ok ok ok ok ok ok ok",
        "movie movie movie movie movie movie movie",
    ],
    "Short neutral 😐": [
        "It was okay.",
        "Not my thing.",
        "Meh.",
        "I don't know how I feel about it.",
        "It happened.",
    ],
}

NEGATION_PAIRS = [
    ("This movie was good.", "This movie was not good."),
    ("I liked the ending.", "I didn't like the ending."),
    ("The acting was great.", "The acting was not great."),
    ("It was enjoyable.", "It was not enjoyable."),
    ("I recommend it.", "I do not recommend it."),
]

CHALLENGES = [
    "Flip the prediction by changing ≤ 3 words.",
    "Keep the label, but drop confidence below the Neutral threshold.",
    "Make it POSITIVE without words like 'great', 'amazing', 'love'.",
    "Make it NEGATIVE without words like 'boring', 'bad', 'worst'.",
]

BANNED_POS = {"great", "amazing", "love", "loved", "excellent", "fantastic", "brilliant"}
BANNED_NEG = {"boring", "bad", "worst", "awful", "terrible", "trash", "horrible"}


def cycle_example(category: str):
    """Cycle through examples for a given category."""
    key = f"idx_{category}"
    if key not in st.session_state:
        st.session_state[key] = -1
    st.session_state[key] = (st.session_state[key] + 1) % len(EXAMPLE_BANKS[category])
    st.session_state["review_text"] = EXAMPLE_BANKS[category][st.session_state[key]]


# Helpers
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return load(path)


@st.cache_data
def load_metrics_json():
    if not os.path.exists(METRICS_JSON):
        return None
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def confidence_text(conf: float) -> str:
    if conf >= 0.90:
        return "Very confident"
    if conf >= 0.75:
        return "Confident"
    if conf >= 0.60:
        return "Somewhat confident"
    return "Uncertain"


def label_with_emoji(label: str) -> str:
    if label == "POSITIVE":
        return "POSITIVE 😄"
    if label == "NEGATIVE":
        return "NEGATIVE 😡"
    return "NEUTRAL / UNCERTAIN 😐"


def predict(text: str, model, neutral_threshold: float):
    proba = model.predict_proba([text])[0]  # [neg, pos]
    pred_idx = int(np.argmax(proba))
    conf = float(proba[pred_idx])

    if conf < neutral_threshold:
        label = "NEUTRAL / UNCERTAIN"
    else:
        label = "POSITIVE" if pred_idx == 1 else "NEGATIVE"

    return label, conf, proba

def get_pipeline_parts(pipeline):
    """Safely get (tfidf, clf) regardless of exact pipeline class."""
    try:
        tfidf = pipeline.named_steps.get("tfidf", None)
        clf = pipeline.named_steps.get("clf", None)
        return tfidf, clf
    except Exception:
        return None, None


def extract_linear_coef(pipeline):
    """
    Return (tfidf, coef_vector) for models that expose coef_.
    Supports:
      - LogisticRegression / Linear models with coef_
      - CalibratedClassifierCV wrapping a linear estimator with coef_
    """
    tfidf, clf = get_pipeline_parts(pipeline)
    if tfidf is None or clf is None:
        return None

    # direct linear model
    if hasattr(clf, "coef_"):
        return tfidf, clf.coef_[0]

    if hasattr(clf, "calibrated_classifiers_"):
        try:
            base = clf.calibrated_classifiers_[0].estimator
            if hasattr(base, "coef_"):
                return tfidf, base.coef_[0]
        except Exception:
            pass

    return None


def extract_nb_log_odds(pipeline):
    """
    Return (tfidf, log_odds_vector) for MultinomialNB-like models.
    MultinomialNB exposes feature_log_prob_ shape (n_classes, n_features).
    We compute log-odds: log P(word|POS) - log P(word|NEG).
    """
    tfidf, clf = get_pipeline_parts(pipeline)
    if tfidf is None or clf is None:
        return None

    if hasattr(clf, "feature_log_prob_") and clf.feature_log_prob_.shape[0] == 2:
        # assuming class 0 = NEG, class 1 = POS (common in your setup)
        log_odds = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        return tfidf, log_odds

    return None


def top_contributing_words(pipeline, text: str, top_k: int = 8):
    """
    Returns (pos_words, neg_words) with per-text contributions.

    Works for:
      - Linear coef_ models
      - MultinomialNB via log-odds from feature_log_prob_
    """
    lin = extract_linear_coef(pipeline)
    nb = extract_nb_log_odds(pipeline)

    if lin is None and nb is None:
        return None  # unsupported model type

    tfidf, weights = lin if lin is not None else nb

    X = tfidf.transform([text])  # sparse
    feature_names = np.array(tfidf.get_feature_names_out())

    # Contribution = tfidf_value * weight
    contrib = X.toarray()[0] * weights

    pos_idx = np.argsort(contrib)[::-1]
    neg_idx = np.argsort(contrib)

    pos_words = [(feature_names[i], float(contrib[i])) for i in pos_idx if contrib[i] > 0][:top_k]
    neg_words = [(feature_names[i], float(contrib[i])) for i in neg_idx if contrib[i] < 0][:top_k]

    return pos_words, neg_words


def plot_word_contrib(words_scores, title: str):
    """
    words_scores: list[(word, score)] where score can be +/-.
    """
    if not words_scores:
        st.caption("Nothing to plot.")
        return
    words = [w for w, _ in words_scores]
    scores = [s for _, s in words_scores]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(words, scores)
    ax.set_title(title)
    ax.set_ylabel("Contribution")
    ax.tick_params(axis="x", rotation=40)
    st.pyplot(fig, clear_figure=True)


@st.cache_data
def compute_eval_graphs(_model, sample_size: int = 2000, seed: int = 42):
    # ds = load_dataset("imdb")
    ds = load_dataset("imdb", revision="main")
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_texts), size=min(sample_size, len(test_texts)), replace=False)

    X = [test_texts[i] for i in idx]
    y = test_labels[idx]

    proba = _model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y, preds)
    fpr, tpr, _ = roc_curve(y, proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)

    max_conf = np.maximum(proba, 1 - proba)

    return {
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "max_conf": max_conf,
    }


@st.cache_data
def load_imdb_subset(split="test", n=2000, seed=42):
    # ds = load_dataset("imdb")
    # ds = load_dataset("imdb", download_mode="force_redownload")
    try:
        ds = load_dataset("imdb", revision="main")
    except Exception as e:
        st.error("Could not load the IMDB dataset from Hugging Face right now. Please try again later.")
        st.exception(e)
        st.stop()
    data = ds[split]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=min(n, len(data)), replace=False)
    texts = [data[int(i)]["text"] for i in idx]
    labels = np.array([data[int(i)]["label"] for i in idx])
    return texts, labels


def safe_image(path: str):
    return os.path.exists(path)

def normalize_tokens(text: str):
    # simple tokenization (good enough for the UI)
    return re.findall(r"\b[\w']+\b", text.lower())

def tokenize_for_diff(text: str):
    # consistent, simple tokens for counting edits
    return re.findall(r"\b[\w']+\b", text.lower())

def word_changes_count(a: str, b: str) -> int:
    """
    Count token changes by position (simple + good for classroom demos).
    If lengths differ, extra tokens count as changes.
    """
    A = tokenize_for_diff(a)
    B = tokenize_for_diff(b)
    n = max(len(A), len(B))
    changes = 0
    for i in range(n):
        wa = A[i] if i < len(A) else None
        wb = B[i] if i < len(B) else None
        if wa != wb:
            changes += 1
    return changes

def check_challenge(
    challenge: str,
    base_text: str,
    base_label: str,
    base_conf: float,
    new_label: str,
    new_conf: float,
    edited_text: str,
    neutral_threshold: float
):
    if challenge.startswith("Flip"):
        changes = word_changes_count(base_text, edited_text)
        flipped = (new_label != base_label)
        ok = flipped and (changes <= 3)

        if ok:
            return True, f"✅ Flipped with {changes} word changes (≤ 3)."
        if not flipped:
            return False, f"Not flipped yet! Changes used: {changes}/3. Try swapping 1–3 strong sentiment words."
        return False, f"Flipped, but you used {changes} word changes (> 3). Try again with smaller edits."

    if challenge.startswith("Keep the label"):
        # Make this behave like humans expect: keep POS or NEG, not NEUTRAL baseline
        if base_label == "NEUTRAL / UNCERTAIN":
            return False, "Baseline is NEUTRAL/UNCERTAIN. Pick a POSITIVE or NEGATIVE baseline first (Analyze in Tab 1)."

        ok = (new_label == base_label) and (new_conf < neutral_threshold) and (new_label != "NEUTRAL / UNCERTAIN")
        if ok:
            return True, "✅ Same label, but now uncertain (below threshold)."
        if new_label != base_label:
            return False, "Label changed! Try keeping the sentiment direction, but weaken the strong cues."
        return False, "Confidence still too high! Remove/soften strong sentiment words, or add mixed cues."

    if "POSITIVE without" in challenge:
        toks = set(normalize_tokens(edited_text))
        banned_used = toks.intersection(BANNED_POS)
        ok = (new_label == "POSITIVE") and not banned_used
        if ok:
            return True, "✅ Positive without the obvious positive words."
        if banned_used:
            return False, f"You used banned words: {', '.join(sorted(banned_used))}"
        return False, "Still not POSITIVE. Try cues like 'enjoyable', 'fun', 'moving', 'worth it', 'rewatch'."

    if "NEGATIVE without" in challenge:
        toks = set(normalize_tokens(edited_text))
        banned_used = toks.intersection(BANNED_NEG)
        ok = (new_label == "NEGATIVE") and not banned_used
        if ok:
            return True, "✅ Negative without the obvious negative words."
        if banned_used:
            return False, f"You used banned words: {', '.join(sorted(banned_used))}"
        return False, "Still not NEGATIVE. Try cues like 'messy', 'predictable', 'cringe', 'wasted', 'dragged'."

    return False, "Keep experimenting!"

# Sidebar
st.sidebar.title("⚙️ Controls")

# Model choice (if files exist)
available_models = {}
for display, fname in OPTIONAL_MODEL_FILES.items():
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        available_models[display] = path

if not available_models:
    available_models = {"Best (auto)": MODEL_PATH}

model_choice = st.sidebar.selectbox("Model", list(available_models.keys()), index=0)
model_path = available_models[model_choice]

neutral_threshold = st.sidebar.slider(
    "Neutral threshold",
    0.50, 0.90, 0.60, 0.01,
    help="If confidence is below this threshold, the app returns NEUTRAL/UNCERTAIN."
)

learning_mode = st.sidebar.toggle(
    "🎓 Learning mode",
    value=True,
    help="Adds guided steps + mini challenges for learning / classroom demos."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick examples (click to cycle)")

for category in EXAMPLE_BANKS.keys():
    st.sidebar.button(
        category,
        on_click=cycle_example,
        args=(category,),
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Click a category multiple times to cycle through examples.")

# Main header
st.title("🎬 Sentiment Analysis - IMDB Reviews")
st.write(
    "An interactive demo using **TF-IDF + classic ML** (LogReg / Naive Bayes / calibrated Linear SVM). "
    "Type a review and see the prediction + confidence."
)

model = load_model(model_path)
if model is None:
    st.error(
        f"Model not found at `{model_path}`.\n\n"
        "Train it from the project root:\n"
        "`python -m src.train_sklearn_imdb`"
    )
    st.stop()

metrics = load_metrics_json()

if st.sidebar.button("🧹 Clear cache & reload"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["✅ Try it", "🧪 Playground", "🗂️ Dataset explorer", "🧠 How it works", "📊 Evaluation", "ℹ️ About"]
)


# Tab 1: Try it (guided + polished)
with tab1:
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""

    st.caption("Click sidebar examples to fill the textbox (each click cycles).")

    st.markdown("**Quick start:**")
    cbtn = st.columns(6)
    keys = list(EXAMPLE_BANKS.keys())
    for i, cat in enumerate(keys[:6]):
        if cbtn[i].button(cat, use_container_width=True):
            cycle_example(cat)

    if learning_mode:
        st.info(
            "Learning flow: **Pick an example → Analyze → See why → Go to the 🧪 Playground tab → Try the mini challenge**.\n\n"
            "Tip: The challenge is checked when you click **Analyze EDITED text** in the Playground."
        )


    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.subheader("Write or paste a review")
        st.session_state["review_text"] = st.text_area(
            "Review text",
            value=st.session_state["review_text"],
            height=200,
            placeholder="e.g. The acting was great but the story made no sense...",
            label_visibility="collapsed",
        )
        st.caption("Tip: Try sarcasm, mixed reviews, or very short text to see where classic NLP struggles.")

    with colB:
        st.subheader("Run")
        run = st.button("Analyze", type="primary", use_container_width=True)

        if learning_mode:
            challenge = st.selectbox("Mini challenge", CHALLENGES)
        else:
            challenge = None

        st.markdown("#### Output")
        st.write("- Label (POS / NEG / NEUTRAL)")
        st.write("- Confidence + probabilities")
        st.caption(f"Neutral rule: if max probability < {neutral_threshold:.2f} ➜ NEUTRAL/UNCERTAIN")

    if run:
        text = st.session_state["review_text"].strip()
        if not text:
            st.warning("Please enter some text first.")
        else:
            label, conf, proba = predict(text, model, neutral_threshold)
            neg_p, pos_p = float(proba[0]), float(proba[1])

            st.markdown("---")
            out1, out2, out3 = st.columns([1.2, 1, 1])

            with out1:
                st.metric("Prediction", label_with_emoji(label))
                st.caption(f"{confidence_text(conf)} • threshold={neutral_threshold:.2f} • model={model_choice}")

            with out2:
                st.metric("Confidence", f"{conf:.3f}")
                st.progress(min(1.0, max(0.0, conf)))

            with out3:
                st.write("**Probabilities**")
                st.write(f"NEGATIVE: `{neg_p:.3f}`")
                st.progress(neg_p)
                st.write(f"POSITIVE: `{pos_p:.3f}`")
                st.progress(pos_p)

            explained = top_contributing_words(model, text, top_k=10)

            with st.expander("Explain this prediction (top contributing words + plots)"):
                if explained is None:
                    st.write("This model type doesn't expose linear weights for explanation.")
                else:
                    pos_words, neg_words = explained
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Words pushing POSITIVE**")
                        for w, s in pos_words:
                            st.write(f"- `{w}` ({s:+.4f})")
                        plot_word_contrib(pos_words[:8], "Positive contributions (top)")
                    with c2:
                        st.markdown("**Words pushing NEGATIVE**")
                        for w, s in neg_words:
                            st.write(f"- `{w}` ({s:+.4f})")
                        plot_word_contrib(neg_words[:8], "Negative contributions (top)")

            with st.expander("Why can nonsense text still get a confident label? (bag-of-words / “bag of holding”)"):
                st.markdown(
                    """
Classic TF-IDF models are **bag-of-words**:
- they don't understand grammar, meaning, or sarcasm
- they mostly react to **which tokens appear** and learned weights

If you write low-information text (e.g. repeated common words like *“and and and…”*),
the TF-IDF vector can become almost empty, and the classifier may fall back to its learned **bias**.
That can produce confident outputs even when the text doesn't “mean” anything.
"""
                )

            # store baseline for Playground tab (challenge checking)
            st.session_state["baseline_text"] = text
            st.session_state["baseline_label"] = label
            st.session_state["baseline_conf"] = conf
            st.session_state["baseline_challenge"] = challenge
            st.session_state["baseline_model"] = model_choice
            st.session_state["baseline_threshold"] = float(neutral_threshold)

            # tell the user what to do next
            if learning_mode and challenge:
                st.success(
                    f"Baseline saved for the mini challenge ✅  Now open **🧪 Playground** and click **Analyze EDITED text** to check your attempt.\n\n"
                    f"Challenge: {challenge}"
                )

# Tab 2: Playground (counterfactuals + negation demo + challenge checker)
with tab2:
    st.subheader("🧪 Playground: learn by changing the input")

    if learning_mode:
        st.info("This tab is where people *actually learn*: remove words, compare negation, and see the model react.")

    base_text = st.session_state.get("baseline_text", st.session_state.get("review_text", ""))
    if not base_text:
        base_text = EXAMPLE_BANKS["Mixed feelings 🤔"][0]

    st.markdown("### A) Counterfactual: remove words and re-run")
    st.caption("Pick words to remove and watch how the prediction changes (bag-of-words behavior).")

    tokens = base_text.split()
    unique_tokens = sorted(set(tokens), key=lambda x: x.lower())
    remove = st.multiselect("Remove words", options=unique_tokens)

    edited = " ".join([t for t in tokens if t not in remove])
    edited_text = st.text_area("Edited text", edited, height=130)

    colx, coly = st.columns([1, 1])
    with colx:
        run_base = st.button("Analyze BASE text", use_container_width=True)
    with coly:
        run_edit = st.button("Analyze EDITED text", type="primary", use_container_width=True)

    if run_base:
        lbl, c, p = predict(base_text, model, neutral_threshold)
        st.write("**BASE**:", label_with_emoji(lbl), f"(conf={c:.3f})", f"NEG={float(p[0]):.3f} POS={float(p[1]):.3f}")

    if run_edit:
        lbl2, c2, p2 = predict(edited_text, model, neutral_threshold)
        st.write("**EDITED**:", label_with_emoji(lbl2), f"(conf={c2:.3f})", f"NEG={float(p2[0]):.3f} POS={float(p2[1]):.3f}")

        # mini challenge check (if baseline exists)
        if learning_mode and st.session_state.get("baseline_label") is not None:
            # warn if baseline was created under different settings
            if (st.session_state.get("baseline_model") != model_choice) or (
                st.session_state.get("baseline_threshold") != float(neutral_threshold)
            ):
                st.warning("Baseline was created with a different model/threshold. Re-run Analyze in Tab 1 for a fair challenge.")

            base_label = st.session_state.get("baseline_label")
            base_conf = st.session_state.get("baseline_conf", 0.0)
            base_text_saved = st.session_state.get("baseline_text", base_text)
            challenge = st.session_state.get("baseline_challenge", CHALLENGES[0]) or CHALLENGES[0]

            ok, msg = check_challenge(
                challenge=challenge,
                base_text=base_text_saved,
                base_label=base_label,
                base_conf=base_conf,
                new_label=lbl2,
                new_conf=c2,
                edited_text=edited_text,
                neutral_threshold=neutral_threshold,
            )

            st.markdown("### 🎯 Mini challenge result")
            st.write(f"**Challenge:** {challenge}")
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

    st.markdown("---")
    st.markdown("### B) Negation demo (classic failure mode)")
    pair = st.selectbox("Choose a negation pair", NEGATION_PAIRS, index=0)
    a_text, b_text = pair

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**A (no negation)**")
        st.write(a_text)
        if st.button("Analyze A", use_container_width=True, key="negA"):
            la, caa, pa = predict(a_text, model, neutral_threshold)
            st.write(label_with_emoji(la), f"(conf={caa:.3f})")
    with cb:
        st.markdown("**B (with negation)**")
        st.write(b_text)
        if st.button("Analyze B", type="primary", use_container_width=True, key="negB"):
            lb, cbb, pb = predict(b_text, model, neutral_threshold)
            st.write(label_with_emoji(lb), f"(conf={cbb:.3f})")

    st.caption("Bag-of-words often under-handles negation because the meaning depends on context, not just token presence.")


# Tab 3: Dataset explorer (mistakes finder)
with tab3:
    st.subheader("🗂️ Dataset explorer: browse real IMDB reviews + find model mistakes")

    st.caption(
        "This uses a subset so it stays responsive. Great for demos: "
        "show a correct prediction, then jump to a mistake and discuss why."
    )

    n = st.slider("Subset size", 200, 5000, 1500, 100)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    if st.button("Load subset", type="primary"):
        texts, labels = load_imdb_subset("test", n=n, seed=int(seed))
        st.session_state["ds_texts"] = texts
        st.session_state["ds_labels"] = labels

        # here i clear old predictions so shapes don't mismatch / no error for users
        st.session_state["ds_preds"] = None
        st.session_state["ds_proba_pos"] = None
        st.session_state["ds_preds_cache_key"] = None

        st.success(f"Loaded {len(texts)} examples. Now click **Compute predictions on subset**.")


    texts = st.session_state.get("ds_texts")
    labels = st.session_state.get("ds_labels")

    if texts is None or labels is None:
        st.info("Click **Load subset** to begin.")
    else:
        # Optional: compute mistakes on demand
        show_only_wrong = st.toggle("Show only wrong predictions", value=False)

        if "ds_preds_cache_key" not in st.session_state:
            st.session_state["ds_preds_cache_key"] = None

        compute_preds = st.button("Compute predictions on subset")
        if compute_preds or (show_only_wrong and st.session_state["ds_preds_cache_key"] != (n, int(seed), model_choice, neutral_threshold)):
            with st.spinner("Predicting on subset…"):
                proba_pos = model.predict_proba(texts)[:, 1]
                preds = (proba_pos >= 0.5).astype(int)

            st.session_state["ds_preds"] = preds
            st.session_state["ds_proba_pos"] = proba_pos
            st.session_state["ds_preds_cache_key"] = (n, int(seed), model_choice, neutral_threshold)

        preds = st.session_state.get("ds_preds")
        proba_pos = st.session_state.get("ds_proba_pos")

        if preds is None or proba_pos is None:
            st.info("Click **Compute predictions on subset**.")
        else:
            if preds is None or proba_pos is None:
                st.info("Click **Compute predictions on subset**.")
                st.stop()

            if len(preds) != len(labels):
                st.warning("Subset changed — please click **Compute predictions on subset** again.")
                st.stop()
            wrong_idx = np.where(preds != labels)[0]
            right_idx = np.where(preds == labels)[0]

            st.write(f"✅ Correct: **{len(right_idx)}** • ❌ Wrong: **{len(wrong_idx)}**")

            if show_only_wrong:
                pool = wrong_idx
                if len(pool) == 0:
                    st.success("No mistakes found in this subset 🎉 (try a larger subset or a different seed).")
            else:
                pool = np.arange(len(texts))

            if len(pool) > 0:
                pick = int(st.selectbox("Pick an example", pool.tolist(), index=0))
                true_label = "POSITIVE" if labels[pick] == 1 else "NEGATIVE"
                pred_label = "POSITIVE" if preds[pick] == 1 else "NEGATIVE"
                ppos = float(proba_pos[pick])
                conf = max(ppos, 1 - ppos)

                st.markdown("---")
                st.markdown("### Example")
                st.write(texts[pick])

                c1, c2, c3 = st.columns(3)
                c1.metric("True label", true_label)
                c2.metric("Pred label", pred_label)
                c3.metric("Confidence", f"{conf:.3f}")

                with st.expander("Explain this example (top contributing words)"):
                    explained = top_contributing_words(model, texts[pick], top_k=10)
                    if explained is None:
                        st.write("This model type doesn't expose linear weights for explanation.")
                    else:
                        pos_words, neg_words = explained
                        a, b = st.columns(2)
                        with a:
                            st.markdown("**Push POSITIVE**")
                            for w, s in pos_words[:8]:
                                st.write(f"- `{w}` ({s:+.4f})")
                        with b:
                            st.markdown("**Push NEGATIVE**")
                            for w, s in neg_words[:8]:
                                st.write(f"- `{w}` ({s:+.4f})")


# ----------------------------
# Tab 4: How it works
# ----------------------------
with tab4:
    st.subheader("Pipeline overview")

    st.info(
        "This project uses a **classical NLP pipeline**: TF-IDF → linear classifier → probabilities → optional NEUTRAL threshold."
    )

    st.markdown(
        """
### End-to-end flow (what happens when you click Analyze)

**Text** ➜ **TF-IDF vectorizer** ➜ **Linear classifier** ➜ **Probabilities** ➜ **Neutral threshold rule**

- TF-IDF turns text into a vector of weighted word/phrase features.
- The classifier learns weights correlated with positive vs negative reviews.
- If the model is not confident enough, we label it **Neutral/Uncertain**.
"""
    )

    with st.expander("Why TF-IDF?"):
        st.markdown(
            """
TF-IDF is a strong baseline for sentiment analysis because:
- it's fast and lightweight
- it handles large vocabularies well
- it often performs surprisingly well on datasets like IMDB

But it is limited because it doesn't understand context or meaning (bag-of-words).
"""
        )

    st.markdown(
        """
### Classifiers used
- **Logistic Regression**: strong baseline + interpretable weights  
- **Multinomial Naive Bayes**: fast baseline  
- **Linear SVM (calibrated)**: often best performance on TF-IDF text

### Neutral / uncertain handling
If the highest class probability is below a threshold, the app outputs:
**NEUTRAL / UNCERTAIN**.
This helps with:
- short text
- mixed sentiment
- sarcasm
- low-information input
"""
    )

    with st.expander("Pros / Cons"):
        st.markdown(
            """
**Pros**
- Fast training & inference
- Strong baseline performance on IMDB
- Easier to deploy than deep models
- Can be interpreted (top contributing words for linear models)

**Cons**
- Bag-of-words: no real semantics
- Struggles with sarcasm / irony
- Can be overconfident on low-information input
- Domain shift (IMDB vs Letterboxd/Reddit slang)
"""
        )

# Tab 5: Evaluation (graphs)
with tab5:
    st.subheader("Model performance & visual diagnostics")

    # A) Model comparison from metrics.json (if available)
    if metrics and "models" in metrics:
        st.markdown("### Model comparison (from training run)")
        model_names = list(metrics["models"].keys())
        test_acc = [metrics["models"][m].get("test_accuracy", None) for m in model_names]
        val_acc = [metrics["models"][m].get("val_accuracy", None) for m in model_names]

        if all(v is not None for v in test_acc) and all(v is not None for v in val_acc):
            st.bar_chart(
                {
                    "validation_accuracy": {m: float(metrics["models"][m]["val_accuracy"]) for m in model_names},
                    "test_accuracy": {m: float(metrics["models"][m]["test_accuracy"]) for m in model_names},
                }
            )
            best = metrics.get("best_model", None)
            if best:
                st.success(f"Best model selected: **{best}**")
        else:
            st.caption("Metrics JSON found, but accuracies were missing.")
    else:
        st.warning(
            "No `data/imdb_metrics.json` found. Run training to generate it:\n"
            "`python -m src.train_sklearn_imdb`"
        )

    st.markdown("---")

    st.markdown("### Evaluation graphs (computed on a subset of IMDB test set)")
    st.caption("Click to compute on a subset so the app stays responsive.")

    sample_size = st.slider("Sample size", 500, 5000, 2000, 500)
    compute = st.button("Compute graphs on subset", type="primary")

    if compute:
        with st.spinner("Computing graphs…"):
            res = compute_eval_graphs(model, sample_size=sample_size)
    
        cm = res["cm"]
        tn, fp, fn, tp = cm.ravel()

        total = tn + fp + fn + tp
        acc = (tp + tn) / total if total else 0.0
        precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
        recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)) if (precision_pos + recall_pos) else 0.0

        st.markdown("### Results")

        # Confusion Matrix row
        left, right = st.columns([1.1, 1.2], gap="large")
        with left:
            fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.6))
            disp = ConfusionMatrixDisplay(res["cm"], display_labels=["NEG", "POS"])
            disp.plot(ax=ax_cm, values_format="d", colorbar=False)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm, clear_figure=True)
        with right:
            st.markdown("### Confusion Matrix (what each box means)")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("True Negatives (TN)", int(tn))
            m2.metric("False Positives (FP)", int(fp))
            m3.metric("False Negatives (FN)", int(fn))
            m4.metric("True Positives (TP)", int(tp))

            st.markdown(
                """
        **How to read it:**
        - **TN:** negative review predicted as negative
        - **FP:** negative review predicted as positive
        - **FN:** positive review predicted as negative
        - **TP:** positive review predicted as positive
        """
            )

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Accuracy", f"{acc:.3f}")
            k2.metric("Precision (POS)", f"{precision_pos:.3f}")
            k3.metric("Recall (POS)", f"{recall_pos:.3f}")
            k4.metric("Specificity (NEG)", f"{specificity:.3f}")
            k5.metric("F1 (POS)", f"{f1:.3f}")


        st.markdown("---")

        # ROC row
        left, right = st.columns([1.1, 1.2], gap="large")
        with left:
            fig_roc, ax_roc = plt.subplots(figsize=(4.8, 3.6))
            ax_roc.plot(res["fpr"], res["tpr"], label=f"AUC = {res['roc_auc']:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc, clear_figure=True)
        with right:
            roc_auc = res["roc_auc"]

            st.markdown("### ROC Curve (what it tells you)")
            st.markdown(
                """
        ROC shows the trade-off between:
        - **TPR / Recall** (catching positives)
        - **FPR** (accidentally calling negatives “positive”)

        AUC is a single number summary:
        - **0.5** ≈ random guessing  
        - **1.0** = perfect separation
        """
            )

            if roc_auc >= 0.90:
                st.success(f"AUC = {roc_auc:.3f} → Very strong separation.")
            elif roc_auc >= 0.80:
                st.info(f"AUC = {roc_auc:.3f} → Good separation.")
            elif roc_auc >= 0.70:
                st.warning(f"AUC = {roc_auc:.3f} → Moderate separation.")
            else:
                st.error(f"AUC = {roc_auc:.3f} → Weak separation (close to random).")

        st.markdown("---")

        # PR row
        left, right = st.columns([1.1, 1.2], gap="large")
        with left:
            fig_pr, ax_pr = plt.subplots(figsize=(4.8, 3.6))
            ax_pr.plot(res["recall"], res["precision"], label=f"AP = {res['ap']:.3f}")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision–Recall Curve")
            ax_pr.legend(loc="lower left")
            st.pyplot(fig_pr, clear_figure=True)
        with right:
            ap = res["ap"]

            st.markdown("### Precision–Recall (when it’s useful)")
            st.markdown(
                """
        Precision–Recall is helpful when:
        - classes are imbalanced, or
        - you care about avoiding FP/FN.

        **AP (Average Precision)** summarizes the PR curve:
        higher is better.
        """
            )

            if ap >= 0.90:
                st.success(f"AP = {ap:.3f} → Excellent trade-off.")
            elif ap >= 0.80:
                st.info(f"AP = {ap:.3f} → Good trade-off.")
            elif ap >= 0.70:
                st.warning(f"AP = {ap:.3f} → Moderate trade-off.")
            else:
                st.error(f"AP = {ap:.3f} → Weak trade-off.")

        st.markdown("---")

        # Confidence histogram row
        left, right = st.columns([1.1, 1.2], gap="large")
        with left:
            fig_hist, ax_hist = plt.subplots(figsize=(4.8, 3.6))
            ax_hist.hist(res["max_conf"], bins=25)
            ax_hist.set_xlabel("Max class probability")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Confidence Histogram")
            st.pyplot(fig_hist, clear_figure=True)
        with right:
            max_conf = res["max_conf"]
            mean_conf = float(np.mean(max_conf))
            median_conf = float(np.median(max_conf))
            low_conf_rate = float(np.mean(max_conf < neutral_threshold))

            st.markdown("### Confidence Histogram (why it matters)")
            st.markdown(
                f"""
        This plot shows **how sure the model usually is**.

        Each example gives `[P(NEG), P(POS)]`.  
        We take **max probability** as “confidence”.

        Your **Neutral threshold** is `{neutral_threshold:.2f}`:
        below that, the demo would label it **NEUTRAL/UNCERTAIN**.
        """
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean confidence", f"{mean_conf:.3f}")
            c2.metric("Median confidence", f"{median_conf:.3f}")
            c3.metric(f"Below threshold", f"{low_conf_rate*100:.1f}%")

            if low_conf_rate >= 0.30:
                st.warning("Many examples fall below the threshold → you'll see lots of NEUTRAL/UNCERTAIN.")
            elif low_conf_rate >= 0.15:
                st.info("Some uncertainty → NEUTRAL/UNCERTAIN will show up sometimes (nice for demos).")
            else:
                st.success("Model is usually confident → few NEUTRAL/UNCERTAIN cases.")

    # added so the tab ishidden when no visuals found
    st.markdown("---")
    cm_png = os.path.join(VISUALS_DIR, "confusion_matrix.png")
    roc_png = os.path.join(VISUALS_DIR, "roc_curve.png")
    pr_png = os.path.join(VISUALS_DIR, "pr_curve.png")

    if safe_image(cm_png) or safe_image(roc_png) or safe_image(pr_png):
        with st.expander("📁 Saved evaluation plots (exported PNGs)"):
            cols = st.columns(3)
            if safe_image(cm_png):
                cols[0].image(cm_png, caption="Confusion Matrix", use_container_width=True)
            if safe_image(roc_png):
                cols[1].image(roc_png, caption="ROC Curve", use_container_width=True)
            if safe_image(pr_png):
                cols[2].image(pr_png, caption="Precision-Recall Curve", use_container_width=True)

# ----------------------------
# Tab 6: About
# ----------------------------
with tab6:
    st.subheader("About")

    st.markdown(
        """
Hi! I'm **Maria Hadjichristoforou**, a Computer Science student at the **University of Cyprus**.

I built this project as a hands-on NLP and Machine Learning project to **explore how it actually behaves in practice**.  
I wanted something people could *interact with*, experiment on, and learn from. Moreover I wanted to show where the model succeeds, where it struggles, and why those failures happen.

Instead of treating sentiment analysis as a black box, this demo focuses on:
- interpretability (what words push a prediction one way or another),
- uncertainty (when the model *shouldn't* be confident),
- and classic limitations of bag-of-words approaches like TF-IDF.

### Course inspiration
This project was inspired by my Erasmus course:

- **DS817: Algorithms We Live By - SDU**
- Taught by **Prof. Pantelis Pipergias Analytis**  
  Find him on [LinkedIn](https://www.linkedin.com/in/pantelis-pipergias-analytis-31068146/)

"""
    )

    st.markdown("### Links")
    st.markdown(
        """
- GitHub: https://github.com/foroum  
- LinkedIn: https://www.linkedin.com/in/mhadjichristoforou/  
"""
    )

st.caption("Classic NLP sentiment analysis demo")