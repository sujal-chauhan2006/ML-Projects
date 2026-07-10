import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Shield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# NLTK setup (safe download, only runs once)
# ---------------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

ps = PorterStemmer()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 10px;
        font-size: 1rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 0;
        font-size: 1.05rem;
        transition: transform 0.15s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }
    .result-card {
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 1.2rem;
        animation: fadeIn 0.4s ease-in;
    }
    .spam-card {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
    }
    .ham-card {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #86efac;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stat-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

try:
    tfidf, model = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ---------------------------------------------------------------------------
# Session state for history / stats
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "spam_count" not in st.session_state:
    st.session_state.spam_count = 0
if "ham_count" not in st.session_state:
    st.session_state.ham_count = 0

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def transformed_data(message: str) -> str:
    message = message.lower()
    tokens = nltk.word_tokenize(message)

    y = [t for t in tokens if t.isalnum()]

    stop_words = set(stopwords.words("english"))
    y = [t for t in y if t not in stop_words and t not in string.punctuation]

    y = [ps.stem(t) for t in y]

    return " ".join(y)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Spam Shield")
    st.markdown(
        "A machine-learning powered classifier that flags spam "
        "emails and SMS messages in real time."
    )
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='stat-box'><b>🚨 Spam</b><br>{st.session_state.spam_count}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='stat-box'><b>✅ Ham</b><br>{st.session_state.ham_count}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("### 🧪 Try an example")
    example = st.selectbox(
        "Sample messages",
        [
            "-- select --",
            "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!!!",
            "Hey, are we still on for lunch tomorrow at 1pm?",
            "URGENT: Your account has been suspended. Verify your details immediately to avoid closure.",
            "Don't forget to bring the report for the meeting.",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.spam_count = 0
        st.session_state.ham_count = 0
        st.rerun()

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown("<p class='main-title'>📩 Email / SMS Spam Classifier</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Paste a message below and let the model do the rest</p>", unsafe_allow_html=True)

if not artifacts_loaded:
    st.warning(
        "⚠️ Could not find `vectorizer.pkl` and/or `model.pkl` in the app directory. "
        "Add them alongside this script to enable predictions."
    )

default_text = "" if example == "-- select --" else example
input_sms = st.text_area("Enter the message", value=default_text, height=150, placeholder="Type or paste a message here...")

col1, col2 = st.columns([3, 1])
with col1:
    predict_clicked = st.button("🔍 Predict", disabled=not artifacts_loaded)
with col2:
    show_details = st.toggle("Show details", value=False)

if predict_clicked:
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analyzing message..."):
            transform_sms = transformed_data(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = model.predict(vector_input)[0]

            # Try to get a confidence score if the model supports it
            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector_input)[0]
                confidence = max(proba) * 100

        if result == 1:
            st.session_state.spam_count += 1
            label = "🚨 Spam Detected"
            css_class = "spam-card"
        else:
            st.session_state.ham_count += 1
            label = "✅ Not Spam"
            css_class = "ham-card"

        conf_text = f"<br><span style='font-size:0.95rem;font-weight:400;'>Confidence: {confidence:.1f}%</span>" if confidence is not None else ""
        st.markdown(
            f"<div class='result-card {css_class}'>{label}{conf_text}</div>",
            unsafe_allow_html=True,
        )

        if show_details:
            with st.expander("🔬 Preprocessing details", expanded=True):
                st.markdown("**Cleaned & stemmed text used for prediction:**")
                st.code(transform_sms or "(empty after preprocessing)")

        st.session_state.history.insert(
            0, {"message": input_sms, "label": "Spam" if result == 1 else "Not Spam"}
        )

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Recent predictions")
    for item in st.session_state.history[:5]:
        icon = "🚨" if item["label"] == "Spam" else "✅"
        preview = item["message"][:80] + ("..." if len(item["message"]) > 80 else "")
        st.markdown(f"{icon} **{item['label']}** — _{preview}_")