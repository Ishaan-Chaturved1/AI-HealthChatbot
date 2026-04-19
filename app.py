"""
AI-Healthcare Chatbot — RAG + Groq (Open Source LLM)
Streamlit UI Version with PDF RAG Pipeline
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from difflib import get_close_matches
from groq import Groq

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2942 50%, #0f1923 100%);
    color: #e8edf5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 25, 35, 0.95) !important;
    border-right: 1px solid rgba(100, 180, 255, 0.15);
}
[data-testid="stSidebar"] * { color: #c9d8ea !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(100,180,255,0.2) !important;
    color: #e8edf5 !important;
    border-radius: 8px;
}

/* Main header */
.health-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.health-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #64b4ff, #a0d4ff, #64b4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.health-header p {
    color: #7a9bbf;
    font-size: 1rem;
    font-weight: 300;
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, rgba(30,60,100,0.6), rgba(20,40,70,0.8));
    border: 1px solid rgba(100,180,255,0.25);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.pred-card h3 {
    font-family: 'DM Serif Display', serif;
    color: #64b4ff;
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
}
.pred-card .disease-name {
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8edf5;
    letter-spacing: -0.3px;
}
.confidence-badge {
    display: inline-block;
    background: rgba(100,180,255,0.15);
    border: 1px solid rgba(100,180,255,0.4);
    color: #64b4ff;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.4rem;
}
.rag-badge {
    display: inline-block;
    background: rgba(100,255,180,0.1);
    border: 1px solid rgba(100,255,180,0.3);
    color: #64ffb4;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-left: 0.5rem;
}
.symptom-tag {
    display: inline-block;
    background: rgba(100,180,255,0.1);
    border: 1px solid rgba(100,180,255,0.2);
    color: #a0c8f0;
    padding: 0.2rem 0.7rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin: 0.2rem;
}

/* Chat messages */
.chat-container {
    max-height: 480px;
    overflow-y: auto;
    padding: 0.5rem 0;
}
.chat-msg {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    align-items: flex-start;
}
.chat-msg.user { flex-direction: row-reverse; }

.chat-bubble {
    max-width: 78%;
    padding: 0.9rem 1.2rem;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.6;
}
.chat-msg.user .chat-bubble {
    background: linear-gradient(135deg, #1e5fa0, #1a4a80);
    border: 1px solid rgba(100,180,255,0.3);
    color: #e8edf5;
    border-top-right-radius: 4px;
}
.chat-msg.assistant .chat-bubble {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #d4e2f0;
    border-top-left-radius: 4px;
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, #1e5fa0, #0d3060); border: 1px solid rgba(100,180,255,0.3); }
.avatar.user { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); }

/* Input area */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(100,180,255,0.2) !important;
    color: #e8edf5 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(100,180,255,0.5) !important;
    box-shadow: 0 0 0 2px rgba(100,180,255,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e5fa0, #1a4a80) !important;
    color: #e8edf5 !important;
    border: 1px solid rgba(100,180,255,0.3) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2570b8, #1e5fa0) !important;
    border-color: rgba(100,180,255,0.5) !important;
    transform: translateY(-1px);
}

/* Divider */
hr { border-color: rgba(100,180,255,0.1) !important; }

/* Section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a7a9b;
    margin-bottom: 0.8rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100,180,255,0.2); border-radius: 10px; }

/* Warning/info boxes */
.stAlert { border-radius: 12px !important; }

/* RAG status box */
.rag-status {
    background: rgba(100,255,180,0.05);
    border: 1px solid rgba(100,255,180,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #64ffb4;
    margin-bottom: 0.5rem;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PATHS & DATA LOADING (cached)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Training ML model…")
def load_model_and_data():
    TRAIN_PATH = BASE_DIR / "Data" / "Training.csv"
    TEST_PATH  = BASE_DIR / "Data" / "Testing.csv"

    training = pd.read_csv(TRAIN_PATH)
    testing  = pd.read_csv(TEST_PATH)

    for df in (training, testing):
        df.columns = df.columns.str.replace(r"\.\d+$", "", regex=True)

    training = training.loc[:, ~training.columns.duplicated()]
    testing  = testing.loc[:,  ~testing.columns.duplicated()]

    X = training.iloc[:, :-1]
    y = training["prognosis"]

    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.33, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}
    return clf, encoder, symptoms_dict, X.columns.tolist()


@st.cache_data(show_spinner=False)
def load_knowledge_base():
    DESC_PATH       = BASE_DIR / "MasterData" / "symptom_Description.csv"
    SEVERITY_PATH   = BASE_DIR / "MasterData" / "Symptom_severity.csv"
    PRECAUTION_PATH = BASE_DIR / "MasterData" / "symptom_precaution.csv"

    description_dict = {}
    severity_dict    = {}
    precaution_dict  = {}

    df = pd.read_csv(DESC_PATH, header=None)
    for _, row in df.iterrows():
        description_dict[str(row[0])] = str(row[1])

    df = pd.read_csv(SEVERITY_PATH, header=None)
    for _, row in df.iterrows():
        try:
            severity_dict[str(row[0])] = int(row[1])
        except Exception:
            pass

    df = pd.read_csv(PRECAUTION_PATH, header=None)
    for _, row in df.iterrows():
        precaution_dict[str(row[0])] = list(row[1:5])

    return description_dict, severity_dict, precaution_dict


# ─────────────────────────────────────────────
# RAG PIPELINE — loads pre-built FAISS index
# saved from rag.ipynb (no re-ingestion needed)
# ─────────────────────────────────────────────
#
# ONE-TIME SETUP: in rag.ipynb, after building
# your vectorstore, run this cell once:
#
#   vectorstore.save_local("faiss_index")
#
# That writes faiss_index/ next to your notebook.
# app.py will load it instantly every time after.
# ─────────────────────────────────────────────

FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index")

@st.cache_resource(show_spinner="Loading RAG index…")
def load_rag_pipeline(index_path: str):
    """
    Loads the FAISS index + embeddings that were built and saved
    in rag.ipynb. Takes ~1-2 seconds (no re-ingestion, no re-embedding).
    Returns (retriever, num_vectors) or (None, 0) on failure.
    """
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        if not Path(index_path).exists():
            return None, 0

        # Must use the same embedding model as in the notebook
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # loads from local HuggingFace cache — no re-download if
            # already on disk from when you ran the notebook
        )

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
        num_vectors = vectorstore.index.ntotal

        return retriever, num_vectors

    except Exception as e:
        st.warning(f"Could not load RAG index: {e}")
        return None, 0


def retrieve_rag_context(retriever, query: str) -> str:
    """Run the retriever and return formatted context string."""
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)
    except Exception:
        return ""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "feaver": "fever",
    "cold": "chills",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input: str, all_symptoms: list) -> list:
    extracted = []
    text = user_input.lower()

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(
            word,
            [s.replace("_", " ") for s in all_symptoms],
            n=1, cutoff=0.8,
        )
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))


def predict_disease(symptoms_list, clf, encoder, symptoms_dict):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    probs      = clf.predict_proba([input_vector])[0]
    idx        = np.argmax(probs)
    disease    = encoder.inverse_transform([idx])[0]
    confidence = round(probs[idx] * 100, 2)
    return disease, confidence


def retrieve_csv_context(disease, description_dict, precaution_dict):
    """Build context from the CSV knowledge base."""
    sections = []
    if disease in description_dict:
        sections.append(f"DISEASE DESCRIPTION:\n{description_dict[disease]}")
    if disease in precaution_dict:
        sections.append(
            "PRECAUTIONS:\n" + "\n".join(f"- {p}" for p in precaution_dict[disease])
        )
    return "\n\n".join(sections)


MAX_CONTEXT_CHARS  = 1200   # max chars for RAG/CSV context block
MAX_HISTORY_TURNS  = 4      # keep last N user+assistant pairs in conv history
MAX_RESPONSE_TOKENS = 512   # keep responses short to stay under TPM

SYSTEM_PROMPT = """You are a concise AI health assistant.
Use the provided medical context only. Be brief and safe.
Always suggest consulting a doctor. Max 3 short paragraphs."""


def _trim_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate context to stay within token budget."""
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _trim_history(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """Keep only the first message (initial diagnosis prompt) + last N turns."""
    if len(messages) <= 1 + max_turns * 2:
        return messages
    return [messages[0]] + messages[-(max_turns * 2):]


def ask_groq(client, messages):
    from groq import APIConnectionError, APIStatusError, RateLimitError
    trimmed = _trim_history(messages)
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *trimmed,
            ],
            temperature=0.7,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        return response.choices[0].message.content
    except APIConnectionError:
        return (
            "⚠️ **Connection error** — couldn't reach Groq's API.\n\n"
            "Please check:\n"
            "- Your internet connection\n"
            "- Whether a VPN or firewall is blocking outbound HTTPS\n"
            "- Groq service status at https://status.groq.com\n\n"
            "Then try again."
        )
    except RateLimitError as e:
        return (
            f"⚠️ **Rate limit exceeded** — too many tokens per minute on the free tier.\n\n"
            "Wait 10–15 seconds and try again, or upgrade at https://console.groq.com/settings/billing"
        )
    except APIStatusError as e:
        return f"⚠️ **Groq API error {e.status_code}** — {e.message}"
    except Exception as e:
        return f"⚠️ **Unexpected error** — {str(e)}"


def render_chat(conversation):
    for msg in conversation:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            continue
        if role == "user":
            st.markdown(f"""
            <div class="chat-msg user">
                <div class="avatar user">👤</div>
                <div class="chat-bubble">{content}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-msg assistant">
                <div class="avatar bot">🩺</div>
                <div class="chat-bubble">{content}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "conversation": [],
        "diagnosed": False,
        "disease": None,
        "confidence": None,
        "detected_symptoms": [],
        "groq_client": None,
        "rag_retriever": None,
        "rag_chunks": 0,
        "rag_active": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
clf, encoder, symptoms_dict, all_symptoms = load_model_and_data()
description_dict, severity_dict, precaution_dict = load_knowledge_base()

# Auto-load pre-built FAISS index saved from rag.ipynb
if not st.session_state.rag_active and Path(FAISS_INDEX_PATH).exists():
    retriever, n_vecs = load_rag_pipeline(FAISS_INDEX_PATH)
    if retriever:
        st.session_state.rag_retriever = retriever
        st.session_state.rag_chunks    = n_vecs
        st.session_state.rag_active    = True


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Profile")
    st.markdown("---")

    name   = st.text_input("Full Name", placeholder="Your name")
    age    = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("---")
    st.markdown("## 🔑 API Key")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        value=os.environ.get("GROQ_API_KEY", ""),
    )
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
        st.success("API key set ✓", icon="✅")

    st.markdown("---")

    # ── RAG Settings ──────────────────────────────
    st.markdown("## 📚 RAG — PDF Knowledge Base")
    st.caption("Save your FAISS index from the notebook once, then it loads here instantly.")

    if st.button("🔄 Reload RAG Index", use_container_width=True):
        load_rag_pipeline.clear()
        with st.spinner("Loading RAG index…"):
            retriever, n_vecs = load_rag_pipeline(FAISS_INDEX_PATH)
            st.session_state.rag_retriever = retriever
            st.session_state.rag_chunks    = n_vecs
            st.session_state.rag_active    = retriever is not None
        if st.session_state.rag_active:
            st.success(f"RAG loaded — {n_vecs} vectors ✓", icon="✅")
        else:
            st.error("faiss_index/ not found. Run vectorstore.save_local(\'faiss_index\') in your notebook first.")

    if st.session_state.rag_active:
        st.markdown(
            f'<div class="rag-status">📗 RAG active &nbsp;·&nbsp; {st.session_state.rag_chunks} vectors loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("⚠️ faiss_index/ not found. Save it from rag.ipynb first.")

    st.markdown("---")

    if st.session_state.diagnosed:
        st.markdown("## 📋 Session Summary")
        st.markdown(f"**Disease:** {st.session_state.disease}")
        st.markdown(f"**Confidence:** {st.session_state.confidence}%")
        st.markdown(f"**Symptoms detected:** {len(st.session_state.detected_symptoms)}")
        rag_label = "✅ Yes" if st.session_state.rag_active else "❌ No"
        st.markdown(f"**RAG used:** {rag_label}")
        st.markdown("---")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        for key in ["conversation", "diagnosed", "disease", "confidence", "detected_symptoms"]:
            st.session_state[key] = [] if key in ["conversation", "detected_symptoms"] else (False if key == "diagnosed" else None)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("⚠️ This is an AI assistant, not a substitute for professional medical advice.")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="health-header">
    <h1>AI Health Assistant</h1>
    <p>Describe your symptoms and get an intelligent health assessment powered by AI + RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Diagnosis Section ──────────────────────────────────────
if not st.session_state.diagnosed:
    st.markdown('<p class="section-label">Step 1 — Symptom Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        symptom_input = st.text_area(
            "Describe your symptoms",
            placeholder="e.g. I have a high fever, headache, and body aches since yesterday...",
            height=110,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    if analyze_btn:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not symptom_input.strip():
            st.warning("Please describe your symptoms first.")
        else:
            with st.spinner("Analyzing symptoms…"):
                symptoms = extract_symptoms(symptom_input, all_symptoms)

            if not symptoms:
                st.error("No recognizable symptoms detected. Please try describing differently.")
            else:
                disease, confidence = predict_disease(symptoms, clf, encoder, symptoms_dict)

                # ── Build combined context: CSV KB + RAG PDFs ─────────
                csv_context = retrieve_csv_context(disease, description_dict, precaution_dict)

                rag_context = ""
                if st.session_state.rag_active and st.session_state.rag_retriever:
                    rag_query   = f"{disease} symptoms treatment precaution"
                    rag_context = retrieve_rag_context(
                        st.session_state.rag_retriever, rag_query
                    )

                # Trim each context source to stay within TPM limits
                csv_trimmed = _trim_context(csv_context, 600)
                rag_trimmed = _trim_context(rag_context, 600) if rag_context else ""

                combined_context = csv_trimmed
                if rag_trimmed:
                    combined_context += f"\n\nFROM MEDICAL LITERATURE:\n{rag_trimmed}"

                st.session_state.diagnosed         = True
                st.session_state.disease           = disease
                st.session_state.confidence        = confidence
                st.session_state.detected_symptoms = symptoms

                rag_note = " (RAG-enhanced)" if rag_trimmed else ""
                initial_prompt = f"""Patient: {name or 'Unknown'}, {age}y, {gender}
Symptoms: {', '.join(symptoms)}
Prediction: {disease} ({confidence}%){rag_note}

Context:
{combined_context}

Give: 1) brief explanation 2) key precautions 3) when to seek urgent care. Be concise."""
                st.session_state.conversation = [{"role": "user", "content": initial_prompt}]

                with st.spinner("Generating assessment…"):
                    reply = ask_groq(st.session_state.groq_client, st.session_state.conversation)

                st.session_state.conversation.append({"role": "assistant", "content": reply})
                st.rerun()

# ── Results & Chat Section ─────────────────────────────────
if st.session_state.diagnosed:

    # Prediction card
    rag_badge = '<span class="rag-badge">📚 RAG enhanced</span>' if st.session_state.rag_active else ""
    symptoms_html = "".join(
        f'<span class="symptom-tag">{s.replace("_", " ")}</span>'
        for s in st.session_state.detected_symptoms
    )
    st.markdown(f"""
    <div class="pred-card">
        <h3>🔬 Diagnosis Result</h3>
        <div class="disease-name">{st.session_state.disease} {rag_badge}</div>
        <div class="confidence-badge">⚡ {st.session_state.confidence}% confidence</div>
        <div style="margin-top:1rem;">
            <div class="section-label" style="margin-bottom:0.5rem;">Detected Symptoms</div>
            {symptoms_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Conversation</p>', unsafe_allow_html=True)

    # Chat display (skip the long initial user prompt)
    display_conv = [
        msg for i, msg in enumerate(st.session_state.conversation) if i != 0
    ]
    with st.container():
        render_chat(display_conv)

    st.markdown("---")

    # Follow-up input — RAG retrieval also enriches follow-up answers
    col1, col2 = st.columns([5, 1])
    with col1:
        follow_up = st.text_input(
            "Ask a follow-up question",
            placeholder="e.g. What foods should I avoid? How long does recovery take?",
            label_visibility="collapsed",
            key="follow_up_input",
        )
    with col2:
        send_btn = st.button("Send ➤", use_container_width=True)

    if send_btn and follow_up.strip():
        # Augment follow-up with RAG context (trimmed to stay within TPM)
        user_message = follow_up
        if st.session_state.rag_active and st.session_state.rag_retriever:
            rag_ctx = retrieve_rag_context(st.session_state.rag_retriever, follow_up)
            if rag_ctx:
                user_message = (
                    f"{follow_up}\n\n"
                    f"[Context: {_trim_context(rag_ctx, 400)}]"
                )

        st.session_state.conversation.append({"role": "user", "content": user_message})
        with st.spinner("Thinking…"):
            reply = ask_groq(st.session_state.groq_client, st.session_state.conversation)
        # Store clean follow-up for display, but keep augmented version in conv history
        st.session_state.conversation[-1]["content"] = follow_up  # display clean text
        st.session_state.conversation.append({"role": "assistant", "content": reply})

        # Re-inject augmented version for LLM history continuity
        st.session_state.conversation[-2]["content"] = user_message
        st.rerun()