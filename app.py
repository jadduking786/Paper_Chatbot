import os
import tempfile
from pathlib import Path

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📚 StudyMate AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your study material and ask questions from it.</div>',
    unsafe_allow_html=True
)


# =========================================================
# GROQ API KEY
# =========================================================

def get_groq_api_key():

    # Streamlit Cloud Secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Environment variable
    return os.getenv("GROQ_API_KEY")


api_key = get_groq_api_key()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Groq Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile"
        ]
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=1500,
        value=800,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=300,
        value=100,
        step=50
    )

    top_k = st.slider(
        "Relevant Chunks",
        min_value=1,
        max_value=8,
        value=4
    )

    st.divider()

    if api_key:
        st.success("Groq API key detected ✅")
    else:
        st.warning(
            "Groq API key not found.\n\n"
            "Add GROQ_API_KEY in Streamlit Secrets."
        )


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


with st.spinner("Loading embedding model..."):
    embedding_model = load_embedding_model()


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file):

    text = ""

    reader = PdfReader(file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(file):

    document = Document(file)

    text = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


def extract_txt_text(file):

    return file.read().decode(
        "utf-8",
        errors="ignore"
    )


def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        return extract_pdf_text(uploaded_file)

    elif file_name.endswith(".docx"):

        return extract_docx_text(uploaded_file)

    elif file_name.endswith(".txt"):

        return extract_txt_text(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file format."
        )


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.replace("\x00", " ")

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    text,
    chunk_size=800,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if end >= len(words):
            break

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# =========================================================
# SEARCH DOCUMENT
# =========================================================

def search_document(
    question,
    index,
    chunks,
    k=4
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        question_embedding,
        min(k, len(chunks))
    )

    results = []

    for i in indices[0]:

        if i >= 0 and i < len(chunks):

            results.append(chunks[i])

    return results


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# =========================================================
# GROQ RESPONSE
# =========================================================

def ask_groq(
    question,
    context,
    model
):

    client = get_groq_client()

    if client is None:

        return (
            "Groq API key nahi mili. "
            "Please Streamlit Secrets mein "
            "GROQ_API_KEY add karein."
        )

    system_prompt = """
You are StudyMate AI, an educational assistant.

Your job is to answer questions using the
provided study material.

IMPORTANT RULES:

1. Use the provided context as the primary source.
2. Do not invent information.
3. If the answer is not available in the uploaded
   material, clearly say:
   "This information is not available in the uploaded material."
4. Explain answers in simple language.
5. For exam preparation, use clear headings and bullet points.
6. Do not unnecessarily make answers very long.
"""

    user_prompt = f"""
STUDY MATERIAL:

{context}


QUESTION:

{question}


Answer the question using the study material above.
"""

    try:

        response = client.chat.completions.create(

            model=model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.2,

            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Groq API error: {str(e)}"


# =========================================================
# SUMMARY
# =========================================================

def generate_summary(text, model):

    client = get_groq_client()

    if client is None:

        return (
            "Groq API key nahi mili. "
            "Please GROQ_API_KEY add karein."
        )

    # Limit very large documents to avoid
    # sending excessive text to the model.
    words = text.split()

    max_words = 12000

    if len(words) > max_words:

        text_for_summary = " ".join(
            words[:max_words]
        )

        truncated = True

    else:

        text_for_summary = text
        truncated = False

    prompt = f"""
You are an expert study assistant.

Create an exam-friendly summary of the following
study material.

Include:

1. Main topic
2. Important definitions
3. Key concepts
4. Important points
5. Important dates/names if present
6. Short conclusion

Use simple English and clear bullet points.

Do not add facts that are not present in the material.

STUDY MATERIAL:

{text_for_summary}
"""

    try:

        response = client.chat.completions.create(

            model=model,

            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful academic study assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=2500
        )

        summary = response.choices[0].message.content

        if truncated:

            summary += (
                "\n\n⚠️ Note: The document was very large, "
                "so the summary used the first portion of the text."
            )

        return summary

    except Exception as e:

        return f"Summary error: {str(e)}"


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📄 Upload Study Material")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX or TXT",
    type=["pdf", "docx", "txt"]
)


# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing your document..."
        ):

            try:

                raw_text = extract_text(
                    uploaded_file
                )

                text = clean_text(
                    raw_text
                )

                if not text.strip():

                    st.error(
                        "Document se text extract nahi ho saka."
                    )

                    st.stop()

                chunks = create_chunks(
                    text,
                    chunk_size,
                    chunk_overlap
                )

                index = create_faiss_index(
                    chunks
                )

                st.session_state.document_text = text
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.messages = []

                st.success(
                    f"Document processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    f"Error processing document: {str(e)}"
                )


# =========================================================
# DOCUMENT INFORMATION
# =========================================================

if st.session_state.document_text:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Document",
            st.session_state.document_name
        )

    with col2:

        st.metric(
            "Words",
            len(
                st.session_state.document_text.split()
            )
        )

    with col3:

        st.metric(
            "Chunks",
            len(
                st.session_state.chunks
            )
        )


    # =====================================================
    # SUMMARY BUTTON
    # =====================================================

    st.divider()

    st.subheader("📝 Document Summary")

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating summary..."
        ):

            summary = generate_summary(
                st.session_state.document_text,
                model_name
            )

        st.markdown(
            '<div class="answer-box">',
            unsafe_allow_html=True
        )

        st.markdown(summary)

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =====================================================
    # CHAT
    # =====================================================

    st.divider()

    st.subheader("💬 Ask Questions")

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    question = st.chat_input(
        "Ask a question from your uploaded material..."
    )


    if question:

        # User message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Search relevant chunks

        with st.spinner(
            "Searching your study material..."
        ):

            relevant_chunks = search_document(
                question,
                st.session_state.index,
                st.session_state.chunks,
                top_k
            )

        context = "\n\n---\n\n".join(
            relevant_chunks
        )


        # Generate answer

        with st.chat_message("assistant"):

            with st.spinner(
                "Generating answer..."
            ):

                answer = ask_groq(
                    question,
                    context,
                    model_name
                )

            st.markdown(answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


else:

    st.info(
        "👆 Start by uploading your PDF, DOCX or TXT study material."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📚 StudyMate AI — RAG-based study assistant"
)
