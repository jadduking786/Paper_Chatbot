import os

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 StudyMate AI")

st.write(
    "Upload your study material and ask questions "
    "from your document."
)


# ============================================================
# GROQ API KEY
# ============================================================

def get_api_key():

    # Streamlit Cloud Secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Local environment variable
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_api_key()


# ============================================================
# MODEL
# ============================================================

# OpenAI GPT-OSS model running through Groq
MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.info(
        f"AI Model:\n\n"
        f"`{MODEL_NAME}`"
    )

    chunk_size = st.slider(
        "Chunk Size",
        300,
        1200,
        700,
        100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        50,
        250,
        100,
        50
    )

    top_k = st.slider(
        "Relevant Chunks",
        1,
        8,
        4
    )

    st.divider()

    if GROQ_API_KEY:

        st.success("Groq API Key detected ✅")

    else:

        st.error(
            "GROQ_API_KEY not found."
        )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner("Loading AI embedding model..."):

    embedding_model = load_embedding_model()


# ============================================================
# SESSION STATE
# ============================================================

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text
            text += "\n"

    return text


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file):

    document = Document(file)

    text = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text.append(
                paragraph.text.strip()
            )

    return "\n".join(text)


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file):

    return file.read().decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# GENERAL TEXT EXTRACTION
# ============================================================

def extract_text(file):

    file_name = file.name.lower()

    if file_name.endswith(".pdf"):

        return extract_pdf(file)

    elif file_name.endswith(".docx"):

        return extract_docx(file)

    elif file_name.endswith(".txt"):

        return extract_txt(file)

    else:

        raise ValueError(
            "Only PDF, DOCX and TXT files are supported."
        )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:

            lines.append(line)

    return "\n".join(lines)


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def create_chunks(
    text,
    chunk_size,
    overlap
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

        if end >= len(words):

            break

    return chunks


# ============================================================
# CREATE EMBEDDINGS + FAISS
# ============================================================

def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
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


# ============================================================
# SEARCH RELEVANT CHUNKS
# ============================================================

def search_document(
    question,
    index,
    chunks,
    top_k
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
        min(top_k, len(chunks))
    )

    results = []

    for index_number in indices[0]:

        if index_number >= 0:

            results.append(
                chunks[index_number]
            )

    return results


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    if not GROQ_API_KEY:

        return None

    return Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# ASK GPT-OSS
# ============================================================

def ask_ai(
    question,
    context
):

    client = get_groq_client()

    if client is None:

        return (
            "❌ GROQ_API_KEY nahi mili. "
            "Please Streamlit Cloud ke Secrets mein "
            "GROQ_API_KEY add karein."
        )

    system_prompt = """
You are StudyMate AI, an educational assistant.

Your task is to answer questions from the user's
uploaded study material.

Rules:

1. Use the provided document context.
2. Do not invent information.
3. If the answer is not present in the document,
   say clearly:
   "This information is not available in the uploaded material."
4. Explain difficult concepts in simple language.
5. For exam preparation, use headings and bullet points.
6. Keep the answer relevant to the question.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


QUESTION:

{question}


Answer the question using the document context.
"""

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

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

            max_tokens=2000
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            f"❌ Groq API Error:\n\n{str(e)}"
        )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(text):

    client = get_groq_client()

    if client is None:

        return (
            "❌ GROQ_API_KEY nahi mili."
        )

    # Keep summary input manageable
    words = text.split()

    maximum_words = 12000

    if len(words) > maximum_words:

        summary_text = " ".join(
            words[:maximum_words]
        )

    else:

        summary_text = text

    prompt = f"""
You are an expert study assistant.

Create an exam-friendly summary of the following
study material.

Include:

1. Main topic
2. Important definitions
3. Key concepts
4. Important points
5. Important names or dates if available
6. Short conclusion

Use simple English.

Use headings and bullet points.

Do not add information that is not present
in the uploaded material.

STUDY MATERIAL:

{summary_text}
"""

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful academic assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=3000
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            f"❌ Summary Error:\n\n{str(e)}"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📄 Upload Your Study Material")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX or TXT",
    type=[
        "pdf",
        "docx",
        "txt"
    ]
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    # Process only if a new document is uploaded
    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing document..."
        ):

            try:

                # Extract text
                raw_text = extract_text(
                    uploaded_file
                )

                # Clean text
                text = clean_text(
                    raw_text
                )

                if not text.strip():

                    st.error(
                        "❌ Document mein readable text nahi mila."
                    )

                    st.stop()

                # Create chunks
                chunks = create_chunks(
                    text,
                    chunk_size,
                    chunk_overlap
                )

                # Create FAISS database
                index = create_vector_database(
                    chunks
                )

                # Save everything
                st.session_state.document_text = text

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.chunks = chunks

                st.session_state.index = index

                st.session_state.messages = []

                st.success(
                    "✅ Document successfully processed!"
                )

            except Exception as e:

                st.error(
                    f"❌ Document processing error: {str(e)}"
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.document_text:

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Document",
            st.session_state.document_name
        )

    with col2:

        st.metric(
            "📝 Words",
            len(
                st.session_state.document_text.split()
            )
        )

    with col3:

        st.metric(
            "🧩 Chunks",
            len(
                st.session_state.chunks
            )
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📝 AI Summary")

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating summary with GPT-OSS..."
        ):

            summary = generate_summary(
                st.session_state.document_text
            )

        st.markdown(summary)


    # ========================================================
    # CHAT
    # ========================================================

    st.divider()

    st.subheader("💬 Ask Questions From Your Document")

    # Previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    question = st.chat_input(
        "Example: What are the key features of this topic?"
    )


    if question:

        # Add user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Search document
        with st.spinner(
            "Searching document..."
        ):

            relevant_chunks = search_document(
                question,
                st.session_state.index,
                st.session_state.chunks,
                top_k
            )


        # Create context
        context = "\n\n---\n\n".join(
            relevant_chunks
        )


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "GPT-OSS is generating answer..."
            ):

                answer = ask_ai(
                    question,
                    context
                )

            st.markdown(answer)


        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


else:

    st.info(
        "👆 Upload your study material to start."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📚 StudyMate AI | RAG + FAISS + OpenAI GPT-OSS via Groq"
)
