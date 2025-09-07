# import streamlit as st
# import faiss
# import pickle
# import subprocess
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os

# # -----------------------------
# # Load FAISS index & metadata
# # -----------------------------
# st.set_page_config(page_title="Multilingual RAG Chatbot", layout="wide")

# st.title("🌐 Multilingual RAG Chatbot (Offline)")
# st.write("Ask me questions in English, Hindi, Marathi, Tamil, Bengali, etc. I’ll answer using context from local documents.")

# # Load FAISS index
# index = faiss.read_index("faiss_index.index")

# # Load metadata (doc names, texts)
# with open("metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)

# # Load embeddings model (same one you used for building index)
# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # -----------------------------
# # Chat session state
# # -----------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []

# # -----------------------------
# # Query Input
# # -----------------------------
# query = st.text_input("💬 Your Question:", "")

# if st.button("Ask"):
#     if query.strip():
#         # Embed query
#         query_emb = embedder.encode([query])
#         D, I = index.search(np.array(query_emb), k=3)

#         # Collect top docs
#         context = ""
#         sources = []
#         for idx in I[0]:
#             doc_name, doc_text = metadata[idx]
#             context += f"Source: {doc_name}\n{doc_text}\n\n"
#             sources.append(doc_name)

#         # Prepare llama.cpp prompt
#         prompt = f"""You are an expert assistant. Use only the provided context to answer the question concisely.

# Context:
# {context}

# Question: {query}
# Answer:"""

#         # Run llama.cpp (replace with your local binary + model path)
#         # Example command: ./llama.exe -m models/llama-3b.gguf -p "{prompt}"
#         llama_path = r"C:\Users\Mandar Pawale\llama_cpp\llama-cli.exe"
#         model_path = r"C:\Users\Mandar Pawale\llama_cpp\models\open-llama-3b-v2-instruct.Q8_0.gguf"
#         try:
#             result = subprocess.run(
#                 [llama_path, "-m", model_path, "-p", prompt],
#                 capture_output=True,
#                 text=True
#             )
#             answer = result.stdout.strip()
#         except Exception as e:
#             answer = f"⚠️ Error running llama.cpp: {e}"

#         # Save chat history
#         st.session_state.history.append((query, answer, sources))

# # -----------------------------
# # Display Chat History
# # -----------------------------
# for q, a, s in st.session_state.history:
#     st.markdown(f"**🧑 You:** {q}")
#     st.markdown(f"**🤖 Bot:** {a}")
#     st.markdown(f"_Sources: {', '.join(s)}_")
#     st.markdown("---")




# import streamlit as st
# import os
# import re
# import subprocess

# # 📂 Folders & model paths
# DOCS_FOLDER = "C:/Users/Mandar Pawale/Desktop/RagChatBot/data"
# LLAMA_CPP_PATH = "C:/Users/Mandar Pawale/llama_cpp/llama-cli.exe"
# MODEL_PATH = "C:/Users/Mandar Pawale/llama_cpp/models/open-llama-3b-v2-instruct.Q8_0.gguf"

# # Load docs into dictionary
# def load_docs():
#     docs = {}
#     for file in os.listdir(DOCS_FOLDER):
#         if file.endswith(".txt"):
#             with open(os.path.join(DOCS_FOLDER, file), "r", encoding="utf-8") as f:
#                 docs[file] = f.read()
#     return docs

# DOCS = load_docs()

# # Simple language detection & file mapping
# LANG_MAP = {
#     "en": "doc_en.txt",
#     "hi": "doc_hi.txt",
#     "mr": "doc_mr.txt",
#     "ta": "doc_ta.txt",
#     "te": "doc_te.txt",
#     "bn": "doc_bn.txt",
#     "gu": "doc_gu.txt",
#     "kn": "doc_kn.txt",
#     "ml": "doc_ml.txt",
#     "pa": "doc_pa.txt",
# }

# def detect_language(question: str) -> str:
#     if re.search(r"[\u0900-\u097F]", question):  # Hindi/Marathi (Devanagari)
#         if "क्या" in question or "है" in question:
#             return "hi"
#         else:
#             return "mr"
#     elif re.search(r"[\u0B80-\u0BFF]", question):  # Tamil
#         return "ta"
#     elif re.search(r"[\u0C00-\u0C7F]", question):  # Telugu
#         return "te"
#     elif re.search(r"[\u0980-\u09FF]", question):  # Bengali
#         return "bn"
#     elif re.search(r"[\u0A80-\u0AFF]", question):  # Gujarati
#         return "gu"
#     elif re.search(r"[\u0C80-\u0CFF]", question):  # Kannada
#         return "kn"
#     elif re.search(r"[\u0D00-\u0D7F]", question):  # Malayalam
#         return "ml"
#     elif re.search(r"[\u0A00-\u0A7F]", question):  # Punjabi
#         return "pa"
#     else:
#         return "en"

# # Build contextual prompt
# def build_prompt(question: str) -> str:
#     lang = detect_language(question)
#     filename = LANG_MAP.get(lang, "doc_en.txt")
#     context = DOCS.get(filename, "")
#     return f"""

# Context:
# Source: {filename} {context}

# Question: {question}
# Answer:
# """

# # Call llama.cpp
# def ask_llama(user_question: str) -> str:
#     prompt = build_prompt(user_question)

#     result = subprocess.run(
#         [
#             LLAMA_CPP_PATH,
#             "-m", MODEL_PATH,
#             "-p", prompt,
#             "--n-predict", "256",
#             "--temp", "0.7"
#         ],
#         capture_output=True,
#         text=True
#     )

#     return result.stdout.strip()

# # ================= Streamlit UI =================
# st.set_page_config(page_title="RAG Multilingual Chatbot", page_icon="🤖", layout="centered")

# st.title("🤖 Multilingual RAG Chatbot")
# st.markdown("Ask your question in **English, Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Punjabi**")

# # Chat input
# user_q = st.text_input("Ask me anything:")

# if user_q:
#     with st.spinner("Thinking..."):
#         answer = ask_llama(user_q)
#     st.markdown(f"**🤖 Bot:** {answer}")



import streamlit as st
import os
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# --- Configuration ---
MODEL_PATH = r"Models\mistral-7b-instruct-v0.2.Q5_K_M.gguf"
DOCS_FOLDER = r"C:/Users/Mandar Pawale/Desktop/RagChatBot/data"
# better multilingual embeddings for Indian languages
EMBEDDING_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# --- RAG Setup (cached for efficiency) ---
@st.cache_resource
def load_llm():
    """Load the LLM once and cache it."""
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,   # use GPU acceleration if available
            n_ctx=4096,        # larger context window for Mistral
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"❌ Failed to load LLM: {e}. Please check '{MODEL_PATH}' exists and is a valid GGUF file.")
        st.stop()

@st.cache_resource
def load_embedding_model():
    """Load the Sentence-Transformer model and cache it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def index_documents():
    """
    Loads, chunks, and indexes all documents using FAISS.
    Returns the FAISS index and a list of document chunks.
    """
    chunks = []
    chunk_meta = []  # To store metadata like language for each chunk
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            try:
                lang = filename.split('_')[1].split('.')[0]
            except Exception:
                lang = "unknown"
            file_path = os.path.join(DOCS_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple chunking by paragraph
                for paragraph in content.split('\n\n'):
                    if paragraph.strip():
                        chunks.append(paragraph)
                        chunk_meta.append({"filename": filename, "language": lang})

    if not chunks:
        st.error("⚠️ No documents found in the 'data' folder. Please add some .txt files.")
        st.stop()

    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks, chunk_meta

# --- Initialize Models ---
if "llm" not in st.session_state:
    st.session_state.llm = load_llm()
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = load_embedding_model()
if "faiss_index" not in st.session_state or "document_chunks" not in st.session_state:
    with st.spinner("📑 Indexing documents... this may take a moment."):
        st.session_state.faiss_index, st.session_state.document_chunks, st.session_state.chunk_meta = index_documents()

# --- Retrieval ---
def retrieve_context(question: str, top_k: int = 2) -> str:
    """Finds and returns the most relevant document chunks."""
    question_embedding = st.session_state.embedding_model.encode([question]).astype("float32")

    # Search the FAISS index
    distances, indices = st.session_state.faiss_index.search(question_embedding, top_k)

    # Get the relevant chunks
    retrieved_chunks = [st.session_state.document_chunks[i] for i in indices[0]]

    return "\n\n".join(retrieved_chunks)

def build_prompt(question: str) -> str:
    """Builds a contextual prompt with retrieved information."""
    context = retrieve_context(question)

    return f"""
You are a multilingual assistant (Mistral 7B).
The user may ask questions in English or an Indian language.
Important instructions:
1. DO NOT translate the question into English before understanding it.
2. Directly interpret the question in its original language.
3. Answer strictly using the context provided.
4. If the context does not contain the answer, reply only: "I don't know."
5. Respond in the same language as the user's question.


Context:
{context}

User question: {question}
Answer:
"""

# --- Query LLM ---
def ask_llama(user_question: str) -> str:
    """Generates a response from the LLM."""
    prompt = build_prompt(user_question)

    output = st.session_state.llm(
        prompt,
        max_tokens=512,
        stop=["\nQuestion:", "User:", "OPTIONS:", "Answer:"],
        temperature=0.7,
        echo=False
    )

    response_text = output['choices'][0]['text'].strip()

    # Clean up unwanted prefixes
    unwanted_prefixes = ["Answer:", "OPTIONS:", "[1].", "[2].", "[3].", "[4]."]
    for prefix in unwanted_prefixes:
        if response_text.startswith(prefix):
            response_text = response_text[len(prefix):].strip()

    if not response_text:
        return "I'm sorry, I couldn't find an answer in the provided context."

    return response_text

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Multilingual Chatbot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans&family=Noto+Sans-Bengali&family=Noto+Sans-Devanagari&family=Noto+Sans-Gujarati&family=Noto+Sans-Gurmukhi&family=Noto+Sans-Kannada&family=Noto+Sans-Malayalam&family=Noto+Sans-Tamil&family=Noto+Sans-Telugu&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans Devanagari', 'Noto Sans Bengali',
                     'Noto Sans Tamil', 'Noto Sans Telugu',
                     'Noto Sans Gujarati', 'Noto Sans Kannada',
                     'Noto Sans Malayalam', 'Noto Sans Gurmukhi',
                     'Noto Sans', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Multilingual RAG Chatbot (Mistral-7B)")
st.markdown("Ask your question in Marathi, Telugu, Hindi, or English, based on the provided context files.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_q := st.chat_input("Ask me anything:"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Show retrieved context (debugging)
    with st.expander("🔎 Retrieved Context"):
        st.write(retrieve_context(user_q))

    # Get assistant answer
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            answer = ask_llama(user_q)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
