"""
IMDb Movie Chatbot - Streamlit UI
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Optional
import time
import streamlit as st

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.tools import tool

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
log_filename = os.path.join(LOG_DIR, f"chatbot_{datetime.now().strftime('%Y%m%d')}.log")

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

logger = logging.getLogger("MovieChatbot")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.propagate = False

# ============================================================
# INITIALIZE COMPONENTS (cached for performance)
# ============================================================

@st.cache_resource
def load_dataset():
    """Load the IMDb dataset."""
    DATASET_PATH = "IMDb_Dataset (1).csv"
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset loaded: {df.shape[0]} movies")
    return df

@st.cache_resource
def create_movie_descriptions(_df):
    """Create movie descriptions for each movie."""
    def create_description(row):
        title = row['Title'] if pd.notna(row['Title']) else 'Unknown Title'
        year = int(row['Year']) if pd.notna(row['Year']) else 'Unknown Year'
        genre = row['Genre'] if pd.notna(row['Genre']) else 'Unknown Genre'
        director = row['Director'] if pd.notna(row['Director']) else 'Unknown Director'
        cast = row['Star Cast'] if pd.notna(row['Star Cast']) else 'Unknown Cast'
        rating = row['IMDb Rating'] if pd.notna(row['IMDb Rating']) else 'N/A'
        metascore = row['MetaScore'] if pd.notna(row['MetaScore']) else 'N/A'
        certificate = row['Certificates'] if pd.notna(row['Certificates']) else 'Not Rated'
        duration = int(row['Duration (minutes)']) if pd.notna(row['Duration (minutes)']) else 'Unknown'
        poster = row['Poster-src'] if pd.notna(row['Poster-src']) else ''

        description = f"""
Movie Title: {title}
Year: {year}
Genre: {genre}
Director: {director}
Star Cast: {cast}
IMDb Rating: {rating}/10
MetaScore: {metascore}
Certificate: {certificate}
Duration: {duration} minutes
Poster URL: {poster}

This is a {genre} movie titled "{title}" released in {year}.
It was directed by {director} and stars {cast}.
The film has an IMDb rating of {rating}/10 and a MetaScore of {metascore}.
It is rated {certificate} with a runtime of {duration} minutes.
""".strip()
        return description

    df = _df.copy()
    df['description'] = df.apply(create_description, axis=1)
    return df

@st.cache_resource
def create_documents(_df):
    """Convert DataFrame to LangChain documents."""
    documents = []
    for idx, row in _df.iterrows():
        metadata = {
            'title': row['Title'] if pd.notna(row['Title']) else 'Unknown',
            'year': int(row['Year']) if pd.notna(row['Year']) else 0,
            'genre': row['Genre'] if pd.notna(row['Genre']) else 'Unknown',
            'director': row['Director'] if pd.notna(row['Director']) else 'Unknown',
            'rating': float(row['IMDb Rating']) if pd.notna(row['IMDb Rating']) else 0.0,
            'certificate': row['Certificates'] if pd.notna(row['Certificates']) else 'Not Rated',
            'poster_url': row['Poster-src'] if pd.notna(row['Poster-src']) else '',
            'duration': int(row['Duration (minutes)']) if pd.notna(row['Duration (minutes)']) else 0,
        }
        doc = Document(page_content=row['description'], metadata=metadata)
        documents.append(doc)
    return documents

@st.cache_resource
def get_embeddings():
    """Initialize OpenAI Embeddings."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def get_vectorstore(_documents, _embeddings):
    """Create or load FAISS vector store."""
    VECTORSTORE_PATH = "imdb_vectorstore"

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded existing vector store with {vectorstore.index.ntotal} vectors")
    else:
        vectorstore = FAISS.from_documents(documents=_documents, embedding=_embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info(f"Created new vector store with {vectorstore.index.ntotal} vectors")

    return vectorstore

@st.cache_resource
def get_llm():
    """Initialize ChatOpenAI."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )

# ============================================================
# PROMPT TEMPLATE
# ============================================================
MOVIE_PROMPT_TEMPLATE = """You are an expert Movie Recommendation Assistant with access to the IMDb movie database.
Your role is to help users discover movies based on their preferences and queries.

Use the following movie information from our database to answer the user's question:

{context}

Guidelines:
1. Only recommend movies from the provided context - do not make up movie information
2. Provide relevant details like title, year, genre, director, cast, and ratings when available
3. If the user asks for recommendations, suggest movies that match their criteria
4. If no relevant movies are found in the context, politely say so
5. Be conversational and helpful in your responses
6. Format your response clearly with movie details

User Question: {question}

Helpful Answer:"""

# ============================================================
# CHAIN CLASSES
# ============================================================

class SimpleDocumentChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def invoke(self, input_dict):
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")

        if isinstance(context, list):
            context_text = "\n\n".join([doc.page_content for doc in context])
        else:
            context_text = str(context)

        formatted_prompt = self.prompt.format(context=context_text, question=question)
        response = self.llm.invoke(formatted_prompt)
        return response.content

class SimpleRetrievalChain:
    def __init__(self, retriever, combine_docs_chain):
        self.retriever = retriever
        self.combine_docs_chain = combine_docs_chain

    def invoke(self, input_dict):
        question = input_dict.get("input", "")
        docs = self.retriever.invoke(question)
        answer = self.combine_docs_chain.invoke({"context": docs, "input": question})
        return {"answer": answer, "context": docs, "input": question}

# ============================================================
# MOVIE CHATBOT CLASS
# ============================================================

class MovieChatbot:
    def __init__(self, retrieval_chain, vectorstore):
        self.retrieval_chain = retrieval_chain
        self.vectorstore = vectorstore
        self.request_count = 0
        self.session_start = time.time()

    def _validate_input(self, user_input: str) -> Tuple[bool, str]:
        if not user_input or not user_input.strip():
            return False, "Please enter a question or request about movies."
        if len(user_input.strip()) < 3:
            return False, "Please provide a more detailed question."
        if len(user_input) > 1000:
            return False, "Your question is too long. Please keep it under 1000 characters."
        return True, ""

    def chat(self, user_input: str) -> Tuple[str, List[Dict]]:
        """Process user input and return response with movie posters."""
        self.request_count += 1

        is_valid, error_msg = self._validate_input(user_input)
        if not is_valid:
            return f"⚠️ {error_msg}", []

        try:
            response = self.retrieval_chain.invoke({"input": user_input})

            # Extract poster information from context
            posters = []
            for doc in response.get('context', [])[:5]:
                meta = doc.metadata
                if meta.get('poster_url'):
                    posters.append({
                        'title': meta.get('title', 'Unknown'),
                        'year': meta.get('year', ''),
                        'rating': meta.get('rating', ''),
                        'genre': meta.get('genre', ''),
                        'poster_url': meta.get('poster_url', '')
                    })

            return response['answer'], posters

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return f"⚠️ An error occurred: {str(e)}", []

    def get_stats(self) -> Dict:
        return {
            "total_requests": self.request_count,
            "session_duration": time.time() - self.session_start
        }

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title="IMDb Movie Chatbot",
        page_icon="🎬",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
    .stChatMessage {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🎬 IMDb Movie Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered Movie Discovery Assistant</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot helps you discover movies from the IMDb database.

        **You can ask about:**
        - Movie recommendations by genre
        - Films by specific actors or directors
        - Highly rated movies
        - Movies from specific years
        - And much more!
        """)

        st.header("Example Queries")
        example_queries = [
            "Recommend some comedy movies",
            "Find movies starring Tom Hanks",
            "What are good documentaries?",
            "Movies directed by Christopher Nolan",
            "Highly rated biography films",
        ]

        for query in example_queries:
            if st.button(query, key=f"example_{query}"):
                st.session_state.example_query = query

        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.posters = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "posters" not in st.session_state:
        st.session_state.posters = []
    if "chatbot" not in st.session_state:
        with st.spinner("🔄 Loading chatbot components..."):
            try:
                # Load all components
                df = load_dataset()
                df = create_movie_descriptions(df)
                documents = create_documents(df)
                embeddings = get_embeddings()
                vectorstore = get_vectorstore(documents, embeddings)
                llm = get_llm()

                # Create chains
                prompt = PromptTemplate(
                    template=MOVIE_PROMPT_TEMPLATE,
                    input_variables=["context", "question"]
                )
                combine_docs_chain = SimpleDocumentChain(llm=llm, prompt=prompt)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                retrieval_chain = SimpleRetrievalChain(retriever=retriever, combine_docs_chain=combine_docs_chain)

                st.session_state.chatbot = MovieChatbot(retrieval_chain, vectorstore)
                st.success("✅ Chatbot loaded successfully!")
            except Exception as e:
                st.error(f"Error loading chatbot: {str(e)}")
                st.stop()

    # Main chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle example query from sidebar
        if "example_query" in st.session_state:
            prompt = st.session_state.example_query
            del st.session_state.example_query

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, posters = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.posters = posters
            st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask me about movies..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, posters = st.session_state.chatbot.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.posters = posters

    with col2:
        st.subheader("🖼️ Movie Posters")
        if st.session_state.posters:
            for poster in st.session_state.posters:
                if poster.get('poster_url'):
                    st.image(poster['poster_url'], caption=f"{poster['title']} ({poster['year']}) ⭐ {poster['rating']}", width=150)
        else:
            st.info("Movie posters will appear here after you ask about movies!")

        # Stats section
        st.markdown("---")
        st.subheader("📊 Session Stats")
        if "chatbot" in st.session_state:
            stats = st.session_state.chatbot.get_stats()
            st.metric("Total Queries", stats["total_requests"])
            st.metric("Session Duration", f"{stats['session_duration']:.0f}s")

if __name__ == "__main__":
    main()
