"""
IMDb Movie Chatbot - Streamlit UI with Multi-Agent System
Features: Multiple specialized agents, trending movies, interactive quiz, rich UI
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Optional, Any
import time
import random
import streamlit as st

# Load environment variables from .env file (for local development)
load_dotenv()

# ============================================================
# API KEY CONFIGURATION (supports both local and Streamlit Cloud)
# ============================================================
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment variables."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    return None

# Set the API key
api_key = get_openai_api_key()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("OpenAI API key not found! Please set it in Streamlit secrets or .env file.")
    st.info("""
    **For Streamlit Cloud:** Go to Settings > Secrets > Add: `OPENAI_API_KEY = "your-key"`

    **For local:** Create `.env` file with: `OPENAI_API_KEY=your-key`
    """)
    st.stop()

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("MovieChatbot")
logger.setLevel(logging.DEBUG)

# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_resource
def load_dataset():
    """Load the IMDb dataset."""
    DATASET_PATH = "IMDb_Dataset (1).csv"
    df = pd.read_csv(DATASET_PATH)
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

        return f"""Movie Title: {title}
Year: {year}
Genre: {genre}
Director: {director}
Star Cast: {cast}
IMDb Rating: {rating}/10
MetaScore: {metascore}
Certificate: {certificate}
Duration: {duration} minutes
Poster URL: {poster}

This is a {genre} movie titled "{title}" released in {year}, directed by {director} and starring {cast}. It has an IMDb rating of {rating}/10."""

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
            'cast': row['Star Cast'] if pd.notna(row['Star Cast']) else 'Unknown',
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
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=_documents, embedding=_embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

@st.cache_resource
def get_llm():
    """Initialize ChatOpenAI."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000)

# ============================================================
# MULTI-AGENT SYSTEM
# ============================================================

class QueryClassifier:
    """Classifies user queries to route to the appropriate agent."""

    QUERY_TYPES = {
        "genre_search": ["genre", "comedy", "action", "drama", "horror", "thriller", "romance", "documentary", "biography", "adventure", "sci-fi", "animation", "fantasy"],
        "actor_search": ["actor", "actress", "starring", "star", "played by", "acted", "cast"],
        "director_search": ["director", "directed", "filmmaker", "made by"],
        "rating_search": ["rating", "rated", "best", "top", "highest", "score", "imdb"],
        "year_search": ["year", "released", "came out", "from 19", "from 20", "in 19", "in 20"],
        "comparison": ["compare", "versus", "vs", "difference", "better", "which one"],
        "recommendation": ["recommend", "suggest", "similar", "like", "should i watch", "what to watch"],
        "specific_movie": ["tell me about", "what is", "details about", "info about", "information"],
    }

    def classify(self, query: str) -> str:
        """Classify the query type."""
        query_lower = query.lower()

        for query_type, keywords in self.QUERY_TYPES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return query_type

        return "general_search"


class BaseAgent:
    """Base class for all specialized agents."""

    def __init__(self, name: str, llm, vectorstore, df):
        self.name = name
        self.llm = llm
        self.vectorstore = vectorstore
        self.df = df
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def get_context(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents."""
        return self.retriever.invoke(query)

    def format_context(self, docs: List[Document]) -> str:
        """Format documents as context string."""
        return "\n\n".join([doc.page_content for doc in docs])

    def invoke(self, query: str) -> Dict[str, Any]:
        """Process the query - to be implemented by subclasses."""
        raise NotImplementedError


class GenreRecommendationAgent(BaseAgent):
    """Agent specialized in genre-based recommendations."""

    def __init__(self, llm, vectorstore, df):
        super().__init__("Genre Recommendation Agent", llm, vectorstore, df)

    def invoke(self, query: str) -> Dict[str, Any]:
        docs = self.get_context(query)
        context = self.format_context(docs)

        prompt = f"""You are a movie genre expert. Based on the user's query about movie genres, provide recommendations.

Context (movies from database):
{context}

User Query: {query}

Provide genre-specific recommendations with movie details (title, year, rating, brief description).
Format nicely with bullet points or numbered lists."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        posters = [doc.metadata for doc in docs if doc.metadata.get('poster_url')]
        return {"answer": response.content, "posters": posters[:5], "agent": self.name}


class ActorDirectorAgent(BaseAgent):
    """Agent specialized in actor and director searches."""

    def __init__(self, llm, vectorstore, df):
        super().__init__("Actor/Director Search Agent", llm, vectorstore, df)

    def invoke(self, query: str) -> Dict[str, Any]:
        docs = self.get_context(query, k=7)
        context = self.format_context(docs)

        prompt = f"""You are a movie industry expert specializing in actors and directors.

Context (movies from database):
{context}

User Query: {query}

Provide information about the actor/director's filmography from the context.
Include movie titles, years, genres, and ratings. Highlight notable works."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        posters = [doc.metadata for doc in docs if doc.metadata.get('poster_url')]
        return {"answer": response.content, "posters": posters[:5], "agent": self.name}


class RatingFilterAgent(BaseAgent):
    """Agent specialized in rating-based searches."""

    def __init__(self, llm, vectorstore, df):
        super().__init__("Rating Filter Agent", llm, vectorstore, df)

    def invoke(self, query: str) -> Dict[str, Any]:
        # Filter high-rated movies from dataframe
        high_rated = self.df[self.df['IMDb Rating'] >= 7.5].nlargest(10, 'IMDb Rating')

        docs = self.get_context(query)
        context = self.format_context(docs)

        # Add top rated info
        top_movies = "\n".join([f"- {row['Title']} ({row['Year']}) - Rating: {row['IMDb Rating']}"
                                for _, row in high_rated.head(5).iterrows()])

        prompt = f"""You are a movie critic expert focused on highly-rated films.

Top Rated Movies in Database:
{top_movies}

Additional Context:
{context}

User Query: {query}

Recommend movies based on ratings. Explain why these movies are highly rated."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        posters = [doc.metadata for doc in docs if doc.metadata.get('poster_url')]
        return {"answer": response.content, "posters": posters[:5], "agent": self.name}


class MovieComparisonAgent(BaseAgent):
    """Agent specialized in comparing movies."""

    def __init__(self, llm, vectorstore, df):
        super().__init__("Movie Comparison Agent", llm, vectorstore, df)

    def invoke(self, query: str) -> Dict[str, Any]:
        docs = self.get_context(query, k=6)
        context = self.format_context(docs)

        prompt = f"""You are a film analyst expert at comparing movies.

Context (movies from database):
{context}

User Query: {query}

Compare the movies mentioned. Create a comparison table or structured analysis covering:
- Genre and themes
- Ratings and reception
- Cast and crew
- What makes each unique
Provide a recommendation on which to watch based on preferences."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        posters = [doc.metadata for doc in docs if doc.metadata.get('poster_url')]
        return {"answer": response.content, "posters": posters[:5], "agent": self.name}


class RecommendationAgent(BaseAgent):
    """Agent specialized in personalized recommendations."""

    def __init__(self, llm, vectorstore, df):
        super().__init__("Personalized Recommendation Agent", llm, vectorstore, df)

    def invoke(self, query: str) -> Dict[str, Any]:
        docs = self.get_context(query, k=8)
        context = self.format_context(docs)

        prompt = f"""You are a personalized movie recommendation expert.

Context (movies from database):
{context}

User Query: {query}

Provide personalized recommendations based on the user's preferences.
Explain WHY each movie is recommended and how it matches their interests.
Include a mix of popular and hidden gems."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        posters = [doc.metadata for doc in docs if doc.metadata.get('poster_url')]
        return {"answer": response.content, "posters": posters[:5], "agent": self.name}


class GeneralSearchAgent(BaseAgent):
    """General purpose search agent."""

    def __init__(self, llm, vectorstore, df):
        super().__init__("General Search Agent", llm, vectorstore, df)

    def invoke(self, query: str) -> Dict[str, Any]:
        docs = self.get_context(query)
        context = self.format_context(docs)

        prompt = f"""You are a knowledgeable movie assistant.

Context (movies from database):
{context}

User Query: {query}

Provide a helpful, informative response about movies. Include relevant details like titles, years, ratings, and descriptions."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        posters = [doc.metadata for doc in docs if doc.metadata.get('poster_url')]
        return {"answer": response.content, "posters": posters[:5], "agent": self.name}


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents."""

    def __init__(self, llm, vectorstore, df):
        self.classifier = QueryClassifier()
        self.agents = {
            "genre_search": GenreRecommendationAgent(llm, vectorstore, df),
            "actor_search": ActorDirectorAgent(llm, vectorstore, df),
            "director_search": ActorDirectorAgent(llm, vectorstore, df),
            "rating_search": RatingFilterAgent(llm, vectorstore, df),
            "year_search": GeneralSearchAgent(llm, vectorstore, df),
            "comparison": MovieComparisonAgent(llm, vectorstore, df),
            "recommendation": RecommendationAgent(llm, vectorstore, df),
            "specific_movie": GeneralSearchAgent(llm, vectorstore, df),
            "general_search": GeneralSearchAgent(llm, vectorstore, df),
        }
        self.conversation_history = []

    def process(self, query: str) -> Dict[str, Any]:
        """Process query through appropriate agent."""
        # Validate input
        if not query or len(query.strip()) < 3:
            return {
                "answer": "Please provide a more detailed question about movies.",
                "posters": [],
                "agent": "Input Validator"
            }

        # Classify and route
        query_type = self.classifier.classify(query)
        agent = self.agents.get(query_type, self.agents["general_search"])

        # Process through agent
        result = agent.invoke(query)

        # Store in history
        self.conversation_history.append({
            "query": query,
            "query_type": query_type,
            "agent": result["agent"],
            "response": result["answer"][:200]
        })

        return result

    def get_stats(self) -> Dict:
        """Get orchestrator statistics."""
        agent_usage = {}
        for item in self.conversation_history:
            agent = item["agent"]
            agent_usage[agent] = agent_usage.get(agent, 0) + 1

        return {
            "total_queries": len(self.conversation_history),
            "agent_usage": agent_usage
        }


# ============================================================
# TRENDING & QUIZ FEATURES
# ============================================================

class TrendingMovies:
    """Provides trending movie suggestions."""

    def __init__(self, df):
        self.df = df

    def get_top_rated(self, n: int = 5) -> List[Dict]:
        """Get top rated movies."""
        top = self.df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_by_genre(self, genre: str, n: int = 5) -> List[Dict]:
        """Get top movies by genre."""
        genre_df = self.df[self.df['Genre'].str.contains(genre, case=False, na=False)]
        top = genre_df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Get recent high-rated movies."""
        recent = self.df[self.df['Year'] >= 2015].nlargest(n, 'IMDb Rating')
        return self._format_movies(recent)

    def get_hidden_gems(self, n: int = 5) -> List[Dict]:
        """Get hidden gems (high rating but less known)."""
        gems = self.df[(self.df['IMDb Rating'] >= 7.5) & (self.df['IMDb Rating'] <= 8.5)]
        sample = gems.sample(n=min(n, len(gems)))
        return self._format_movies(sample)

    def _format_movies(self, df) -> List[Dict]:
        movies = []
        for _, row in df.iterrows():
            movies.append({
                'title': row['Title'],
                'year': int(row['Year']) if pd.notna(row['Year']) else 'N/A',
                'rating': row['IMDb Rating'],
                'genre': row['Genre'],
                'poster_url': row['Poster-src'] if pd.notna(row['Poster-src']) else '',
                'director': row['Director'] if pd.notna(row['Director']) else 'Unknown'
            })
        return movies


class MovieQuiz:
    """Interactive movie quiz game."""

    def __init__(self, df):
        self.df = df
        self.current_question = None
        self.score = 0
        self.questions_asked = 0

    def generate_question(self) -> Dict:
        """Generate a random quiz question."""
        question_types = [
            self._year_question,
            self._director_question,
            self._rating_question,
            self._genre_question
        ]

        question_func = random.choice(question_types)
        return question_func()

    def _year_question(self) -> Dict:
        movie = self.df.sample(1).iloc[0]
        correct_year = int(movie['Year'])

        # Generate wrong options
        wrong_years = [correct_year + random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(3)]
        options = [correct_year] + wrong_years
        random.shuffle(options)

        return {
            "question": f"In what year was '{movie['Title']}' released?",
            "options": options,
            "correct": correct_year,
            "movie": movie['Title']
        }

    def _director_question(self) -> Dict:
        movie = self.df[self.df['Director'].notna()].sample(1).iloc[0]
        correct_director = movie['Director']

        # Get random wrong directors
        other_directors = self.df[self.df['Director'] != correct_director]['Director'].dropna().unique()
        wrong_directors = list(np.random.choice(other_directors, min(3, len(other_directors)), replace=False))

        options = [correct_director] + wrong_directors
        random.shuffle(options)

        return {
            "question": f"Who directed '{movie['Title']}'?",
            "options": options,
            "correct": correct_director,
            "movie": movie['Title']
        }

    def _rating_question(self) -> Dict:
        movie = self.df.sample(1).iloc[0]
        correct_rating = round(movie['IMDb Rating'], 1)

        wrong_ratings = [round(correct_rating + random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]), 1) for _ in range(3)]
        wrong_ratings = [max(1.0, min(10.0, r)) for r in wrong_ratings]

        options = [correct_rating] + wrong_ratings
        random.shuffle(options)

        return {
            "question": f"What is the IMDb rating of '{movie['Title']}'?",
            "options": options,
            "correct": correct_rating,
            "movie": movie['Title']
        }

    def _genre_question(self) -> Dict:
        movie = self.df[self.df['Genre'].notna()].sample(1).iloc[0]
        correct_genre = movie['Genre'].split(',')[0].strip()

        all_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller", "Documentary", "Animation"]
        wrong_genres = [g for g in all_genres if g.lower() not in correct_genre.lower()][:3]

        options = [correct_genre] + wrong_genres
        random.shuffle(options)

        return {
            "question": f"What is the primary genre of '{movie['Title']}'?",
            "options": options,
            "correct": correct_genre,
            "movie": movie['Title']
        }

    def check_answer(self, question: Dict, answer) -> bool:
        """Check if the answer is correct."""
        self.questions_asked += 1
        is_correct = str(answer) == str(question['correct'])
        if is_correct:
            self.score += 1
        return is_correct


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title="IMDb Movie Chatbot",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #E50914; text-align: center;}
    .sub-header {font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 1rem;}
    .agent-badge {background: #E50914; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem;}
    .movie-card {background: #f8f9fa; border-radius: 10px; padding: 10px; margin: 5px; text-align: center; border: 1px solid #ddd;}
    .stat-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; text-align: center;}
    .quiz-card {background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 4px solid #ffc107;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🎬 IMDb Movie Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent AI Movie Discovery System</p>', unsafe_allow_html=True)

    # Initialize components
    if "initialized" not in st.session_state:
        with st.spinner("🔄 Loading AI agents and movie database..."):
            try:
                df = load_dataset()
                df = create_movie_descriptions(df)
                documents = create_documents(df)
                embeddings = get_embeddings()
                vectorstore = get_vectorstore(documents, embeddings)
                llm = get_llm()

                st.session_state.df = df
                st.session_state.orchestrator = MultiAgentOrchestrator(llm, vectorstore, df)
                st.session_state.trending = TrendingMovies(df)
                st.session_state.quiz = MovieQuiz(df)
                st.session_state.messages = []
                st.session_state.posters = []
                st.session_state.initialized = True
                st.success("✅ All agents loaded successfully!")
            except Exception as e:
                st.error(f"Error loading: {str(e)}")
                st.stop()

    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.radio("Choose Feature:", ["💬 Chat with Agents", "🔥 Trending Movies", "🎮 Movie Quiz", "📊 Statistics"])

        st.markdown("---")
        st.header("🤖 Active Agents")
        agents = ["🎭 Genre Agent", "🎬 Actor/Director Agent", "⭐ Rating Agent",
                  "🔄 Comparison Agent", "💡 Recommendation Agent", "🔍 Search Agent"]
        for agent in agents:
            st.markdown(f"- {agent}")

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.posters = []
            st.rerun()

    # Main content based on page selection
    if page == "💬 Chat with Agents":
        render_chat_page()
    elif page == "🔥 Trending Movies":
        render_trending_page()
    elif page == "🎮 Movie Quiz":
        render_quiz_page()
    elif page == "📊 Statistics":
        render_stats_page()


def render_chat_page():
    """Render the main chat interface."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat with Movie Agents")

        # Example queries
        st.markdown("**Quick queries:**")
        quick_queries = [
            "Recommend comedy movies",
            "Movies by Christopher Nolan",
            "Top rated documentaries",
            "Compare action vs thriller",
        ]
        cols = st.columns(4)
        for i, query in enumerate(quick_queries):
            if cols[i].button(query, key=f"quick_{i}"):
                st.session_state.pending_query = query

        # Chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "agent" in msg:
                    st.markdown(f'<span class="agent-badge">🤖 {msg["agent"]}</span>', unsafe_allow_html=True)
                st.markdown(msg["content"])

        # Handle pending query from quick buttons
        if "pending_query" in st.session_state:
            query = st.session_state.pending_query
            del st.session_state.pending_query
            process_query(query)

        # Chat input
        if prompt := st.chat_input("Ask about movies..."):
            process_query(prompt)

    with col2:
        st.subheader("🖼️ Movie Posters")
        if st.session_state.posters:
            for poster in st.session_state.posters[:4]:
                if poster.get('poster_url'):
                    try:
                        st.image(poster['poster_url'], caption=f"{poster.get('title', 'Movie')} ({poster.get('year', '')}) ⭐{poster.get('rating', 'N/A')}", width=140)
                    except:
                        pass
        else:
            st.info("Posters appear here after queries!")


def process_query(query: str):
    """Process a user query through the multi-agent system."""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents processing..."):
            result = st.session_state.orchestrator.process(query)
            st.markdown(f'<span class="agent-badge">🤖 {result["agent"]}</span>', unsafe_allow_html=True)
            st.markdown(result["answer"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "agent": result["agent"]
            })
            st.session_state.posters = result.get("posters", [])


def render_trending_page():
    """Render trending movies page."""
    st.subheader("🔥 Trending Movies")

    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Top Rated", "🆕 Recent Hits", "💎 Hidden Gems", "🎭 By Genre"])

    with tab1:
        st.markdown("### Top Rated Movies")
        movies = st.session_state.trending.get_top_rated(8)
        display_movie_grid(movies)

    with tab2:
        st.markdown("### Recent High-Rated Movies (2015+)")
        movies = st.session_state.trending.get_recent(8)
        display_movie_grid(movies)

    with tab3:
        st.markdown("### Hidden Gems")
        if st.button("🔄 Discover New Gems"):
            st.rerun()
        movies = st.session_state.trending.get_hidden_gems(8)
        display_movie_grid(movies)

    with tab4:
        genre = st.selectbox("Select Genre:", ["Action", "Comedy", "Drama", "Documentary", "Horror", "Romance", "Thriller", "Animation"])
        movies = st.session_state.trending.get_by_genre(genre, 8)
        display_movie_grid(movies)


def display_movie_grid(movies: List[Dict]):
    """Display movies in a grid format."""
    cols = st.columns(4)
    for i, movie in enumerate(movies):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="movie-card">
                <h4>{movie['title'][:20]}{'...' if len(movie['title']) > 20 else ''}</h4>
                <p>📅 {movie['year']} | ⭐ {movie['rating']}</p>
                <p style="font-size:0.8rem">{movie['genre'][:30]}</p>
            </div>
            """, unsafe_allow_html=True)
            if movie.get('poster_url'):
                try:
                    st.image(movie['poster_url'], width=120)
                except:
                    pass


def render_quiz_page():
    """Render the movie quiz page."""
    st.subheader("🎮 Movie Trivia Quiz")

    quiz = st.session_state.quiz

    # Score display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🏆 Score</h3>
            <h2>{quiz.score}/{quiz.questions_asked}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        accuracy = (quiz.score / quiz.questions_asked * 100) if quiz.questions_asked > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <h3>🎯 Accuracy</h3>
            <h2>{accuracy:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Generate new question
    if "current_question" not in st.session_state or st.session_state.current_question is None:
        st.session_state.current_question = quiz.generate_question()

    question = st.session_state.current_question

    st.markdown(f"""
    <div class="quiz-card">
        <h3>❓ {question['question']}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Answer options
    cols = st.columns(2)
    for i, option in enumerate(question['options']):
        if cols[i % 2].button(str(option), key=f"option_{i}", use_container_width=True):
            is_correct = quiz.check_answer(question, option)

            if is_correct:
                st.success(f"✅ Correct! The answer is {question['correct']}")
                st.balloons()
            else:
                st.error(f"❌ Wrong! The correct answer was {question['correct']}")

            st.session_state.current_question = None
            time.sleep(1.5)
            st.rerun()

    if st.button("⏭️ Skip Question"):
        st.session_state.current_question = None
        st.rerun()

    if st.button("🔄 Reset Quiz"):
        quiz.score = 0
        quiz.questions_asked = 0
        st.session_state.current_question = None
        st.rerun()


def render_stats_page():
    """Render statistics page."""
    st.subheader("📊 Session Statistics")

    stats = st.session_state.orchestrator.get_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>💬 Total Queries</h3>
            <h2>{stats['total_queries']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🎬 Movies in DB</h3>
            <h2>{len(st.session_state.df):,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🤖 Active Agents</h3>
            <h2>6</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Agent Usage")

    if stats['agent_usage']:
        for agent, count in stats['agent_usage'].items():
            st.progress(count / max(stats['total_queries'], 1), text=f"{agent}: {count} queries")
    else:
        st.info("No queries yet. Start chatting to see agent usage!")

    st.markdown("---")
    st.markdown("### 📈 Dataset Overview")

    df = st.session_state.df
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Genres:**")
        genre_counts = df['Genre'].value_counts().head(5)
        st.bar_chart(genre_counts)

    with col2:
        st.markdown("**Rating Distribution:**")
        rating_hist = df['IMDb Rating'].value_counts().sort_index()
        st.line_chart(rating_hist)


if __name__ == "__main__":
    main()
