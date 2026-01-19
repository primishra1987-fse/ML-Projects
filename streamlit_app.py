"""
IMDb Movie Chatbot - Enhanced Multi-Agent System
Features:
- 10+ Specialized AI Agents
- Voice-based search capability
- Multimodal support (poster analysis)
- Mood-based recommendations
- Review/Sentiment analysis
- Interactive quizzes & trivia
- Watchlist management
- Conversation memory with context retention
- Comprehensive error handling
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Optional, Any, Set
import time
import random
import hashlib
import re
from collections import deque
import streamlit as st
from abc import ABC, abstractmethod

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
log_file = os.path.join(LOG_DIR, f"chatbot_{datetime.now().strftime('%Y%m%d')}.log")

logger = logging.getLogger("MovieChatbot")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(file_handler)

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================
DATASET_PATH = "IMDb_Dataset (1).csv"
VECTORSTORE_PATH = "imdb_vectorstore"
CACHE_MAX_SIZE = 100
CACHE_TTL_SECONDS = 3600  # 1 hour
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # seconds
SEMANTIC_SIMILARITY_THRESHOLD = 0.92

# Mood mappings for mood-based recommendations
MOOD_GENRE_MAPPING = {
    "happy": ["Comedy", "Animation", "Musical", "Family"],
    "sad": ["Drama", "Romance"],
    "excited": ["Action", "Adventure", "Sci-Fi", "Thriller"],
    "scared": ["Horror", "Mystery", "Thriller"],
    "romantic": ["Romance", "Drama", "Comedy"],
    "thoughtful": ["Documentary", "Biography", "Drama", "History"],
    "nostalgic": ["Classic", "Family", "Animation"],
    "adventurous": ["Adventure", "Action", "Fantasy", "Sci-Fi"],
    "relaxed": ["Comedy", "Animation", "Family", "Documentary"],
    "inspired": ["Biography", "Documentary", "Drama", "Sport"]
}

# ============================================================
# CACHING SYSTEM
# ============================================================
class QueryCache:
    """Dual-layer caching with exact match and semantic similarity."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.ttl = ttl
        self.exact_cache: Dict[str, Dict] = {}
        self.semantic_cache: List[Dict] = []
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    def _hash_query(self, query: str) -> str:
        """Generate hash for exact matching."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str, embedding: Optional[List[float]] = None) -> Optional[Dict]:
        """Try to get cached response."""
        # Try exact match first
        query_hash = self._hash_query(query)
        if query_hash in self.exact_cache:
            entry = self.exact_cache[query_hash]
            if time.time() - entry['timestamp'] < self.ttl:
                self.stats["exact_hits"] += 1
                logger.info(f"Cache exact hit for query: {query[:50]}...")
                return entry['response']
            else:
                del self.exact_cache[query_hash]

        # Try semantic match if embedding provided
        if embedding:
            for entry in self.semantic_cache:
                if time.time() - entry['timestamp'] >= self.ttl:
                    continue
                similarity = self._cosine_similarity(embedding, entry['embedding'])
                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    self.stats["semantic_hits"] += 1
                    logger.info(f"Cache semantic hit (sim={similarity:.3f}) for query: {query[:50]}...")
                    return entry['response']

        self.stats["misses"] += 1
        return None

    def set(self, query: str, response: Dict, embedding: Optional[List[float]] = None):
        """Store response in cache."""
        # Evict oldest if needed
        if len(self.exact_cache) >= self.max_size:
            oldest = min(self.exact_cache.items(), key=lambda x: x[1]['timestamp'])
            del self.exact_cache[oldest[0]]

        query_hash = self._hash_query(query)
        entry = {
            'query': query,
            'response': response,
            'timestamp': time.time()
        }
        self.exact_cache[query_hash] = entry

        if embedding:
            if len(self.semantic_cache) >= self.max_size:
                self.semantic_cache.pop(0)
            self.semantic_cache.append({**entry, 'embedding': embedding})

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = sum(self.stats.values())
        hit_rate = ((self.stats["exact_hits"] + self.stats["semantic_hits"]) / total * 100) if total > 0 else 0
        return {**self.stats, "hit_rate_percent": round(hit_rate, 2)}


class RateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()

    def is_allowed(self) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, wait_time)."""
        now = time.time()

        # Remove old requests
        while self.requests and now - self.requests[0] > self.window_seconds:
            self.requests.popleft()

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True, 0

        wait_time = int(self.window_seconds - (now - self.requests[0])) + 1
        return False, wait_time


# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_resource
def load_dataset():
    """Load the IMDb dataset with error handling."""
    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded successfully with {len(df)} movies")
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATASET_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

@st.cache_resource
def create_movie_descriptions(_df):
    """Create rich movie descriptions for embedding."""
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

        # Enhanced description with more context for better retrieval
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

This is a {genre} movie titled "{title}" released in {year}, directed by {director} and starring {cast}. It has an IMDb rating of {rating}/10 and MetaScore of {metascore}. The film runs for {duration} minutes and is rated {certificate}."""

    df = _df.copy()
    df['description'] = df.apply(create_description, axis=1)
    return df

@st.cache_resource
def create_documents(_df):
    """Convert DataFrame to LangChain documents with rich metadata."""
    documents = []
    for idx, row in _df.iterrows():
        metadata = {
            'title': row['Title'] if pd.notna(row['Title']) else 'Unknown',
            'year': int(row['Year']) if pd.notna(row['Year']) else 0,
            'genre': row['Genre'] if pd.notna(row['Genre']) else 'Unknown',
            'director': row['Director'] if pd.notna(row['Director']) else 'Unknown',
            'rating': float(row['IMDb Rating']) if pd.notna(row['IMDb Rating']) else 0.0,
            'metascore': float(row['MetaScore']) if pd.notna(row['MetaScore']) else 0.0,
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
    if os.path.exists(VECTORSTORE_PATH):
        try:
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing FAISS vector store")
        except Exception as e:
            logger.warning(f"Error loading vector store, recreating: {e}")
            vectorstore = FAISS.from_documents(documents=_documents, embedding=_embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
    else:
        vectorstore = FAISS.from_documents(documents=_documents, embedding=_embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info("Created new FAISS vector store")
    return vectorstore

@st.cache_resource
def get_llm():
    """Initialize ChatOpenAI with optimized settings."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1200)

# ============================================================
# ENHANCED QUERY CLASSIFIER
# ============================================================

class EnhancedQueryClassifier:
    """Advanced query classifier with multi-label support and confidence scoring."""

    QUERY_PATTERNS = {
        "genre_search": {
            "keywords": ["genre", "comedy", "action", "drama", "horror", "thriller", "romance",
                        "documentary", "biography", "adventure", "sci-fi", "animation", "fantasy",
                        "mystery", "crime", "war", "western", "musical", "sport", "family"],
            "patterns": [r"(find|show|get|list).*(comedy|action|drama|horror)", r"\b(genres?)\b"]
        },
        "actor_search": {
            "keywords": ["actor", "actress", "starring", "star", "played by", "acted", "cast", "featuring"],
            "patterns": [r"movies?\s+(with|starring|featuring)\s+", r"(actor|actress)\s+\w+"]
        },
        "director_search": {
            "keywords": ["director", "directed", "filmmaker", "made by", "directed by"],
            "patterns": [r"(directed|director)\s+by?\s*\w+", r"films?\s+by\s+"]
        },
        "rating_search": {
            "keywords": ["rating", "rated", "best", "top", "highest", "score", "imdb", "metascore", "lowest"],
            "patterns": [r"(top|best|highest)\s+\d*\s*(rated|movies?)", r"rating\s*(above|over|below|under)\s*\d"]
        },
        "year_search": {
            "keywords": ["year", "released", "came out", "from 19", "from 20", "in 19", "in 20", "decade", "era"],
            "patterns": [r"(in|from|since|before|after)\s*(19|20)\d{2}", r"(decade|era|years?)"]
        },
        "comparison": {
            "keywords": ["compare", "versus", "vs", "difference", "better", "which one", "or"],
            "patterns": [r"(compare|versus|vs\.?)\s+", r"(better|worse)\s+(than|movie)"]
        },
        "recommendation": {
            "keywords": ["recommend", "suggest", "similar", "like", "should i watch", "what to watch", "suggestions"],
            "patterns": [r"(recommend|suggest)\s+(me\s+)?", r"similar\s+to\s+", r"movies?\s+like\s+"]
        },
        "mood_based": {
            "keywords": ["mood", "feeling", "feel like", "in the mood", "happy", "sad", "excited", "scared",
                        "romantic", "thoughtful", "nostalgic", "adventurous", "relaxed", "inspired"],
            "patterns": [r"(mood|feeling|feel\s+like)", r"i('m|\s+am)\s+(happy|sad|excited|scared)"]
        },
        "trivia": {
            "keywords": ["trivia", "fact", "facts", "interesting", "did you know", "fun fact", "behind the scenes"],
            "patterns": [r"(trivia|facts?)\s+(about|for)", r"interesting\s+(facts?|things?)"]
        },
        "review_sentiment": {
            "keywords": ["review", "reviews", "critics", "audience", "reception", "thoughts on", "opinion"],
            "patterns": [r"(reviews?|critics?|reception)", r"(thoughts?|opinions?)\s+on"]
        },
        "specific_movie": {
            "keywords": ["tell me about", "what is", "details about", "info about", "information", "plot", "story"],
            "patterns": [r"(tell|what).*(about|is)\s+", r"(plot|story)\s+of"]
        },
        "duration_search": {
            "keywords": ["duration", "runtime", "long", "short", "hours", "minutes", "length"],
            "patterns": [r"(duration|runtime|length)", r"(how\s+long|short\s+movies?)"]
        }
    }

    def classify(self, query: str) -> Tuple[str, float]:
        """Classify query with confidence score."""
        query_lower = query.lower()
        scores = {}

        for query_type, config in self.QUERY_PATTERNS.items():
            score = 0

            # Check keywords
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1

            # Check regex patterns
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    score += 2

            if score > 0:
                scores[query_type] = score

        if not scores:
            return "general_search", 0.5

        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 5.0, 1.0)

        return best_type, confidence

    def get_all_matches(self, query: str) -> List[Tuple[str, float]]:
        """Get all matching query types with scores."""
        query_lower = query.lower()
        matches = []

        for query_type, config in self.QUERY_PATTERNS.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    score += 2

            if score > 0:
                matches.append((query_type, min(score / 5.0, 1.0)))

        return sorted(matches, key=lambda x: x[1], reverse=True)


# ============================================================
# CONVERSATION MEMORY SYSTEM
# ============================================================

class ConversationMemory:
    """Advanced conversation memory with context retention."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict] = []
        self.context_entities: Set[str] = set()  # Track mentioned movies, actors, etc.
        self.user_preferences: Dict = {
            "liked_genres": [],
            "liked_movies": [],
            "disliked_genres": [],
            "mentioned_actors": [],
            "mentioned_directors": []
        }

    def add_turn(self, user_query: str, assistant_response: str, metadata: Dict = None):
        """Add a conversation turn."""
        turn = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(turn)

        # Maintain max size
        if len(self.history) > self.max_turns:
            self.history.pop(0)

        # Extract entities from the response
        self._extract_entities(user_query, assistant_response, metadata)

    def _extract_entities(self, query: str, response: str, metadata: Dict):
        """Extract and track entities from conversation."""
        if metadata:
            if "posters" in metadata:
                for poster in metadata["posters"]:
                    if poster.get("title"):
                        self.context_entities.add(poster["title"])

    def get_context_summary(self) -> str:
        """Get a summary of conversation context."""
        if not self.history:
            return ""

        recent = self.history[-3:]  # Last 3 turns
        summary = "Recent conversation context:\n"
        for turn in recent:
            summary += f"User: {turn['user'][:100]}...\n"
            summary += f"Assistant: {turn['assistant'][:150]}...\n\n"

        if self.context_entities:
            summary += f"Previously discussed: {', '.join(list(self.context_entities)[:5])}"

        return summary

    def get_full_history(self) -> List[Dict]:
        """Get full conversation history."""
        return self.history

    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.context_entities = set()


# ============================================================
# BASE AGENT CLASS
# ============================================================

class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""

    def __init__(self, name: str, description: str, llm, vectorstore, df):
        self.name = name
        self.description = description
        self.llm = llm
        self.vectorstore = vectorstore
        self.df = df
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def get_context(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents."""
        self.retriever.search_kwargs["k"] = k
        return self.retriever.invoke(query)

    def format_context(self, docs: List[Document]) -> str:
        """Format documents as context string."""
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def extract_posters(self, docs: List[Document], limit: int = 5) -> List[Dict]:
        """Extract poster information from documents."""
        posters = []
        for doc in docs:
            if doc.metadata.get('poster_url'):
                posters.append(doc.metadata)
        return posters[:limit]

    @abstractmethod
    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        """Process the query - to be implemented by subclasses."""
        pass

    def _create_response(self, answer: str, docs: List[Document], extra_data: Dict = None) -> Dict[str, Any]:
        """Create standardized response."""
        response = {
            "answer": answer,
            "posters": self.extract_posters(docs),
            "agent": self.name,
            "description": self.description
        }
        if extra_data:
            response.update(extra_data)
        return response


# ============================================================
# SPECIALIZED AGENTS (10+ Agents)
# ============================================================

class GenreRecommendationAgent(BaseAgent):
    """Agent specialized in genre-based recommendations."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Genre Expert",
            "Specializes in genre-based movie recommendations",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=8)
        doc_context = self.format_context(docs)

        prompt = f"""You are an expert movie curator specializing in film genres. Your role is to provide insightful, engaging genre-based recommendations.

CONVERSATION CONTEXT:
{context}

MOVIE DATABASE CONTEXT:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Identify the genre(s) the user is interested in
2. Recommend 3-5 movies that best represent that genre
3. For each movie, explain WHY it's a great example of the genre
4. Include: Title, Year, Rating, Director, and a brief compelling description
5. If applicable, mention subgenres or genre blends
6. Add a "Pro Tip" about the genre for movie enthusiasts

Format your response with clear headers and bullet points for readability."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class ActorDirectorAgent(BaseAgent):
    """Agent specialized in actor and director filmography searches."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Filmography Expert",
            "Specializes in actor and director career analysis",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=10)
        doc_context = self.format_context(docs)

        prompt = f"""You are a film industry expert with deep knowledge of actors' and directors' careers.

CONVERSATION CONTEXT:
{context}

FILMOGRAPHY DATABASE:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Identify the actor/director mentioned in the query
2. List their notable films from the database with:
   - Title and Year
   - Genre
   - IMDb Rating
   - Their role (starring/supporting for actors, director for directors)
3. Highlight their career trajectory and notable achievements
4. Mention any frequent collaborations with other actors/directors
5. Suggest their "must-watch" films for newcomers

Format with clear sections and engaging descriptions."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class RatingFilterAgent(BaseAgent):
    """Agent specialized in rating-based searches and analysis."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Rating Analyst",
            "Expert in IMDb ratings and critical reception",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        # Get top rated movies from dataframe
        high_rated = self.df[self.df['IMDb Rating'] >= 8.0].nlargest(15, 'IMDb Rating')

        docs = self.get_context(query, k=8)
        doc_context = self.format_context(docs)

        top_movies = "\n".join([
            f"- {row['Title']} ({row['Year']}) - IMDb: {row['IMDb Rating']}/10, MetaScore: {row['MetaScore']}"
            for _, row in high_rated.head(10).iterrows()
        ])

        prompt = f"""You are a film critic and ratings analyst with expertise in understanding movie quality metrics.

CONVERSATION CONTEXT:
{context}

TOP RATED MOVIES IN DATABASE:
{top_movies}

ADDITIONAL CONTEXT:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the user's rating preferences
2. Recommend movies based on IMDb ratings and MetaScore
3. Explain what makes highly-rated movies stand out
4. Include a mix of:
   - Critically acclaimed masterpieces (8.5+)
   - Solid crowd-pleasers (7.5-8.5)
   - Hidden gems with surprising ratings
5. Provide rating context (what the numbers mean)

Format with ratings prominently displayed."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class MovieComparisonAgent(BaseAgent):
    """Agent specialized in comparing movies."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Comparison Analyst",
            "Expert at comparing and contrasting films",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=8)
        doc_context = self.format_context(docs)

        prompt = f"""You are a film analyst expert at detailed movie comparisons.

CONVERSATION CONTEXT:
{context}

MOVIE DATABASE:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Identify the movies to compare from the query
2. Create a structured comparison covering:

   **Basic Info Table:**
   | Aspect | Movie 1 | Movie 2 |
   |--------|---------|---------|
   | Year | | |
   | Director | | |
   | Genre | | |
   | Rating | | |
   | Duration | | |

3. **Thematic Comparison:**
   - Plot and storytelling approach
   - Visual style and cinematography
   - Acting performances
   - Cultural impact

4. **Verdict:**
   - Who should watch which movie
   - Best for different moods/occasions

5. **Final Recommendation** based on user's likely preference

Be objective but engaging."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class MoodBasedAgent(BaseAgent):
    """Agent that recommends movies based on user's mood."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Mood Curator",
            "Recommends movies based on your current mood",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        # Detect mood from query
        query_lower = query.lower()
        detected_mood = "relaxed"  # default

        for mood, genres in MOOD_GENRE_MAPPING.items():
            if mood in query_lower:
                detected_mood = mood
                break

        # Get movies matching mood genres
        mood_genres = MOOD_GENRE_MAPPING.get(detected_mood, ["Drama"])
        genre_pattern = '|'.join(mood_genres)
        mood_movies = self.df[self.df['Genre'].str.contains(genre_pattern, case=False, na=False)]
        top_mood = mood_movies.nlargest(10, 'IMDb Rating')

        docs = self.get_context(f"{detected_mood} {' '.join(mood_genres)}", k=8)
        doc_context = self.format_context(docs)

        mood_movie_list = "\n".join([
            f"- {row['Title']} ({row['Year']}) - {row['Genre']} - {row['IMDb Rating']}/10"
            for _, row in top_mood.head(8).iterrows()
        ])

        prompt = f"""You are an empathetic movie curator who understands how films can match and enhance moods.

DETECTED MOOD: {detected_mood.upper()}
RECOMMENDED GENRES FOR THIS MOOD: {', '.join(mood_genres)}

CONVERSATION CONTEXT:
{context}

MOOD-MATCHED MOVIES:
{mood_movie_list}

ADDITIONAL CONTEXT:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Acknowledge the user's mood warmly
2. Explain why certain genres work well for their mood
3. Recommend 4-5 movies that perfectly match their emotional state
4. For each movie explain:
   - Why it fits the mood
   - What emotional journey to expect
   - Best viewing conditions (alone, with friends, etc.)
5. Include a "Mood Booster" tip

Be warm, understanding, and enthusiastic!"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs, {"detected_mood": detected_mood})


class TriviaAgent(BaseAgent):
    """Agent that provides movie trivia and interesting facts."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Trivia Master",
            "Provides fascinating movie trivia and behind-the-scenes facts",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=6)
        doc_context = self.format_context(docs)

        prompt = f"""You are an entertaining movie trivia expert with encyclopedic knowledge of cinema history.

CONVERSATION CONTEXT:
{context}

MOVIE DATABASE:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Identify the movie(s) the user is asking about
2. Share 5-7 fascinating trivia facts including:
   - Behind-the-scenes stories
   - Casting decisions and almost-cast actors
   - Production challenges and solutions
   - Easter eggs and hidden details
   - Box office and cultural impact facts
   - Award nominations and wins
3. Use engaging language like "Did you know..." and "Fun fact:"
4. Include a "Mind-Blowing Fact" as the finale
5. If relevant, connect trivia to other movies

Make it fun and educational!"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class ReviewSentimentAgent(BaseAgent):
    """Agent that analyzes and synthesizes movie reviews and reception."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Review Analyst",
            "Analyzes critical and audience reception",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=6)
        doc_context = self.format_context(docs)

        # Get rating info for the movies
        movie_ratings = []
        for doc in docs[:3]:
            title = doc.metadata.get('title', 'Unknown')
            rating = doc.metadata.get('rating', 0)
            metascore = doc.metadata.get('metascore', 0)
            movie_ratings.append(f"{title}: IMDb {rating}/10, MetaScore {metascore}")

        prompt = f"""You are a film critic who synthesizes critical and audience reception into insightful analysis.

CONVERSATION CONTEXT:
{context}

MOVIE RATINGS DATA:
{chr(10).join(movie_ratings)}

MOVIE DETAILS:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Analyze the critical reception of the movie(s) mentioned
2. Provide a balanced review summary covering:

   **Critical Reception:**
   - What critics praised
   - Any criticisms or controversies
   - Award recognition

   **Audience Reception:**
   - General audience response
   - Cult following or divisive opinions
   - Legacy and lasting impact

   **Rating Analysis:**
   - What the IMDb and MetaScore ratings tell us
   - Gap between critic and audience scores (if any)

3. Give your **Verdict:** Should the user watch it?
4. **Best For:** Who would enjoy this movie most

Be analytical but accessible."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class SimilarMoviesAgent(BaseAgent):
    """Agent that finds similar movies based on various criteria."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Similarity Finder",
            "Finds movies similar to ones you love",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=12)
        doc_context = self.format_context(docs)

        prompt = f"""You are a movie recommendation expert who finds perfect movie matches based on what users already love.

CONVERSATION CONTEXT:
{context}

SIMILAR MOVIES DATABASE:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Identify the reference movie(s) the user mentions
2. Analyze what makes that movie special:
   - Genre and subgenre
   - Themes and tone
   - Visual style
   - Narrative structure
   - Emotional impact

3. Recommend 5-6 similar movies, categorized by:

   **Same Director/Cast:**
   - Movies by the same filmmakers

   **Same Genre & Tone:**
   - Movies with similar feel

   **Thematic Siblings:**
   - Movies exploring similar themes

   **Hidden Gems:**
   - Lesser-known movies you might love

4. For each recommendation, explain the CONNECTION to the original
5. Rate similarity: "Very Similar", "Spiritually Similar", "If you liked X, try Y"

Make connections clear and compelling!"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class DurationAgent(BaseAgent):
    """Agent for finding movies by duration/runtime."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Runtime Advisor",
            "Finds movies based on available time",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        # Parse duration preferences from query
        query_lower = query.lower()

        if any(word in query_lower for word in ["short", "quick", "brief"]):
            duration_filter = self.df[self.df['Duration (minutes)'] <= 100]
            duration_desc = "short (under 100 minutes)"
        elif any(word in query_lower for word in ["long", "epic", "extended"]):
            duration_filter = self.df[self.df['Duration (minutes)'] >= 150]
            duration_desc = "epic length (150+ minutes)"
        else:
            duration_filter = self.df[(self.df['Duration (minutes)'] >= 90) &
                                       (self.df['Duration (minutes)'] <= 130)]
            duration_desc = "standard length (90-130 minutes)"

        top_duration = duration_filter.nlargest(10, 'IMDb Rating')

        docs = self.get_context(query, k=6)
        doc_context = self.format_context(docs)

        duration_list = "\n".join([
            f"- {row['Title']} ({row['Year']}) - {row['Duration (minutes)']} min - {row['IMDb Rating']}/10"
            for _, row in top_duration.head(8).iterrows()
        ])

        prompt = f"""You are a movie guide who helps people find films that fit their available time.

DURATION PREFERENCE: {duration_desc}

MOVIES MATCHING DURATION:
{duration_list}

ADDITIONAL CONTEXT:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Acknowledge the user's time constraints
2. Recommend movies that fit their available time
3. For each movie include:
   - Title, Year, and exact runtime
   - Genre and rating
   - Why it's worth the time investment
4. Group by time slots:
   - Quick Watch (under 90 min)
   - Standard (90-120 min)
   - Extended (120-150 min)
   - Epic (150+ min)
5. Add a "Best Value for Time" pick

Be helpful and time-conscious!"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class YearEraAgent(BaseAgent):
    """Agent for exploring movies by year, decade, or era."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "Era Explorer",
            "Expert in cinema history across decades",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        # Try to extract year/decade from query
        year_match = re.search(r'(19|20)\d{2}', query)
        decade_match = re.search(r'(19|20)\d{1}0s', query)

        if decade_match:
            decade_start = int(decade_match.group()[:4])
            era_filter = self.df[(self.df['Year'] >= decade_start) &
                                  (self.df['Year'] < decade_start + 10)]
            era_desc = f"the {decade_match.group()}"
        elif year_match:
            year = int(year_match.group())
            era_filter = self.df[(self.df['Year'] >= year - 2) &
                                  (self.df['Year'] <= year + 2)]
            era_desc = f"around {year}"
        else:
            era_filter = self.df[self.df['Year'] >= 2020]
            era_desc = "recent years (2020+)"

        top_era = era_filter.nlargest(10, 'IMDb Rating')

        docs = self.get_context(query, k=8)
        doc_context = self.format_context(docs)

        era_list = "\n".join([
            f"- {row['Title']} ({row['Year']}) - {row['Genre']} - {row['IMDb Rating']}/10"
            for _, row in top_era.head(8).iterrows()
        ])

        prompt = f"""You are a cinema historian with deep knowledge of film history across all eras.

ERA/TIME PERIOD: {era_desc}

TOP MOVIES FROM THIS ERA:
{era_list}

ADDITIONAL CONTEXT:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Provide context about cinema during this era
2. Recommend the best movies from this time period
3. For each movie, explain:
   - Why it's significant for its era
   - How it reflects the times
   - Its lasting influence
4. Include:
   - Era-defining classics
   - Innovative films that changed cinema
   - Cultural touchstones
5. Add "Historical Context" about filmmaking during this era

Be informative and nostalgic!"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


class GeneralSearchAgent(BaseAgent):
    """General purpose search agent for miscellaneous queries."""

    def __init__(self, llm, vectorstore, df):
        super().__init__(
            "General Assistant",
            "Versatile movie knowledge assistant",
            llm, vectorstore, df
        )

    def invoke(self, query: str, context: str = "") -> Dict[str, Any]:
        docs = self.get_context(query, k=8)
        doc_context = self.format_context(docs)

        prompt = f"""You are a knowledgeable and friendly movie assistant ready to help with any movie-related query.

CONVERSATION CONTEXT:
{context}

MOVIE DATABASE:
{doc_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Understand what the user is looking for
2. Provide comprehensive, helpful information
3. Include relevant movie recommendations with:
   - Title, Year, Genre
   - IMDb Rating
   - Brief description
   - Why it matches their query
4. Be conversational and engaging
5. If the query is unclear, make reasonable assumptions and note them
6. End with a follow-up question or additional suggestion

Be helpful, informative, and friendly!"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._create_response(response.content, docs)


# ============================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================

class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents with intelligent routing."""

    def __init__(self, llm, vectorstore, df):
        self.classifier = EnhancedQueryClassifier()
        self.memory = ConversationMemory()
        self.cache = QueryCache()
        self.rate_limiter = RateLimiter()

        # Initialize all agents
        self.agents = {
            "genre_search": GenreRecommendationAgent(llm, vectorstore, df),
            "actor_search": ActorDirectorAgent(llm, vectorstore, df),
            "director_search": ActorDirectorAgent(llm, vectorstore, df),
            "rating_search": RatingFilterAgent(llm, vectorstore, df),
            "year_search": YearEraAgent(llm, vectorstore, df),
            "comparison": MovieComparisonAgent(llm, vectorstore, df),
            "recommendation": SimilarMoviesAgent(llm, vectorstore, df),
            "mood_based": MoodBasedAgent(llm, vectorstore, df),
            "trivia": TriviaAgent(llm, vectorstore, df),
            "review_sentiment": ReviewSentimentAgent(llm, vectorstore, df),
            "specific_movie": GeneralSearchAgent(llm, vectorstore, df),
            "duration_search": DurationAgent(llm, vectorstore, df),
            "general_search": GeneralSearchAgent(llm, vectorstore, df),
        }

        self.query_history = []
        self.agent_usage = {}

    def process(self, query: str) -> Dict[str, Any]:
        """Process query through appropriate agent with full pipeline."""
        start_time = time.time()

        # Input validation
        validation_result = self._validate_input(query)
        if not validation_result["valid"]:
            return {
                "answer": validation_result["message"],
                "posters": [],
                "agent": "Input Validator",
                "description": "Validates user input"
            }

        # Rate limiting
        allowed, wait_time = self.rate_limiter.is_allowed()
        if not allowed:
            return {
                "answer": f"You're sending requests too quickly. Please wait {wait_time} seconds.",
                "posters": [],
                "agent": "Rate Limiter",
                "description": "Protects system from overload"
            }

        # Check cache
        cached = self.cache.get(query)
        if cached:
            cached["from_cache"] = True
            return cached

        # Classify query
        query_type, confidence = self.classifier.classify(query)

        # Get appropriate agent
        agent = self.agents.get(query_type, self.agents["general_search"])

        # Get conversation context
        context = self.memory.get_context_summary()

        # Process through agent
        try:
            result = agent.invoke(query, context)
            result["query_type"] = query_type
            result["confidence"] = confidence
            result["processing_time"] = round(time.time() - start_time, 2)
            result["from_cache"] = False

            # Update memory
            self.memory.add_turn(query, result["answer"], {"posters": result.get("posters", [])})

            # Update statistics
            self._update_stats(agent.name, query_type)

            # Cache result
            self.cache.set(query, result)

            logger.info(f"Query processed: type={query_type}, agent={agent.name}, time={result['processing_time']}s")

            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I encountered an issue processing your request. Please try rephrasing your question. Error: {str(e)}",
                "posters": [],
                "agent": "Error Handler",
                "description": "Handles processing errors"
            }

    def _validate_input(self, query: str) -> Dict:
        """Validate user input."""
        if not query:
            return {"valid": False, "message": "Please enter a question about movies."}

        query = query.strip()

        if len(query) < 3:
            return {"valid": False, "message": "Your question is too short. Please provide more details."}

        if len(query) > 1000:
            return {"valid": False, "message": "Your question is too long. Please keep it under 1000 characters."}

        # Check for potentially harmful patterns
        harmful_patterns = [r'<script', r'javascript:', r'DROP TABLE', r'DELETE FROM']
        for pattern in harmful_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {"valid": False, "message": "Invalid input detected. Please ask a movie-related question."}

        return {"valid": True, "message": ""}

    def _update_stats(self, agent_name: str, query_type: str):
        """Update usage statistics."""
        self.agent_usage[agent_name] = self.agent_usage.get(agent_name, 0) + 1
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "agent": agent_name
        })

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        cache_stats = self.cache.get_stats()
        return {
            "total_queries": len(self.query_history),
            "agent_usage": self.agent_usage,
            "cache_stats": cache_stats,
            "agents_available": len(self.agents)
        }

    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()


# ============================================================
# TRENDING MOVIES FEATURE
# ============================================================

class TrendingMovies:
    """Provides trending and curated movie suggestions."""

    def __init__(self, df):
        self.df = df

    def get_top_rated(self, n: int = 8) -> List[Dict]:
        """Get top rated movies of all time."""
        top = self.df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_by_genre(self, genre: str, n: int = 8) -> List[Dict]:
        """Get top movies by genre."""
        genre_df = self.df[self.df['Genre'].str.contains(genre, case=False, na=False)]
        top = genre_df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_recent(self, n: int = 8) -> List[Dict]:
        """Get recent high-rated movies (2015+)."""
        recent = self.df[self.df['Year'] >= 2015].nlargest(n, 'IMDb Rating')
        return self._format_movies(recent)

    def get_hidden_gems(self, n: int = 8) -> List[Dict]:
        """Get hidden gems (good rating, potentially lesser-known)."""
        gems = self.df[(self.df['IMDb Rating'] >= 7.5) & (self.df['IMDb Rating'] <= 8.5)]
        sample = gems.sample(n=min(n, len(gems)))
        return self._format_movies(sample)

    def get_by_decade(self, decade: int, n: int = 8) -> List[Dict]:
        """Get top movies from a specific decade."""
        decade_df = self.df[(self.df['Year'] >= decade) & (self.df['Year'] < decade + 10)]
        top = decade_df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_by_director(self, director: str, n: int = 8) -> List[Dict]:
        """Get movies by a specific director."""
        director_df = self.df[self.df['Director'].str.contains(director, case=False, na=False)]
        top = director_df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_short_movies(self, max_duration: int = 100, n: int = 8) -> List[Dict]:
        """Get highly-rated short movies."""
        short_df = self.df[self.df['Duration (minutes)'] <= max_duration]
        top = short_df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def get_epic_movies(self, min_duration: int = 150, n: int = 8) -> List[Dict]:
        """Get highly-rated epic-length movies."""
        epic_df = self.df[self.df['Duration (minutes)'] >= min_duration]
        top = epic_df.nlargest(n, 'IMDb Rating')
        return self._format_movies(top)

    def _format_movies(self, df) -> List[Dict]:
        """Format dataframe rows as movie dictionaries."""
        movies = []
        for _, row in df.iterrows():
            movies.append({
                'title': row['Title'],
                'year': int(row['Year']) if pd.notna(row['Year']) else 'N/A',
                'rating': row['IMDb Rating'],
                'genre': row['Genre'],
                'poster_url': row['Poster-src'] if pd.notna(row['Poster-src']) else '',
                'director': row['Director'] if pd.notna(row['Director']) else 'Unknown',
                'duration': int(row['Duration (minutes)']) if pd.notna(row['Duration (minutes)']) else 'N/A',
                'cast': row['Star Cast'] if pd.notna(row['Star Cast']) else 'Unknown'
            })
        return movies


# ============================================================
# INTERACTIVE QUIZ SYSTEM
# ============================================================

class MovieQuiz:
    """Enhanced interactive movie quiz with multiple question types."""

    DIFFICULTY_LEVELS = {
        "easy": {"points": 1, "time_bonus": 0},
        "medium": {"points": 2, "time_bonus": 5},
        "hard": {"points": 3, "time_bonus": 10}
    }

    def __init__(self, df):
        self.df = df
        self.score = 0
        self.questions_asked = 0
        self.streak = 0
        self.max_streak = 0
        self.category_scores = {
            "year": {"correct": 0, "total": 0},
            "director": {"correct": 0, "total": 0},
            "rating": {"correct": 0, "total": 0},
            "genre": {"correct": 0, "total": 0},
            "cast": {"correct": 0, "total": 0},
            "duration": {"correct": 0, "total": 0}
        }

    def generate_question(self, difficulty: str = "medium") -> Dict:
        """Generate a random quiz question."""
        question_generators = [
            self._year_question,
            self._director_question,
            self._rating_question,
            self._genre_question,
            self._cast_question,
            self._duration_question
        ]

        generator = random.choice(question_generators)
        question = generator()
        question["difficulty"] = difficulty
        question["points"] = self.DIFFICULTY_LEVELS[difficulty]["points"]
        return question

    def _year_question(self) -> Dict:
        """Generate a year-based question."""
        movie = self.df.sample(1).iloc[0]
        correct_year = int(movie['Year'])

        wrong_years = list(set([
            correct_year + random.choice([-5, -3, -2, -1, 1, 2, 3, 5])
            for _ in range(6)
        ]))[:3]

        options = [correct_year] + wrong_years
        random.shuffle(options)

        return {
            "question": f"In what year was '{movie['Title']}' released?",
            "options": options,
            "correct": correct_year,
            "movie": movie['Title'],
            "category": "year",
            "hint": f"Hint: It stars {movie['Star Cast'].split(',')[0] if pd.notna(movie['Star Cast']) else 'famous actors'}"
        }

    def _director_question(self) -> Dict:
        """Generate a director-based question."""
        movie = self.df[self.df['Director'].notna()].sample(1).iloc[0]
        correct_director = movie['Director']

        other_directors = self.df[self.df['Director'] != correct_director]['Director'].dropna().unique()
        wrong_directors = list(np.random.choice(other_directors, min(3, len(other_directors)), replace=False))

        options = [correct_director] + wrong_directors
        random.shuffle(options)

        return {
            "question": f"Who directed '{movie['Title']}' ({int(movie['Year'])})?",
            "options": options,
            "correct": correct_director,
            "movie": movie['Title'],
            "category": "director",
            "hint": f"Hint: It's a {movie['Genre'].split(',')[0]} film"
        }

    def _rating_question(self) -> Dict:
        """Generate a rating-based question."""
        movie = self.df.sample(1).iloc[0]
        correct_rating = round(movie['IMDb Rating'], 1)

        wrong_ratings = []
        for _ in range(3):
            offset = random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
            wrong = round(correct_rating + offset, 1)
            wrong = max(1.0, min(10.0, wrong))
            if wrong != correct_rating:
                wrong_ratings.append(wrong)

        wrong_ratings = list(set(wrong_ratings))[:3]
        while len(wrong_ratings) < 3:
            wrong_ratings.append(round(random.uniform(5.0, 9.0), 1))

        options = [correct_rating] + wrong_ratings[:3]
        random.shuffle(options)

        return {
            "question": f"What is the IMDb rating of '{movie['Title']}'?",
            "options": options,
            "correct": correct_rating,
            "movie": movie['Title'],
            "category": "rating",
            "hint": f"Hint: It was released in {int(movie['Year'])}"
        }

    def _genre_question(self) -> Dict:
        """Generate a genre-based question."""
        movie = self.df[self.df['Genre'].notna()].sample(1).iloc[0]
        correct_genre = movie['Genre'].split(',')[0].strip()

        all_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller",
                      "Documentary", "Animation", "Sci-Fi", "Adventure", "Crime", "Fantasy"]
        wrong_genres = [g for g in all_genres if g.lower() not in correct_genre.lower()][:3]

        options = [correct_genre] + wrong_genres
        random.shuffle(options)

        return {
            "question": f"What is the primary genre of '{movie['Title']}'?",
            "options": options,
            "correct": correct_genre,
            "movie": movie['Title'],
            "category": "genre",
            "hint": f"Hint: Directed by {movie['Director']}"
        }

    def _cast_question(self) -> Dict:
        """Generate a cast-based question."""
        movie = self.df[self.df['Star Cast'].notna()].sample(1).iloc[0]
        cast_list = [c.strip() for c in str(movie['Star Cast']).split(',')]
        correct_actor = cast_list[0] if cast_list else "Unknown"

        other_actors = []
        for _, row in self.df[self.df['Star Cast'].notna()].sample(10).iterrows():
            actors = [a.strip() for a in str(row['Star Cast']).split(',')]
            other_actors.extend(actors)

        wrong_actors = [a for a in set(other_actors) if a != correct_actor][:3]

        options = [correct_actor] + wrong_actors
        random.shuffle(options)

        return {
            "question": f"Who is the lead actor in '{movie['Title']}'?",
            "options": options,
            "correct": correct_actor,
            "movie": movie['Title'],
            "category": "cast",
            "hint": f"Hint: It's rated {movie['IMDb Rating']}/10"
        }

    def _duration_question(self) -> Dict:
        """Generate a duration-based question."""
        movie = self.df[self.df['Duration (minutes)'].notna()].sample(1).iloc[0]
        correct_duration = int(movie['Duration (minutes)'])

        wrong_durations = [
            correct_duration + random.choice([-30, -20, -15, 15, 20, 30])
            for _ in range(3)
        ]
        wrong_durations = [max(60, d) for d in wrong_durations]

        options = [correct_duration] + wrong_durations
        random.shuffle(options)

        return {
            "question": f"How long is '{movie['Title']}' (in minutes)?",
            "options": [f"{opt} min" for opt in options],
            "correct": f"{correct_duration} min",
            "movie": movie['Title'],
            "category": "duration",
            "hint": f"Hint: It's a {movie['Genre'].split(',')[0]} film"
        }

    def check_answer(self, question: Dict, answer) -> Dict:
        """Check answer and update scores."""
        self.questions_asked += 1
        category = question.get("category", "general")

        is_correct = str(answer) == str(question['correct'])

        if category in self.category_scores:
            self.category_scores[category]["total"] += 1

        result = {
            "correct": is_correct,
            "correct_answer": question['correct'],
            "points_earned": 0
        }

        if is_correct:
            points = question.get("points", 1)
            self.score += points
            self.streak += 1
            self.max_streak = max(self.max_streak, self.streak)

            # Streak bonus
            if self.streak >= 3:
                bonus = self.streak
                self.score += bonus
                points += bonus
                result["streak_bonus"] = bonus

            if category in self.category_scores:
                self.category_scores[category]["correct"] += 1

            result["points_earned"] = points
        else:
            self.streak = 0

        return result

    def get_stats(self) -> Dict:
        """Get comprehensive quiz statistics."""
        accuracy = (self.score / max(self.questions_asked, 1)) * 100

        category_accuracy = {}
        for cat, stats in self.category_scores.items():
            if stats["total"] > 0:
                category_accuracy[cat] = round(stats["correct"] / stats["total"] * 100, 1)

        return {
            "score": self.score,
            "questions_asked": self.questions_asked,
            "accuracy": round(accuracy, 1),
            "current_streak": self.streak,
            "max_streak": self.max_streak,
            "category_accuracy": category_accuracy
        }

    def reset(self):
        """Reset quiz state."""
        self.score = 0
        self.questions_asked = 0
        self.streak = 0
        self.max_streak = 0
        for cat in self.category_scores:
            self.category_scores[cat] = {"correct": 0, "total": 0}


# ============================================================
# WATCHLIST MANAGER
# ============================================================

class WatchlistManager:
    """Manages user's movie watchlist and favorites."""

    def __init__(self):
        self.watchlist: List[Dict] = []
        self.favorites: List[Dict] = []
        self.watched: List[Dict] = []

    def add_to_watchlist(self, movie: Dict) -> bool:
        """Add movie to watchlist."""
        if not any(m['title'] == movie['title'] for m in self.watchlist):
            self.watchlist.append({**movie, "added_at": datetime.now().isoformat()})
            return True
        return False

    def remove_from_watchlist(self, title: str) -> bool:
        """Remove movie from watchlist."""
        for i, movie in enumerate(self.watchlist):
            if movie['title'] == title:
                self.watchlist.pop(i)
                return True
        return False

    def add_to_favorites(self, movie: Dict) -> bool:
        """Add movie to favorites."""
        if not any(m['title'] == movie['title'] for m in self.favorites):
            self.favorites.append({**movie, "added_at": datetime.now().isoformat()})
            return True
        return False

    def mark_as_watched(self, movie: Dict, rating: float = None) -> bool:
        """Mark movie as watched."""
        self.remove_from_watchlist(movie['title'])
        if not any(m['title'] == movie['title'] for m in self.watched):
            watched_entry = {
                **movie,
                "watched_at": datetime.now().isoformat(),
                "user_rating": rating
            }
            self.watched.append(watched_entry)
            return True
        return False

    def get_watchlist(self) -> List[Dict]:
        """Get current watchlist."""
        return self.watchlist

    def get_favorites(self) -> List[Dict]:
        """Get favorites list."""
        return self.favorites

    def get_watched(self) -> List[Dict]:
        """Get watched movies list."""
        return self.watched

    def get_stats(self) -> Dict:
        """Get watchlist statistics."""
        return {
            "watchlist_count": len(self.watchlist),
            "favorites_count": len(self.favorites),
            "watched_count": len(self.watched)
        }


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="IMDb Movie Chatbot - Multi-Agent AI",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced CSS styling
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #E50914 0%, #B20710 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Agent badge */
    .agent-badge {
        background: linear-gradient(135deg, #E50914 0%, #B20710 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Movie card */
    .movie-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .movie-card h4 {
        color: #212529;
        margin: 0 0 8px 0;
        font-size: 1rem;
    }
    .movie-card p {
        color: #6c757d;
        margin: 4px 0;
        font-size: 0.9rem;
    }

    /* Stat card */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stat-card h3 {
        margin: 0 0 8px 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .stat-card h2 {
        margin: 0;
        font-size: 2rem;
    }

    /* Quiz card */
    .quiz-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 15px 0;
    }
    .quiz-card h3 {
        color: #856404;
        margin: 0;
    }

    /* Feature card */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
        transition: all 0.3s;
    }
    .feature-card:hover {
        border-color: #E50914;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.1);
    }

    /* Quick query buttons */
    .quick-query-btn {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .quick-query-btn:hover {
        background: #E50914;
        color: white;
        border-color: #E50914;
    }

    /* Cache indicator */
    .cache-hit {
        background: #d4edda;
        color: #155724;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
    }

    /* Processing time */
    .processing-time {
        color: #6c757d;
        font-size: 0.8rem;
        font-style: italic;
    }

    /* Mood selector */
    .mood-btn {
        padding: 10px 20px;
        border-radius: 25px;
        border: 2px solid #dee2e6;
        background: white;
        cursor: pointer;
        transition: all 0.2s;
        margin: 5px;
    }
    .mood-btn:hover {
        border-color: #E50914;
        transform: scale(1.05);
    }
    .mood-btn.selected {
        background: #E50914;
        color: white;
        border-color: #E50914;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">🎬 IMDb Movie Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent AI Movie Discovery System | 10+ Specialized Agents | Voice Search | Mood-Based Recommendations</p>', unsafe_allow_html=True)

    # Initialize all components
    if "initialized" not in st.session_state:
        with st.spinner("🔄 Initializing AI agents and loading movie database..."):
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
                st.session_state.watchlist = WatchlistManager()
                st.session_state.messages = []
                st.session_state.posters = []
                st.session_state.initialized = True
                st.success("✅ All 10+ AI agents loaded successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error during initialization: {str(e)}")
                logger.error(f"Initialization error: {str(e)}")
                st.stop()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.radio(
            "Choose Feature:",
            [
                "💬 Chat with Agents",
                "🎭 Mood-Based Picks",
                "🔥 Trending Movies",
                "🎮 Movie Quiz",
                "📋 My Watchlist",
                "📊 Statistics"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Agent status
        st.markdown("### 🤖 Active Agents")
        agents_info = [
            ("🎭", "Genre Expert"),
            ("🎬", "Filmography Expert"),
            ("⭐", "Rating Analyst"),
            ("🔄", "Comparison Analyst"),
            ("💭", "Mood Curator"),
            ("🎯", "Similarity Finder"),
            ("📝", "Review Analyst"),
            ("🎲", "Trivia Master"),
            ("⏱️", "Runtime Advisor"),
            ("📅", "Era Explorer"),
            ("🔍", "General Assistant")
        ]

        for emoji, name in agents_info:
            st.markdown(f"<small>{emoji} {name}</small>", unsafe_allow_html=True)

        st.markdown("---")

        # Quick stats
        stats = st.session_state.orchestrator.get_stats()
        st.markdown(f"**Session Stats:**")
        st.markdown(f"- Queries: {stats['total_queries']}")
        st.markdown(f"- Cache Hit Rate: {stats['cache_stats']['hit_rate_percent']}%")

        st.markdown("---")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.posters = []
            st.session_state.orchestrator.clear_history()
            st.rerun()

    # Main content routing
    if page == "💬 Chat with Agents":
        render_chat_page()
    elif page == "🎭 Mood-Based Picks":
        render_mood_page()
    elif page == "🔥 Trending Movies":
        render_trending_page()
    elif page == "🎮 Movie Quiz":
        render_quiz_page()
    elif page == "📋 My Watchlist":
        render_watchlist_page()
    elif page == "📊 Statistics":
        render_stats_page()


def render_chat_page():
    """Render the main chat interface with enhanced features."""
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown("### 💬 Chat with AI Movie Agents")

        # Quick query suggestions
        st.markdown("**Quick Queries:**")
        quick_queries = [
            "🎭 Recommend comedy movies",
            "🎬 Movies by Christopher Nolan",
            "⭐ Top rated documentaries",
            "🔄 Compare Inception vs Interstellar",
            "💡 Movies similar to The Dark Knight",
            "📅 Best movies from the 90s",
            "🎲 Trivia about Pulp Fiction"
        ]

        cols = st.columns(4)
        for i, query in enumerate(quick_queries):
            if cols[i % 4].button(query, key=f"quick_{i}", use_container_width=True):
                # Remove emoji prefix for processing
                clean_query = query.split(' ', 1)[1] if ' ' in query else query
                st.session_state.pending_query = clean_query

        st.markdown("---")

        # Voice search section
        with st.expander("🎤 Voice Search (Beta)", expanded=False):
            st.markdown("""
            **Voice Search Instructions:**
            1. Click the microphone button below
            2. Speak your movie query clearly
            3. The text will appear in the input box

            *Note: Voice recognition uses your browser's built-in speech API.*
            """)

            # Voice input simulation using text input with speech-to-text hint
            voice_input = st.text_input(
                "Voice input will appear here:",
                placeholder="Speak or type your query...",
                key="voice_input"
            )

            if voice_input:
                st.session_state.pending_query = voice_input

        # Chat messages display
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        if "agent" in msg:
                            st.markdown(f'<span class="agent-badge">🤖 {msg["agent"]}</span>', unsafe_allow_html=True)
                        if msg.get("from_cache"):
                            st.markdown('<span class="cache-hit">⚡ From Cache</span>', unsafe_allow_html=True)
                        if "processing_time" in msg:
                            st.markdown(f'<span class="processing-time">Processed in {msg["processing_time"]}s</span>', unsafe_allow_html=True)
                    st.markdown(msg["content"])

        # Handle pending query from quick buttons
        if "pending_query" in st.session_state:
            query = st.session_state.pending_query
            del st.session_state.pending_query
            process_chat_query(query)

        # Chat input
        if prompt := st.chat_input("Ask me anything about movies..."):
            process_chat_query(prompt)

    with col2:
        st.markdown("### 🖼️ Related Movies")

        if st.session_state.posters:
            for idx, poster in enumerate(st.session_state.posters[:5]):
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{poster.get('title', 'Movie')[:25]}{'...' if len(poster.get('title', '')) > 25 else ''}</h4>
                        <p>📅 {poster.get('year', 'N/A')} | ⭐ {poster.get('rating', 'N/A')}</p>
                        <p style="font-size:0.8rem">{poster.get('genre', '')[:30]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if poster.get('poster_url'):
                        try:
                            st.image(poster['poster_url'], width=150)
                        except:
                            pass

                    # Add to watchlist button
                    if st.button(f"➕ Add to Watchlist", key=f"add_poster_{idx}_{poster.get('title', '')[:10]}"):
                        if st.session_state.watchlist.add_to_watchlist(poster):
                            st.success("Added to watchlist!")
                        else:
                            st.info("Already in watchlist")

                    st.markdown("---")
        else:
            st.info("🎬 Movie posters will appear here after your queries!")

            # Show random recommendations
            st.markdown("**Try asking about:**")
            st.markdown("- Genre recommendations")
            st.markdown("- Actor filmographies")
            st.markdown("- Movie comparisons")
            st.markdown("- Rating-based searches")


def process_chat_query(query: str):
    """Process a chat query through the multi-agent system."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 AI agents analyzing your query..."):
            result = st.session_state.orchestrator.process(query)

            # Display agent info
            st.markdown(f'<span class="agent-badge">🤖 {result["agent"]}</span>', unsafe_allow_html=True)

            if result.get("from_cache"):
                st.markdown('<span class="cache-hit">⚡ Cached Response</span>', unsafe_allow_html=True)

            if "processing_time" in result:
                st.markdown(f'<span class="processing-time">Processed in {result["processing_time"]}s</span>', unsafe_allow_html=True)

            st.markdown(result["answer"])

            # Store message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "agent": result["agent"],
                "processing_time": result.get("processing_time"),
                "from_cache": result.get("from_cache", False)
            })

            # Update posters
            st.session_state.posters = result.get("posters", [])


def render_mood_page():
    """Render mood-based movie recommendations page."""
    st.markdown("### 🎭 Mood-Based Movie Recommendations")
    st.markdown("*Tell us how you're feeling, and we'll find the perfect movie!*")

    # Mood selection
    moods = [
        ("😊", "Happy", "Uplifting comedies and feel-good films"),
        ("😢", "Sad", "Emotional dramas for a good cry"),
        ("🤩", "Excited", "Action-packed thrillers and adventures"),
        ("😱", "Scared", "Horror and suspense films"),
        ("💕", "Romantic", "Love stories and romantic comedies"),
        ("🤔", "Thoughtful", "Documentaries and thought-provoking dramas"),
        ("😌", "Relaxed", "Easy-watching comfort films"),
        ("✨", "Inspired", "Motivational biopics and sports films")
    ]

    st.markdown("**How are you feeling today?**")

    cols = st.columns(4)
    selected_mood = None

    for i, (emoji, mood, desc) in enumerate(moods):
        with cols[i % 4]:
            if st.button(f"{emoji}\n{mood}", key=f"mood_{mood}", use_container_width=True, help=desc):
                selected_mood = mood.lower()
                st.session_state.selected_mood = selected_mood

    # If mood was previously selected
    if "selected_mood" in st.session_state:
        selected_mood = st.session_state.selected_mood

    if selected_mood:
        st.markdown("---")
        st.markdown(f"### Movies for when you're feeling **{selected_mood.title()}**")

        # Get mood-based genres
        genres = MOOD_GENRE_MAPPING.get(selected_mood, ["Drama"])

        with st.spinner(f"Finding perfect {selected_mood} movies..."):
            # Process through mood agent
            query = f"I'm feeling {selected_mood}, recommend me some movies"
            result = st.session_state.orchestrator.process(query)

            st.markdown(f'<span class="agent-badge">🤖 {result["agent"]}</span>', unsafe_allow_html=True)
            st.markdown(result["answer"])

            # Display movie cards
            if result.get("posters"):
                st.markdown("---")
                st.markdown("### 🎬 Top Picks for Your Mood")

                cols = st.columns(4)
                for i, poster in enumerate(result["posters"][:8]):
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{poster.get('title', 'Movie')[:20]}</h4>
                            <p>⭐ {poster.get('rating', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if poster.get('poster_url'):
                            try:
                                st.image(poster['poster_url'], width=130)
                            except:
                                pass


def render_trending_page():
    """Render trending movies page with multiple categories."""
    st.markdown("### 🔥 Trending & Curated Movies")

    tabs = st.tabs([
        "⭐ Top Rated",
        "🆕 Recent Hits",
        "💎 Hidden Gems",
        "🎭 By Genre",
        "📅 By Decade",
        "⏱️ By Duration"
    ])

    with tabs[0]:
        st.markdown("#### Top Rated Movies of All Time")
        movies = st.session_state.trending.get_top_rated(12)
        display_movie_grid(movies)

    with tabs[1]:
        st.markdown("#### Recent High-Rated Movies (2015+)")
        movies = st.session_state.trending.get_recent(12)
        display_movie_grid(movies)

    with tabs[2]:
        st.markdown("#### Hidden Gems")
        st.markdown("*Highly-rated movies you might have missed*")
        if st.button("🔄 Discover New Gems", key="refresh_gems"):
            st.rerun()
        movies = st.session_state.trending.get_hidden_gems(12)
        display_movie_grid(movies)

    with tabs[3]:
        st.markdown("#### Top Movies by Genre")
        genre = st.selectbox(
            "Select Genre:",
            ["Action", "Comedy", "Drama", "Documentary", "Horror", "Romance",
             "Thriller", "Animation", "Sci-Fi", "Adventure", "Crime", "Fantasy"],
            key="genre_select"
        )
        movies = st.session_state.trending.get_by_genre(genre, 12)
        display_movie_grid(movies)

    with tabs[4]:
        st.markdown("#### Top Movies by Decade")
        decade = st.selectbox(
            "Select Decade:",
            [2020, 2010, 2000, 1990, 1980, 1970],
            format_func=lambda x: f"{x}s",
            key="decade_select"
        )
        movies = st.session_state.trending.get_by_decade(decade, 12)
        display_movie_grid(movies)

    with tabs[5]:
        st.markdown("#### Movies by Runtime")
        duration_option = st.radio(
            "Choose duration:",
            ["Short (< 100 min)", "Standard (90-130 min)", "Epic (150+ min)"],
            horizontal=True,
            key="duration_radio"
        )

        if "Short" in duration_option:
            movies = st.session_state.trending.get_short_movies(100, 12)
        elif "Epic" in duration_option:
            movies = st.session_state.trending.get_epic_movies(150, 12)
        else:
            movies = st.session_state.trending.get_top_rated(12)

        display_movie_grid(movies)


def display_movie_grid(movies: List[Dict], cols_count: int = 4):
    """Display movies in a responsive grid format."""
    cols = st.columns(cols_count)

    for i, movie in enumerate(movies):
        with cols[i % cols_count]:
            st.markdown(f"""
            <div class="movie-card">
                <h4>{movie['title'][:22]}{'...' if len(movie['title']) > 22 else ''}</h4>
                <p>📅 {movie['year']} | ⭐ {movie['rating']}</p>
                <p style="font-size:0.8rem">🎬 {movie['director'][:20] if movie['director'] != 'Unknown' else 'Unknown Director'}</p>
                <p style="font-size:0.75rem; color: #888;">{movie['genre'][:35]}{'...' if len(movie.get('genre', '')) > 35 else ''}</p>
            </div>
            """, unsafe_allow_html=True)

            if movie.get('poster_url'):
                try:
                    st.image(movie['poster_url'], width=130)
                except:
                    pass

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕", key=f"add_trend_{movie['title'][:10]}_{i}", help="Add to Watchlist"):
                    if st.session_state.watchlist.add_to_watchlist(movie):
                        st.success("Added!")
            with col2:
                if st.button("❤️", key=f"fav_trend_{movie['title'][:10]}_{i}", help="Add to Favorites"):
                    if st.session_state.watchlist.add_to_favorites(movie):
                        st.success("Favorited!")


def render_quiz_page():
    """Render the enhanced movie quiz page."""
    st.markdown("### 🎮 Movie Trivia Challenge")

    quiz = st.session_state.quiz
    stats = quiz.get_stats()

    # Score display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🏆 Score</h3>
            <h2>{stats['score']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>❓ Questions</h3>
            <h2>{stats['questions_asked']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🎯 Accuracy</h3>
            <h2>{stats['accuracy']}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        streak_color = "#28a745" if stats['current_streak'] >= 3 else "#667eea"
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, {streak_color} 0%, #764ba2 100%);">
            <h3>🔥 Streak</h3>
            <h2>{stats['current_streak']}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Difficulty selection
    difficulty = st.radio(
        "Select Difficulty:",
        ["easy", "medium", "hard"],
        horizontal=True,
        format_func=lambda x: f"{x.title()} ({quiz.DIFFICULTY_LEVELS[x]['points']} pts)"
    )

    st.markdown("---")

    # Generate or get current question
    if "current_question" not in st.session_state or st.session_state.current_question is None:
        st.session_state.current_question = quiz.generate_question(difficulty)

    question = st.session_state.current_question

    # Display question
    st.markdown(f"""
    <div class="quiz-card">
        <h3>❓ {question['question']}</h3>
        <p><small>Category: {question['category'].title()} | Points: {question['points']}</small></p>
    </div>
    """, unsafe_allow_html=True)

    # Show hint option
    with st.expander("💡 Need a hint?"):
        st.markdown(question.get('hint', 'No hint available'))

    # Answer options
    st.markdown("**Choose your answer:**")
    cols = st.columns(2)

    for i, option in enumerate(question['options']):
        if cols[i % 2].button(
            str(option),
            key=f"quiz_option_{i}",
            use_container_width=True
        ):
            result = quiz.check_answer(question, option)

            if result['correct']:
                st.success(f"✅ Correct! +{result['points_earned']} points")
                if result.get('streak_bonus'):
                    st.info(f"🔥 Streak bonus: +{result['streak_bonus']} points!")
                st.balloons()
            else:
                st.error(f"❌ Wrong! The correct answer was: {result['correct_answer']}")

            st.session_state.current_question = None
            time.sleep(1.5)
            st.rerun()

    # Control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⏭️ Skip Question", use_container_width=True):
            st.session_state.current_question = None
            st.rerun()

    with col2:
        if st.button("🔄 New Question", use_container_width=True):
            st.session_state.current_question = quiz.generate_question(difficulty)
            st.rerun()

    with col3:
        if st.button("🗑️ Reset Quiz", use_container_width=True):
            quiz.reset()
            st.session_state.current_question = None
            st.rerun()

    # Category performance
    if stats['category_accuracy']:
        st.markdown("---")
        st.markdown("### 📊 Performance by Category")

        cols = st.columns(len(stats['category_accuracy']))
        for i, (cat, accuracy) in enumerate(stats['category_accuracy'].items()):
            with cols[i]:
                color = "#28a745" if accuracy >= 70 else "#ffc107" if accuracy >= 50 else "#dc3545"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 8px; background: {color}20;">
                    <strong>{cat.title()}</strong><br>
                    <span style="font-size: 1.5rem; color: {color};">{accuracy}%</span>
                </div>
                """, unsafe_allow_html=True)


def render_watchlist_page():
    """Render the watchlist management page."""
    st.markdown("### 📋 My Movie Lists")

    watchlist = st.session_state.watchlist
    stats = watchlist.get_stats()

    # Stats overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>📋 Watchlist</h3>
            <h2>{stats['watchlist_count']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>❤️ Favorites</h3>
            <h2>{stats['favorites_count']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>✅ Watched</h3>
            <h2>{stats['watched_count']}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    tabs = st.tabs(["📋 Watchlist", "❤️ Favorites", "✅ Watched"])

    with tabs[0]:
        st.markdown("#### Movies to Watch")
        movies = watchlist.get_watchlist()

        if movies:
            for i, movie in enumerate(movies):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"""
                    **{movie['title']}** ({movie['year']}) - ⭐ {movie['rating']}
                    <br><small>{movie.get('genre', 'Unknown genre')}</small>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("✅ Watched", key=f"watchlist_watched_{i}_{movie['title'][:8]}"):
                        watchlist.mark_as_watched(movie)
                        st.rerun()

                with col3:
                    if st.button("🗑️ Remove", key=f"watchlist_remove_{i}_{movie['title'][:8]}"):
                        watchlist.remove_from_watchlist(movie['title'])
                        st.rerun()

                st.markdown("---")
        else:
            st.info("Your watchlist is empty. Add movies from the chat or trending pages!")

    with tabs[1]:
        st.markdown("#### Favorite Movies")
        favorites = watchlist.get_favorites()

        if favorites:
            display_movie_grid(favorites, cols_count=4)
        else:
            st.info("No favorites yet. Mark movies as favorites to see them here!")

    with tabs[2]:
        st.markdown("#### Movies You've Watched")
        watched = watchlist.get_watched()

        if watched:
            for movie in watched:
                st.markdown(f"""
                - **{movie['title']}** ({movie['year']}) - ⭐ {movie['rating']}
                  <br><small>Watched: {movie.get('watched_at', 'Unknown')[:10]}</small>
                """, unsafe_allow_html=True)
        else:
            st.info("No watched movies yet. Mark movies as watched from your watchlist!")


def render_stats_page():
    """Render comprehensive statistics page."""
    st.markdown("### 📊 Session Statistics & Analytics")

    stats = st.session_state.orchestrator.get_stats()
    df = st.session_state.df

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>💬 Queries</h3>
            <h2>{stats['total_queries']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🎬 Movies</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🤖 Agents</h3>
            <h2>{stats['agents_available']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h3>⚡ Cache Rate</h3>
            <h2>{stats['cache_stats']['hit_rate_percent']}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Agent usage
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Agent Usage Distribution")

        if stats['agent_usage']:
            agent_df = pd.DataFrame([
                {"Agent": agent, "Queries": count}
                for agent, count in stats['agent_usage'].items()
            ])
            st.bar_chart(agent_df.set_index("Agent"))
        else:
            st.info("Start chatting to see agent usage statistics!")

    with col2:
        st.markdown("### 📈 Cache Performance")

        cache_stats = stats['cache_stats']
        cache_df = pd.DataFrame([
            {"Type": "Exact Hits", "Count": cache_stats['exact_hits']},
            {"Type": "Semantic Hits", "Count": cache_stats['semantic_hits']},
            {"Type": "Misses", "Count": cache_stats['misses']}
        ])
        st.bar_chart(cache_df.set_index("Type"))

    st.markdown("---")

    # Dataset analytics
    st.markdown("### 📊 Dataset Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Genre Distribution")
        # Parse genres and count
        genre_counts = {}
        for genres in df['Genre'].dropna():
            for genre in str(genres).split(','):
                genre = genre.strip()
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        st.bar_chart(pd.DataFrame({"Count": top_genres}))

    with col2:
        st.markdown("#### Rating Distribution")
        rating_bins = pd.cut(df['IMDb Rating'], bins=[0, 5, 6, 7, 8, 9, 10])
        rating_counts = rating_bins.value_counts().sort_index()
        st.bar_chart(rating_counts)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Movies by Decade")
        df['Decade'] = (df['Year'] // 10) * 10
        decade_counts = df['Decade'].value_counts().sort_index()
        st.line_chart(decade_counts)

    with col2:
        st.markdown("#### Top Directors (by # of movies)")
        director_counts = df['Director'].value_counts().head(10)
        st.bar_chart(director_counts)

    st.markdown("---")

    # Detailed stats table
    st.markdown("### 📋 Detailed Session Log")

    if stats['agent_usage']:
        log_data = []
        for agent, count in stats['agent_usage'].items():
            log_data.append({
                "Agent": agent,
                "Queries Handled": count,
                "Percentage": f"{count/max(stats['total_queries'], 1)*100:.1f}%"
            })

        st.dataframe(pd.DataFrame(log_data), use_container_width=True)
    else:
        st.info("No queries processed yet. Start chatting to generate statistics!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
