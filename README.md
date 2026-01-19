# IMDb Movie Chatbot - Enhanced Multi-Agent System

An AI-powered conversational movie recommendation system built with LangChain, OpenAI, and FAISS vector search. Features an advanced multi-agent architecture with 11 specialized agents for intelligent query routing.

## Features

### Core Capabilities
- **Natural Language Queries**: Ask about movies in plain English
- **Semantic Search**: Uses FAISS vector store for intelligent movie matching
- **Multi-Agent Architecture**: 11 specialized agents for different query types
- **Conversation Memory**: Remembers context across multiple turns
- **Interactive UI**: Gradio and Streamlit chat interfaces

### Advanced Features
- **Mood-Based Recommendations**: Get movie suggestions based on your current mood
- **Movie Trivia & Quizzes**: Interactive quiz system with 6 question categories
- **Similar Movie Finder**: Discover movies similar to ones you love
- **Duration-Based Filtering**: Find movies that fit your available time
- **Era/Decade Explorer**: Discover movies from specific time periods
- **Movie Comparisons**: Compare movies, directors, or genres
- **Review Sentiment Analysis**: Understand critical reception

### Technical Features
- **Response Streaming**: Real-time token-by-token response display
- **Rate Limiting**: Built-in protection against API abuse (20 req/min)
- **Comprehensive Logging**: Full request/response logging to file
- **Movie Posters**: Visual display of recommended movie posters
- **Session Statistics**: Track response times and usage metrics
- **Query Caching**: Exact match + semantic similarity caching for faster responses

## Multi-Agent Architecture

The chatbot uses an intelligent query routing system that automatically directs your questions to the most appropriate specialized agent:

```
User Query --> Query Classifier --> Agent Router --> Specialized Agent
                    |                    |                   |
              Pattern Match         Confidence          Execute Query
              + Confidence           Scoring             with Tools
                    |                    |                   |
                    v                    v                   v
            Classify Intent      Route to Best        Return Results
                               Agent Match               to User
```

### Specialized Agents

| # | Agent | Specialty | Example Questions |
|---|-------|-----------|-------------------|
| 1 | **Genre Expert** | Genre-based recommendations | "Recommend comedy movies", "Find thriller films" |
| 2 | **Filmography Expert** | Actor & Director searches | "Movies with Tom Hanks", "Films by Nolan" |
| 3 | **Rating Analyst** | Rating-based filtering | "Top rated movies above 8.0", "Highest rated films" |
| 4 | **Comparison Analyst** | Movie/genre comparisons | "Compare Titanic and Avatar", "Thriller vs Horror" |
| 5 | **Mood Curator** | Mood-based recommendations | "I'm feeling happy", "Something for a sad mood" |
| 6 | **Trivia Master** | Movie trivia & quizzes | "Tell me a fact about Inception", "Movie quiz" |
| 7 | **Review Analyst** | Review sentiment analysis | "What do critics think about Parasite?" |
| 8 | **Similarity Finder** | Similar movie recommendations | "Movies like The Matrix", "Similar to Inception" |
| 9 | **Runtime Advisor** | Duration-based filtering | "Movies under 90 minutes", "Long epic films" |
| 10 | **Era Explorer** | Year/decade searches | "Best movies of 2020", "90s classics" |
| 11 | **General Assistant** | General queries & fallback | "What's a good movie?", "Recommend something" |

## Example Questions You Can Ask

### By Genre
```
"Recommend some comedy movies"
"Find action thriller films"
"Show me horror movies"
"What are the best drama films?"
"I want to watch a documentary"
"Find romantic comedies"
```

### By Actor/Director
```
"What movies has Leonardo DiCaprio starred in?"
"Find movies directed by Christopher Nolan"
"Show me Tom Hanks films"
"Movies by Quentin Tarantino"
"Films starring Meryl Streep"
```

### By Rating
```
"Show movies rated above 8.5"
"What are the highest rated movies?"
"Find top-rated documentaries"
"Movies with perfect scores"
```

### By Mood
```
"I'm feeling happy, what should I watch?"
"Something for a romantic evening"
"I want to feel excited and adventurous"
"Feeling sad, need a movie to cheer me up"
"I'm in the mood for something scary"
"Looking for something thought-provoking"
```

### By Duration
```
"Movies under 90 minutes"
"I have 2 hours, what can I watch?"
"Show me epic long movies over 3 hours"
"Quick movies for a lunch break"
```

### By Year/Era
```
"Best movies of 2020"
"Classic films from the 1990s"
"Recent movies from 2023"
"Golden age Hollywood films"
"Movies from the 80s"
```

### Similar Movies
```
"Movies similar to Inception"
"I liked The Dark Knight, suggest similar movies"
"What's like The Matrix?"
"Recommend movies like Pulp Fiction"
```

### Comparisons
```
"Compare The Godfather and Goodfellas"
"What's the difference between thriller and horror?"
"Compare Nolan and Spielberg films"
"Titanic vs Avatar - which is better?"
```

### Trivia & Facts
```
"Tell me an interesting fact about Inception"
"Movie trivia about Star Wars"
"Behind the scenes info about The Dark Knight"
"Give me a movie quiz question"
```

### Review Analysis
```
"What do critics think about Parasite?"
"Is Joker well-received?"
"Why is Shawshank Redemption rated so highly?"
```

### Complex Queries
```
"Find a short comedy from 2020 with good ratings"
"Adventure movies from the 90s with Tom Hanks"
"Highly rated documentaries about music"
"Thriller movies under 2 hours rated above 7.5"
```

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/primishra1987-fse/ML-Projects.git
   cd ML-Projects
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Usage

### Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```
Access at `http://localhost:8501`

**Features in Streamlit UI:**
- Chat interface with conversation history
- Mood-based movie picks page
- Trending movies section
- Interactive movie quiz
- Watchlist management
- Session statistics

### Jupyter Notebook

1. Open `Student_Template.ipynb`
2. Run all cells to initialize the chatbot
3. Launch the Gradio UI from the notebook

### Google Colab

1. Open `IMDb_Movie_Chatbot_Colab.ipynb` in Google Colab
2. Mount your Google Drive
3. Set your OpenAI API key
4. Run all cells to start the chatbot

## Dataset

The IMDb dataset contains **3,173 movies** with the following fields:

| Field | Description |
|-------|-------------|
| Title | Movie title |
| IMDb Rating | Rating out of 10 |
| Year | Release year |
| Certificates | Age rating (PG, R, etc.) |
| Genre | Movie genres |
| Director | Director name |
| Star Cast | Main actors |
| MetaScore | Metacritic score |
| Poster-src | Poster image URL |
| Duration (minutes) | Runtime |

## Project Structure

```
ML-Projects/
├── Student_Template.ipynb           # Main implementation notebook (Gradio UI)
├── Student_Template_Output.ipynb    # Executed notebook with outputs
├── IMDb_Movie_Chatbot_Colab.ipynb   # Google Colab version
├── streamlit_app.py                 # Streamlit web interface
├── test_chatbot.py                  # Comprehensive test suite
├── IMDb_Dataset (1).csv             # Movie dataset
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

## Testing

The test suite includes **70+ test cases** covering all 11 specialized agents:

### Test Categories
- Basic Functionality (7 tests)
- Multi-Agent Routing (11 tests)
- Mood-Based Recommendations (8 tests)
- Duration/Runtime (5 tests)
- Era/Year-Based (5 tests)
- Similar Movies (4 tests)
- Comparisons (4 tests)
- Trivia/Facts (4 tests)
- Review/Sentiment (4 tests)
- Complex Queries (5 tests)
- Edge Cases (10 tests)
- Tool-Specific (7 tests)

### Running Tests

```bash
# Run unit tests (no API required)
python -m pytest test_chatbot.py -v

# Or use unittest
python -m unittest test_chatbot -v

# Print test cases for manual testing
python -c "from test_chatbot import print_test_cases; print_test_cases()"

# View agent coverage
python -c "from test_chatbot import print_agent_coverage; print_agent_coverage()"
```

## Performance

- **Vector Store Caching**: FAISS index is saved locally after first run
- **Query Caching**: Dual-layer cache (exact match + semantic similarity)
- **Rate Limiting**: 20 requests per minute with sliding window
- **Embedding Model**: `text-embedding-3-small` (fast, cost-effective)
- **LLM Model**: `gpt-4o-mini` (balanced speed/quality)

## Technical Implementation

### Query Classification
The `EnhancedQueryClassifier` uses pattern matching with confidence scoring to route queries to the appropriate agent:

```python
AGENT_PATTERNS = {
    "genre": [r'\b(comedy|drama|action|horror|thriller|romance)\b'],
    "mood": [r'\b(feeling|mood|happy|sad|excited|scared)\b'],
    "actor": [r'\b(starring|actor|actress|with)\b.*\b(movie|film)\b'],
    "director": [r'\b(directed|director|by)\b'],
    "rating": [r'\b(rated|rating|score|top)\b'],
    # ... more patterns
}
```

### Mood-to-Genre Mapping
```python
MOOD_GENRE_MAPPING = {
    "happy": ["Comedy", "Animation", "Family"],
    "sad": ["Drama", "Romance"],
    "excited": ["Action", "Adventure", "Thriller"],
    "romantic": ["Romance", "Drama"],
    "scared": ["Horror", "Thriller"],
    "curious": ["Documentary", "Mystery"],
    "nostalgic": ["Drama", "Family"],
    "relaxed": ["Comedy", "Animation"],
    "adventurous": ["Adventure", "Action", "Sci-Fi"],
    "thoughtful": ["Drama", "Biography", "Documentary"]
}
```

### Conversation Memory
The system maintains context across turns by tracking:
- Last 10 conversation exchanges
- Referenced entities (movies, actors, directors)
- User preferences and mood

## Requirements

```
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
langchain-text-splitters>=0.0.1
faiss-cpu>=1.7.4
pandas>=2.0.0
numpy>=1.24.0
gradio>=4.0.0
streamlit>=1.28.0
python-dotenv>=1.0.0
openai>=1.0.0
```

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- IMDb for the movie dataset
- OpenAI for GPT-4o-mini and embeddings
- LangChain for the agent framework
- FAISS for efficient vector search
