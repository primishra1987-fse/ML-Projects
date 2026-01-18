# ML-Projects - IMDb Movie Chatbot

An AI-powered conversational movie recommendation system built with LangChain, OpenAI, and FAISS vector search.

## Features

- **Natural Language Queries**: Ask about movies in plain English
- **Semantic Search**: Uses FAISS vector store for intelligent movie matching
- **Agentic Architecture**: 7 specialized tools for different query types
- **Conversation Memory**: Remembers context across multiple turns
- **Interactive UI**: Gradio and Streamlit chat interfaces
- **Response Streaming**: Real-time token-by-token response display
- **Rate Limiting**: Built-in protection against API abuse (20 req/min)
- **Comprehensive Logging**: Full request/response logging to file
- **Movie Posters**: Visual display of recommended movie posters
- **Session Statistics**: Track response times and usage metrics
- **Query Caching**: Exact match + semantic similarity caching for faster responses

## Architecture

```
User Query → Input Validation → Agent Orchestrator → Tool Selection
                                       ↓
                              [7 Specialized Tools]
                                       ↓
                    FAISS Vector Store ← OpenAI Embeddings
                                       ↓
                              GPT-4o-mini Response
                                       ↓
                              Formatted Output → User
```

## Tools Available

| Tool | Description |
|------|-------------|
| `search_movies_by_query` | Semantic search across all movies |
| `get_movie_details` | Get detailed info about a specific movie |
| `recommend_movies_by_genre` | Genre-based recommendations |
| `find_movies_by_actor` | Find movies featuring an actor |
| `find_movies_by_director` | Find movies by director |
| `get_top_rated_movies` | Filter by IMDb rating |
| `compare_movies` | Compare two movies |

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

### Jupyter Notebook

1. Open `Student_Template.ipynb`
2. Run all cells to initialize the chatbot
3. Launch the Gradio UI from the notebook

### Example Queries

```
"Recommend some comedy movies"
"What movies has Tom Hanks starred in?"
"Find movies directed by Steven Spielberg"
"Show movies rated above 8.0"
"Compare documentary and biography genres"
```

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
├── Student_Template.ipynb       # Main implementation notebook
├── Student_Template_Output.ipynb # Executed notebook with outputs
├── streamlit_app.py             # Streamlit web interface
├── test_chatbot.py              # Test suite
├── IMDb_Dataset (1).csv         # Movie dataset
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## Testing

The test suite includes 15 tests covering:
- Basic functionality (genre, actor, director searches)
- Complex queries (multi-criteria, comparisons)
- Edge cases (empty input, misspelled queries)

All tests pass with 100% success rate.

## Performance

- **Vector Store Caching**: FAISS index is saved locally after first run
- **Embedding Model**: `text-embedding-3-small` (fast, cost-effective)
- **LLM Model**: `gpt-4o-mini` (balanced speed/quality)

## License

This project is for educational purposes.
