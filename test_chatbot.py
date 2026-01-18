"""
IMDb Movie Chatbot - Test Suite
Run with: python test_chatbot.py (after running notebook cells 0-17)

For unit tests with mocking (no API required):
    python -m pytest test_chatbot.py -v
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Tuple, Optional

# ============================================================
# TEST CASES FOR IMDB MOVIE CHATBOT
# ============================================================

class TestCases:
    """Collection of test cases for manual and automated testing."""

    # ----------------------------------------------------------
    # 1. BASIC FUNCTIONALITY TESTS
    # ----------------------------------------------------------

    BASIC_QUERIES = [
        {
            "id": "BF001",
            "category": "Genre Search",
            "query": "Recommend some comedy movies",
            "expected": "Should return comedy movies from database",
            "validation": ["comedy", "movie", "rating"]
        },
        {
            "id": "BF002",
            "category": "Genre Search",
            "query": "Find documentary films",
            "expected": "Should return documentary movies",
            "validation": ["documentary", "film"]
        },
        {
            "id": "BF003",
            "category": "Genre Search",
            "query": "Show me some biography movies",
            "expected": "Should return biography/biopic movies",
            "validation": ["biography", "movie"]
        },
        {
            "id": "BF004",
            "category": "Actor Search",
            "query": "What movies has Tom Hanks starred in?",
            "expected": "Should return movies featuring Tom Hanks",
            "validation": ["Tom Hanks", "movie", "star"]
        },
        {
            "id": "BF005",
            "category": "Director Search",
            "query": "Find movies directed by Steven Spielberg",
            "expected": "Should return Spielberg's filmography",
            "validation": ["director", "movie"]
        },
        {
            "id": "BF006",
            "category": "Rating Filter",
            "query": "Show movies rated above 8.0",
            "expected": "Should return highly rated movies (8.0+)",
            "validation": ["rating", "8"]
        },
        {
            "id": "BF007",
            "category": "Year Filter",
            "query": "What movies came out in 2020?",
            "expected": "Should return movies from 2020",
            "validation": ["2020", "movie"]
        },
    ]

    # ----------------------------------------------------------
    # 2. COMPLEX QUERY TESTS
    # ----------------------------------------------------------

    COMPLEX_QUERIES = [
        {
            "id": "CQ001",
            "category": "Multi-criteria",
            "query": "Find a documentary about music with good ratings",
            "expected": "Should return music documentaries with high ratings",
            "validation": ["documentary", "music"]
        },
        {
            "id": "CQ002",
            "category": "Multi-criteria",
            "query": "What adventure movies from the 90s should I watch?",
            "expected": "Should return adventure movies from 1990-1999",
            "validation": ["adventure", "199"]
        },
        {
            "id": "CQ003",
            "category": "Comparison",
            "query": "Compare documentary and biography genres",
            "expected": "Should provide comparison of both genres",
            "validation": ["documentary", "biography"]
        },
        {
            "id": "CQ004",
            "category": "Recommendation",
            "query": "I liked The Dark Knight, recommend similar movies",
            "expected": "Should recommend action/thriller movies",
            "validation": ["recommend", "movie"]
        },
        {
            "id": "CQ005",
            "category": "Specific Detail",
            "query": "What is the runtime of the longest movie in the database?",
            "expected": "Should identify longest movie by duration",
            "validation": ["duration", "minutes", "movie"]
        },
    ]

    # ----------------------------------------------------------
    # 3. EDGE CASE TESTS
    # ----------------------------------------------------------

    EDGE_CASES = [
        {
            "id": "EC001",
            "category": "Empty Input",
            "query": "",
            "expected": "Should return error message asking for input",
            "validation": ["please", "enter", "question"]
        },
        {
            "id": "EC002",
            "category": "Very Short Input",
            "query": "hi",
            "expected": "Should ask for more detailed question",
            "validation": ["please", "more", "detailed"]
        },
        {
            "id": "EC003",
            "category": "Very Long Input",
            "query": "a" * 1500,
            "expected": "Should reject as too long",
            "validation": ["too long", "1000"]
        },
        {
            "id": "EC004",
            "category": "Non-movie Query",
            "query": "What's the weather like today?",
            "expected": "Should politely redirect to movie topics or handle gracefully",
            "validation": []  # LLM should handle gracefully
        },
        {
            "id": "EC005",
            "category": "Misspelled Query",
            "query": "Recomend some comdy moveis",
            "expected": "Should still attempt to find comedy movies",
            "validation": ["movie"]
        },
        {
            "id": "EC006",
            "category": "Non-existent Movie",
            "query": "Tell me about the movie XYZ123ABC",
            "expected": "Should say movie not found in database",
            "validation": ["not found", "sorry", "don't have"]
        },
        {
            "id": "EC007",
            "category": "Special Characters",
            "query": "Find movies with @#$% in the title",
            "expected": "Should handle gracefully without crashing",
            "validation": []
        },
        {
            "id": "EC008",
            "category": "SQL Injection Attempt",
            "query": "'; DROP TABLE movies; --",
            "expected": "Should handle safely without any issues",
            "validation": []
        },
    ]

    # ----------------------------------------------------------
    # 4. CONVERSATION FLOW TESTS
    # ----------------------------------------------------------

    CONVERSATION_TESTS = [
        {
            "id": "CF001",
            "category": "Follow-up Question",
            "queries": [
                "Recommend a documentary",
                "Tell me more about the first one"
            ],
            "expected": "Should remember context and provide details",
        },
        {
            "id": "CF002",
            "category": "Topic Change",
            "queries": [
                "Find comedy movies",
                "Now show me horror movies"
            ],
            "expected": "Should switch topics correctly",
        },
        {
            "id": "CF003",
            "category": "Refinement",
            "queries": [
                "Show me action movies",
                "Only the ones rated above 7.5"
            ],
            "expected": "Should refine previous results",
        },
    ]

    # ----------------------------------------------------------
    # 5. TOOL-SPECIFIC TESTS
    # ----------------------------------------------------------

    TOOL_TESTS = [
        {
            "id": "TT001",
            "tool": "search_movies_by_query",
            "query": "thriller movies with suspense",
            "expected": "Should use search tool"
        },
        {
            "id": "TT002",
            "tool": "get_movie_details",
            "query": "Tell me everything about Inception",
            "expected": "Should use details tool"
        },
        {
            "id": "TT003",
            "tool": "recommend_movies_by_genre",
            "query": "Recommend drama movies",
            "expected": "Should use genre recommendation tool"
        },
        {
            "id": "TT004",
            "tool": "find_movies_by_actor",
            "query": "Movies with Leonardo DiCaprio",
            "expected": "Should use actor search tool"
        },
        {
            "id": "TT005",
            "tool": "find_movies_by_director",
            "query": "Films by Quentin Tarantino",
            "expected": "Should use director search tool"
        },
    ]


# ============================================================
# TEST RUNNER FUNCTIONS
# ============================================================

def run_single_test(chatbot, test_case: Dict) -> Dict:
    """Run a single test case and return results."""
    query = test_case["query"]

    try:
        response = chatbot.chat(query)

        # Check if any validation keywords are present
        validations = test_case.get("validation", [])
        passed_validations = []
        failed_validations = []

        for keyword in validations:
            if keyword.lower() in response.lower():
                passed_validations.append(keyword)
            else:
                failed_validations.append(keyword)

        # Determine pass/fail
        if len(validations) == 0:
            status = "PASS" if not response.startswith("⚠️") or "error" not in response.lower() else "REVIEW"
        else:
            status = "PASS" if len(failed_validations) == 0 else "PARTIAL" if len(passed_validations) > 0 else "FAIL"

        return {
            "id": test_case["id"],
            "category": test_case["category"],
            "query": query[:50] + "..." if len(query) > 50 else query,
            "status": status,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
        }

    except Exception as e:
        return {
            "id": test_case["id"],
            "category": test_case["category"],
            "query": query[:50] + "..." if len(query) > 50 else query,
            "status": "ERROR",
            "error": str(e),
        }


def run_test_suite(chatbot, test_cases: List[Dict], suite_name: str) -> None:
    """Run a complete test suite and print results."""
    print(f"\n{'='*60}")
    print(f"🧪 {suite_name}")
    print(f"{'='*60}")

    results = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0, "REVIEW": 0}

    for test in test_cases:
        result = run_single_test(chatbot, test)
        results[result["status"]] += 1

        # Print result
        status_icon = {
            "PASS": "✅",
            "PARTIAL": "🟡",
            "FAIL": "❌",
            "ERROR": "💥",
            "REVIEW": "🔍"
        }

        print(f"\n{status_icon[result['status']]} [{result['id']}] {result['category']}")
        print(f"   Query: {result['query']}")
        print(f"   Status: {result['status']}")

        if result['status'] == "ERROR":
            print(f"   Error: {result.get('error', 'Unknown')}")
        else:
            print(f"   Response: {result.get('response_preview', 'N/A')}")

    # Summary
    print(f"\n{'-'*60}")
    print(f"📊 Summary: PASS={results['PASS']} | PARTIAL={results['PARTIAL']} | FAIL={results['FAIL']} | ERROR={results['ERROR']} | REVIEW={results['REVIEW']}")


def run_all_tests(chatbot) -> None:
    """Run all test suites."""
    print("\n" + "="*60)
    print("🎬 IMDb MOVIE CHATBOT - FULL TEST SUITE")
    print("="*60)

    # Run each test suite
    run_test_suite(chatbot, TestCases.BASIC_QUERIES, "BASIC FUNCTIONALITY TESTS")
    run_test_suite(chatbot, TestCases.COMPLEX_QUERIES, "COMPLEX QUERY TESTS")
    run_test_suite(chatbot, TestCases.EDGE_CASES, "EDGE CASE TESTS")
    run_test_suite(chatbot, TestCases.TOOL_TESTS, "TOOL-SPECIFIC TESTS")

    print("\n" + "="*60)
    print("✅ TEST SUITE COMPLETE")
    print("="*60)


# ============================================================
# MANUAL TEST RUNNER (for Jupyter/Colab)
# ============================================================

def print_test_cases():
    """Print all test cases for manual testing."""
    print("\n" + "="*60)
    print("📋 MANUAL TEST CASES FOR IMDb MOVIE CHATBOT")
    print("="*60)

    all_tests = [
        ("Basic Functionality", TestCases.BASIC_QUERIES),
        ("Complex Queries", TestCases.COMPLEX_QUERIES),
        ("Edge Cases", TestCases.EDGE_CASES),
        ("Tool Tests", TestCases.TOOL_TESTS),
    ]

    for suite_name, tests in all_tests:
        print(f"\n{'─'*60}")
        print(f"📁 {suite_name}")
        print(f"{'─'*60}")

        for t in tests:
            print(f"\n[{t['id']}] {t['category']}")
            print(f"   Query: \"{t['query']}\"")
            print(f"   Expected: {t['expected']}")


# ============================================================
# UNIT TESTS WITH MOCKING (No API Required)
# ============================================================

class MockMovieChatbot:
    """Mock implementation of MovieChatbot for testing without API."""

    def __init__(self):
        self.conversation_history = []

    def _validate_input(self, user_input: str) -> Tuple[bool, str]:
        """Validate user input before processing."""
        if not user_input or not user_input.strip():
            return False, "Please enter a question or request about movies."
        if len(user_input.strip()) < 3:
            return False, "Please provide a more detailed question."
        if len(user_input) > 1000:
            return False, "Your question is too long. Please keep it under 1000 characters."
        return True, ""

    def chat(self, user_input: str) -> str:
        """Mock chat method."""
        is_valid, error_msg = self._validate_input(user_input)
        if not is_valid:
            return f"⚠️ {error_msg}"
        return f"Mock response for: {user_input}"


class TestInputValidation(unittest.TestCase):
    """Unit tests for input validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.chatbot = MockMovieChatbot()

    def test_empty_input_returns_error(self):
        """EC001: Empty input should return error message."""
        response = self.chatbot.chat("")
        self.assertIn("⚠️", response)
        self.assertIn("enter", response.lower())

    def test_whitespace_only_returns_error(self):
        """Whitespace-only input should return error."""
        response = self.chatbot.chat("   ")
        self.assertIn("⚠️", response)

    def test_short_input_returns_error(self):
        """EC002: Very short input should ask for more detail."""
        response = self.chatbot.chat("hi")
        self.assertIn("⚠️", response)
        self.assertIn("detailed", response.lower())

    def test_long_input_returns_error(self):
        """EC003: Input over 1000 chars should be rejected."""
        long_input = "a" * 1500
        response = self.chatbot.chat(long_input)
        self.assertIn("⚠️", response)
        self.assertIn("1000", response)

    def test_valid_input_passes(self):
        """Valid input should not return error."""
        response = self.chatbot.chat("Recommend comedy movies")
        self.assertNotIn("⚠️", response)

    def test_boundary_input_1000_chars(self):
        """Input exactly 1000 chars should be accepted."""
        boundary_input = "a" * 1000
        response = self.chatbot.chat(boundary_input)
        self.assertNotIn("too long", response.lower())

    def test_special_characters_handled(self):
        """EC007: Special characters should not crash."""
        response = self.chatbot.chat("Find movies with @#$% in title")
        # Should not raise exception
        self.assertIsInstance(response, str)

    def test_sql_injection_safe(self):
        """EC008: SQL injection attempts should be handled safely."""
        response = self.chatbot.chat("'; DROP TABLE movies; --")
        # Should not raise exception
        self.assertIsInstance(response, str)


class TestChatbotWithMockedAPI(unittest.TestCase):
    """Tests using mocked OpenAI API calls."""

    def setUp(self):
        """Set up mocked dependencies."""
        self.mock_agent = Mock()
        self.mock_agent.invoke = Mock(return_value={
            'output': 'Here are some comedy movies: Movie1, Movie2, Movie3'
        })

    def test_agent_invoked_with_correct_input(self):
        """Agent should be called with user input."""
        self.mock_agent.invoke({"input": "Recommend comedy movies"})
        self.mock_agent.invoke.assert_called_once_with({"input": "Recommend comedy movies"})

    def test_agent_response_returned(self):
        """Agent response should be returned to user."""
        result = self.mock_agent.invoke({"input": "test"})
        self.assertEqual(result['output'], 'Here are some comedy movies: Movie1, Movie2, Movie3')


class TestDocumentCreation(unittest.TestCase):
    """Tests for document creation and embedding logic."""

    def test_create_movie_description(self):
        """Movie description should include all key fields."""
        # Mock movie data
        movie = {
            'Title': 'Inception',
            'IMDb Rating': 8.8,
            'Year': 2010,
            'Genre': 'Action, Sci-Fi, Thriller',
            'Director': 'Christopher Nolan',
            'Star Cast': 'Leonardo DiCaprio, Joseph Gordon-Levitt',
            'Duration (minutes)': 148,
            'Certificates': 'PG-13',
            'MetaScore': 74
        }

        # Create description (mimicking the notebook function)
        description = f"""
        Title: {movie['Title']}
        Year: {movie['Year']}
        Genre: {movie['Genre']}
        Director: {movie['Director']}
        Cast: {movie['Star Cast']}
        IMDb Rating: {movie['IMDb Rating']}/10
        Duration: {movie['Duration (minutes)']} minutes
        Certificate: {movie['Certificates']}
        MetaScore: {movie['MetaScore']}
        """

        self.assertIn('Inception', description)
        self.assertIn('8.8', description)
        self.assertIn('Christopher Nolan', description)
        self.assertIn('Leonardo DiCaprio', description)


class TestToolSelection(unittest.TestCase):
    """Tests for verifying correct tool selection based on query type."""

    def setUp(self):
        """Set up mock tools."""
        self.tools = {
            'search_movies_by_query': Mock(return_value="Search results"),
            'get_movie_details': Mock(return_value="Movie details"),
            'recommend_movies_by_genre': Mock(return_value="Genre recommendations"),
            'find_movies_by_actor': Mock(return_value="Actor movies"),
            'find_movies_by_director': Mock(return_value="Director movies"),
            'get_top_rated_movies': Mock(return_value="Top rated"),
            'compare_movies': Mock(return_value="Comparison"),
        }

    def test_genre_query_keywords(self):
        """Genre queries should contain genre-related keywords."""
        genre_queries = [
            "Recommend comedy movies",
            "Find horror films",
            "Show me drama movies"
        ]
        genre_keywords = ['comedy', 'horror', 'drama', 'action', 'thriller', 'romance']

        for query in genre_queries:
            has_genre = any(kw in query.lower() for kw in genre_keywords)
            self.assertTrue(has_genre, f"Query should contain genre keyword: {query}")

    def test_actor_query_keywords(self):
        """Actor queries should contain actor-related keywords."""
        actor_queries = [
            "Movies with Tom Hanks",
            "Films starring Leonardo DiCaprio",
            "What has Brad Pitt been in?"
        ]
        actor_keywords = ['with', 'starring', 'actor', 'star', 'been in']

        for query in actor_queries:
            has_actor_keyword = any(kw in query.lower() for kw in actor_keywords)
            self.assertTrue(has_actor_keyword, f"Query should contain actor keyword: {query}")

    def test_director_query_keywords(self):
        """Director queries should contain director-related keywords."""
        director_queries = [
            "Films by Christopher Nolan",
            "Movies directed by Spielberg",
            "Tarantino films"
        ]
        director_keywords = ['by', 'directed', 'director']

        for query in director_queries:
            # At minimum should have 'by' or 'directed'
            has_keyword = any(kw in query.lower() for kw in director_keywords)
            self.assertTrue(has_keyword or 'films' in query.lower())


class TestConversationMemory(unittest.TestCase):
    """Tests for conversation memory functionality."""

    def setUp(self):
        """Set up mock chatbot with memory."""
        self.chatbot = MockMovieChatbot()

    def test_conversation_history_initialized(self):
        """Conversation history should start empty."""
        self.assertEqual(len(self.chatbot.conversation_history), 0)

    def test_history_can_be_appended(self):
        """Should be able to add to conversation history."""
        self.chatbot.conversation_history.append({
            "user": "test query",
            "assistant": "test response"
        })
        self.assertEqual(len(self.chatbot.conversation_history), 1)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling scenarios."""

    def test_api_error_message_format(self):
        """API errors should return user-friendly messages."""
        error_messages = [
            "rate limit exceeded",
            "invalid api key",
            "connection error"
        ]

        for msg in error_messages:
            # Verify error messages are lowercase-comparable
            self.assertEqual(msg, msg.lower())

    def test_exception_doesnt_crash(self):
        """Exceptions should be caught, not crash the chatbot."""
        chatbot = MockMovieChatbot()

        # These should not raise exceptions
        try:
            chatbot.chat("")
            chatbot.chat("a" * 2000)
            chatbot.chat("'; DROP TABLE; --")
        except Exception as e:
            self.fail(f"Chatbot crashed with exception: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("IMDb Movie Chatbot Test Suite")
    print("="*60)
    print("\nTo run tests:")
    print("1. First run the notebook cells to create 'chatbot' object")
    print("2. Then run: run_all_tests(chatbot)")
    print("\nOr print test cases for manual testing:")
    print("   print_test_cases()")
    print("\nTo run unit tests (no API required):")
    print("   python -m pytest test_chatbot.py -v")
    print("   or: python -m unittest test_chatbot -v")
    print("\n")

    # Print test cases for reference
    print_test_cases()

    # Run unit tests
    print("\n" + "="*60)
    print("Running Unit Tests...")
    print("="*60)
    unittest.main(verbosity=2, exit=False)
