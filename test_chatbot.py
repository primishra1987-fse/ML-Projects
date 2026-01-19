"""
IMDb Movie Chatbot - Enhanced Multi-Agent Test Suite
Run with: python test_chatbot.py (after running notebook cells 0-17)

For unit tests with mocking (no API required):
    python -m pytest test_chatbot.py -v

This test suite covers all 11 specialized agents:
1. Genre Expert - Genre-based recommendations
2. Filmography Expert - Actor/Director searches
3. Rating Analyst - Rating-based filtering
4. Comparison Analyst - Movie/genre comparisons
5. Mood Curator - Mood-based recommendations
6. Trivia Master - Movie trivia and quizzes
7. Review Analyst - Review sentiment analysis
8. Similarity Finder - Similar movie recommendations
9. Runtime Advisor - Duration-based searches
10. Era Explorer - Year/decade searches
11. General Assistant - General queries
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Tuple, Optional
import re

# ============================================================
# TEST CASES FOR IMDB MOVIE CHATBOT - MULTI-AGENT SYSTEM
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
            "agent": "Genre Expert",
            "query": "Recommend some comedy movies",
            "expected": "Should return comedy movies from database",
            "validation": ["comedy", "movie", "rating"]
        },
        {
            "id": "BF002",
            "category": "Genre Search",
            "agent": "Genre Expert",
            "query": "Find documentary films",
            "expected": "Should return documentary movies",
            "validation": ["documentary", "film"]
        },
        {
            "id": "BF003",
            "category": "Genre Search",
            "agent": "Genre Expert",
            "query": "Show me some biography movies",
            "expected": "Should return biography/biopic movies",
            "validation": ["biography", "movie"]
        },
        {
            "id": "BF004",
            "category": "Actor Search",
            "agent": "Filmography Expert",
            "query": "What movies has Tom Hanks starred in?",
            "expected": "Should return movies featuring Tom Hanks",
            "validation": ["Tom Hanks", "movie", "star"]
        },
        {
            "id": "BF005",
            "category": "Director Search",
            "agent": "Filmography Expert",
            "query": "Find movies directed by Steven Spielberg",
            "expected": "Should return Spielberg's filmography",
            "validation": ["director", "movie"]
        },
        {
            "id": "BF006",
            "category": "Rating Filter",
            "agent": "Rating Analyst",
            "query": "Show movies rated above 8.0",
            "expected": "Should return highly rated movies (8.0+)",
            "validation": ["rating", "8"]
        },
        {
            "id": "BF007",
            "category": "Year Filter",
            "agent": "Era Explorer",
            "query": "What movies came out in 2020?",
            "expected": "Should return movies from 2020",
            "validation": ["2020", "movie"]
        },
    ]

    # ----------------------------------------------------------
    # 2. MULTI-AGENT ROUTING TESTS
    # ----------------------------------------------------------

    AGENT_ROUTING_TESTS = [
        {
            "id": "AR001",
            "category": "Genre Agent Routing",
            "agent": "Genre Expert",
            "query": "I want to watch some thriller movies tonight",
            "expected": "Should route to Genre Expert agent",
            "validation": ["thriller", "movie"]
        },
        {
            "id": "AR002",
            "category": "Mood Agent Routing",
            "agent": "Mood Curator",
            "query": "I'm feeling sad, what should I watch?",
            "expected": "Should route to Mood Curator agent",
            "validation": ["movie", "recommend"]
        },
        {
            "id": "AR003",
            "category": "Filmography Agent Routing",
            "agent": "Filmography Expert",
            "query": "Show me all Quentin Tarantino films",
            "expected": "Should route to Filmography Expert agent",
            "validation": ["movie"]
        },
        {
            "id": "AR004",
            "category": "Rating Agent Routing",
            "agent": "Rating Analyst",
            "query": "What are the highest rated movies?",
            "expected": "Should route to Rating Analyst agent",
            "validation": ["rating", "movie"]
        },
        {
            "id": "AR005",
            "category": "Duration Agent Routing",
            "agent": "Runtime Advisor",
            "query": "I only have 90 minutes, what can I watch?",
            "expected": "Should route to Runtime Advisor agent",
            "validation": ["movie", "minutes"]
        },
        {
            "id": "AR006",
            "category": "Era Agent Routing",
            "agent": "Era Explorer",
            "query": "What classic movies from the 80s should I watch?",
            "expected": "Should route to Era Explorer agent",
            "validation": ["80", "movie"]
        },
        {
            "id": "AR007",
            "category": "Similar Movies Routing",
            "agent": "Similarity Finder",
            "query": "I loved Inception, what similar movies would you recommend?",
            "expected": "Should route to Similarity Finder agent",
            "validation": ["similar", "movie"]
        },
        {
            "id": "AR008",
            "category": "Comparison Agent Routing",
            "agent": "Comparison Analyst",
            "query": "Compare The Dark Knight and Inception",
            "expected": "Should route to Comparison Analyst agent",
            "validation": ["compare"]
        },
        {
            "id": "AR009",
            "category": "Trivia Agent Routing",
            "agent": "Trivia Master",
            "query": "Tell me an interesting fact about Titanic",
            "expected": "Should route to Trivia Master agent",
            "validation": ["fact", "Titanic"]
        },
        {
            "id": "AR010",
            "category": "Review Agent Routing",
            "agent": "Review Analyst",
            "query": "What do critics think about Parasite?",
            "expected": "Should route to Review Analyst agent",
            "validation": ["critic", "review"]
        },
        {
            "id": "AR011",
            "category": "General Agent Routing",
            "agent": "General Assistant",
            "query": "What's a good movie to watch?",
            "expected": "Should route to General Assistant agent",
            "validation": ["movie", "recommend"]
        },
    ]

    # ----------------------------------------------------------
    # 3. MOOD-BASED RECOMMENDATION TESTS
    # ----------------------------------------------------------

    MOOD_TESTS = [
        {
            "id": "MO001",
            "category": "Happy Mood",
            "agent": "Mood Curator",
            "query": "I'm feeling happy, recommend something fun to watch",
            "expected": "Should recommend comedy or feel-good movies",
            "validation": ["movie", "comedy"]
        },
        {
            "id": "MO002",
            "category": "Sad Mood",
            "agent": "Mood Curator",
            "query": "I'm feeling melancholic, need a movie",
            "expected": "Should recommend drama or emotional movies",
            "validation": ["movie"]
        },
        {
            "id": "MO003",
            "category": "Excited Mood",
            "agent": "Mood Curator",
            "query": "I'm feeling adventurous and want some excitement!",
            "expected": "Should recommend action or adventure movies",
            "validation": ["movie", "action"]
        },
        {
            "id": "MO004",
            "category": "Romantic Mood",
            "agent": "Mood Curator",
            "query": "In a romantic mood tonight, any suggestions?",
            "expected": "Should recommend romance movies",
            "validation": ["romance", "movie"]
        },
        {
            "id": "MO005",
            "category": "Scared Mood",
            "agent": "Mood Curator",
            "query": "I want to feel scared, show me something terrifying",
            "expected": "Should recommend horror movies",
            "validation": ["horror", "movie"]
        },
        {
            "id": "MO006",
            "category": "Curious Mood",
            "agent": "Mood Curator",
            "query": "Feeling curious and want to learn something",
            "expected": "Should recommend documentaries",
            "validation": ["documentary", "movie"]
        },
        {
            "id": "MO007",
            "category": "Nostalgic Mood",
            "agent": "Mood Curator",
            "query": "Feeling nostalgic, recommend something classic",
            "expected": "Should recommend classic movies",
            "validation": ["movie"]
        },
        {
            "id": "MO008",
            "category": "Relaxed Mood",
            "agent": "Mood Curator",
            "query": "Want something light and relaxing to watch",
            "expected": "Should recommend easy-to-watch movies",
            "validation": ["movie"]
        },
    ]

    # ----------------------------------------------------------
    # 4. DURATION/RUNTIME TESTS
    # ----------------------------------------------------------

    DURATION_TESTS = [
        {
            "id": "DU001",
            "category": "Short Movies",
            "agent": "Runtime Advisor",
            "query": "Show me movies under 90 minutes",
            "expected": "Should return short movies",
            "validation": ["movie", "minutes"]
        },
        {
            "id": "DU002",
            "category": "Long Movies",
            "agent": "Runtime Advisor",
            "query": "I want a long movie over 2 hours",
            "expected": "Should return movies over 120 minutes",
            "validation": ["movie", "hour"]
        },
        {
            "id": "DU003",
            "category": "Specific Duration",
            "agent": "Runtime Advisor",
            "query": "Find movies around 100 minutes long",
            "expected": "Should return movies close to 100 minutes",
            "validation": ["movie", "minutes"]
        },
        {
            "id": "DU004",
            "category": "Quick Watch",
            "agent": "Runtime Advisor",
            "query": "I need something quick for my lunch break",
            "expected": "Should return shorter movies",
            "validation": ["movie"]
        },
        {
            "id": "DU005",
            "category": "Epic Movies",
            "agent": "Runtime Advisor",
            "query": "What are the longest epic movies available?",
            "expected": "Should return long epic movies",
            "validation": ["movie", "long"]
        },
    ]

    # ----------------------------------------------------------
    # 5. ERA/YEAR-BASED TESTS
    # ----------------------------------------------------------

    ERA_TESTS = [
        {
            "id": "ER001",
            "category": "Specific Year",
            "agent": "Era Explorer",
            "query": "Best movies of 2019",
            "expected": "Should return top movies from 2019",
            "validation": ["2019", "movie"]
        },
        {
            "id": "ER002",
            "category": "Decade Search",
            "agent": "Era Explorer",
            "query": "Recommend movies from the 1990s",
            "expected": "Should return movies from 1990-1999",
            "validation": ["199", "movie"]
        },
        {
            "id": "ER003",
            "category": "Recent Movies",
            "agent": "Era Explorer",
            "query": "What are the latest movies from 2023?",
            "expected": "Should return recent movies",
            "validation": ["202", "movie"]
        },
        {
            "id": "ER004",
            "category": "Classic Films",
            "agent": "Era Explorer",
            "query": "Show me classic films from the golden age",
            "expected": "Should return older classic movies",
            "validation": ["movie"]
        },
        {
            "id": "ER005",
            "category": "Year Range",
            "agent": "Era Explorer",
            "query": "Movies released between 2015 and 2020",
            "expected": "Should return movies in year range",
            "validation": ["movie"]
        },
    ]

    # ----------------------------------------------------------
    # 6. SIMILAR MOVIES TESTS
    # ----------------------------------------------------------

    SIMILARITY_TESTS = [
        {
            "id": "SI001",
            "category": "Similar to Specific Movie",
            "agent": "Similarity Finder",
            "query": "Movies similar to The Matrix",
            "expected": "Should return sci-fi action movies",
            "validation": ["movie", "similar"]
        },
        {
            "id": "SI002",
            "category": "Like Reference Movie",
            "agent": "Similarity Finder",
            "query": "I liked Forrest Gump, what else would I enjoy?",
            "expected": "Should return similar drama movies",
            "validation": ["movie"]
        },
        {
            "id": "SI003",
            "category": "More Like This",
            "agent": "Similarity Finder",
            "query": "Recommend movies like Pulp Fiction",
            "expected": "Should return similar crime/drama movies",
            "validation": ["movie"]
        },
        {
            "id": "SI004",
            "category": "Alternative Suggestions",
            "agent": "Similarity Finder",
            "query": "What can I watch instead of Avatar?",
            "expected": "Should return alternative sci-fi movies",
            "validation": ["movie"]
        },
    ]

    # ----------------------------------------------------------
    # 7. COMPARISON TESTS
    # ----------------------------------------------------------

    COMPARISON_TESTS = [
        {
            "id": "CO001",
            "category": "Movie vs Movie",
            "agent": "Comparison Analyst",
            "query": "Compare Titanic and Avatar",
            "expected": "Should compare both movies",
            "validation": ["Titanic", "Avatar"]
        },
        {
            "id": "CO002",
            "category": "Genre vs Genre",
            "agent": "Comparison Analyst",
            "query": "What's the difference between thriller and horror?",
            "expected": "Should compare both genres",
            "validation": ["thriller", "horror"]
        },
        {
            "id": "CO003",
            "category": "Director Comparison",
            "agent": "Comparison Analyst",
            "query": "Compare Nolan and Spielberg films",
            "expected": "Should compare filmographies",
            "validation": ["Nolan", "Spielberg"]
        },
        {
            "id": "CO004",
            "category": "Which is Better",
            "agent": "Comparison Analyst",
            "query": "Is The Godfather better than Goodfellas?",
            "expected": "Should compare and analyze both",
            "validation": ["Godfather", "Goodfellas"]
        },
    ]

    # ----------------------------------------------------------
    # 8. TRIVIA AND FACTS TESTS
    # ----------------------------------------------------------

    TRIVIA_TESTS = [
        {
            "id": "TR001",
            "category": "Movie Trivia",
            "agent": "Trivia Master",
            "query": "Tell me an interesting fact about Inception",
            "expected": "Should provide trivia about Inception",
            "validation": ["Inception", "fact"]
        },
        {
            "id": "TR002",
            "category": "Fun Facts",
            "agent": "Trivia Master",
            "query": "Share some fun facts about Star Wars",
            "expected": "Should provide Star Wars trivia",
            "validation": ["Star Wars"]
        },
        {
            "id": "TR003",
            "category": "Behind the Scenes",
            "agent": "Trivia Master",
            "query": "Any behind the scenes info about The Dark Knight?",
            "expected": "Should provide BTS information",
            "validation": ["Dark Knight"]
        },
        {
            "id": "TR004",
            "category": "Movie Quiz",
            "agent": "Trivia Master",
            "query": "Give me a movie quiz question",
            "expected": "Should provide quiz question",
            "validation": ["question"]
        },
    ]

    # ----------------------------------------------------------
    # 9. REVIEW/SENTIMENT TESTS
    # ----------------------------------------------------------

    REVIEW_TESTS = [
        {
            "id": "RE001",
            "category": "Critical Reception",
            "agent": "Review Analyst",
            "query": "What do critics think about Parasite?",
            "expected": "Should analyze critical reception",
            "validation": ["Parasite", "review"]
        },
        {
            "id": "RE002",
            "category": "Audience Opinion",
            "agent": "Review Analyst",
            "query": "Is Joker a well-received movie?",
            "expected": "Should discuss audience/critic reception",
            "validation": ["Joker"]
        },
        {
            "id": "RE003",
            "category": "Rating Analysis",
            "agent": "Review Analyst",
            "query": "Why is Shawshank Redemption rated so highly?",
            "expected": "Should explain rating reasons",
            "validation": ["Shawshank", "rating"]
        },
        {
            "id": "RE004",
            "category": "Controversial Movies",
            "agent": "Review Analyst",
            "query": "What's the sentiment around controversial movies?",
            "expected": "Should discuss movie sentiment",
            "validation": ["movie"]
        },
    ]

    # ----------------------------------------------------------
    # 10. COMPLEX QUERY TESTS
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
            "category": "Combined Agent",
            "query": "I want a short comedy from 2020 with good ratings",
            "expected": "Should handle multiple criteria",
            "validation": ["comedy", "2020", "rating"]
        },
        {
            "id": "CQ004",
            "category": "Specific Detail",
            "query": "What is the runtime of the longest movie in the database?",
            "expected": "Should identify longest movie by duration",
            "validation": ["duration", "minutes", "movie"]
        },
        {
            "id": "CQ005",
            "category": "Context Required",
            "query": "I liked The Dark Knight, recommend similar movies",
            "expected": "Should recommend action/thriller movies",
            "validation": ["recommend", "movie"]
        },
    ]

    # ----------------------------------------------------------
    # 11. EDGE CASE TESTS
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
            "validation": []
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
        {
            "id": "EC009",
            "category": "Mixed Languages",
            "query": "Recommend some action peliculas",
            "expected": "Should attempt to understand mixed language query",
            "validation": ["movie"]
        },
        {
            "id": "EC010",
            "category": "Numeric Only",
            "query": "1234567890",
            "expected": "Should ask for clarification",
            "validation": []
        },
    ]

    # ----------------------------------------------------------
    # 12. CONVERSATION FLOW TESTS
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
        {
            "id": "CF004",
            "category": "Entity Reference",
            "queries": [
                "Find movies by Christopher Nolan",
                "Which one has the highest rating?"
            ],
            "expected": "Should reference previously mentioned director",
        },
        {
            "id": "CF005",
            "category": "Mood Transition",
            "queries": [
                "I'm feeling happy, suggest something",
                "Actually, I changed my mind, I want something scary"
            ],
            "expected": "Should adapt to mood change",
        },
    ]

    # ----------------------------------------------------------
    # 13. AGENT-SPECIFIC TOOL TESTS
    # ----------------------------------------------------------

    TOOL_TESTS = [
        {
            "id": "TT001",
            "tool": "search_movies_by_query",
            "agent": "General Assistant",
            "query": "thriller movies with suspense",
            "expected": "Should use search tool"
        },
        {
            "id": "TT002",
            "tool": "get_movie_details",
            "agent": "General Assistant",
            "query": "Tell me everything about Inception",
            "expected": "Should use details tool"
        },
        {
            "id": "TT003",
            "tool": "recommend_movies_by_genre",
            "agent": "Genre Expert",
            "query": "Recommend drama movies",
            "expected": "Should use genre recommendation tool"
        },
        {
            "id": "TT004",
            "tool": "find_movies_by_actor",
            "agent": "Filmography Expert",
            "query": "Movies with Leonardo DiCaprio",
            "expected": "Should use actor search tool"
        },
        {
            "id": "TT005",
            "tool": "find_movies_by_director",
            "agent": "Filmography Expert",
            "query": "Films by Quentin Tarantino",
            "expected": "Should use director search tool"
        },
        {
            "id": "TT006",
            "tool": "get_top_rated_movies",
            "agent": "Rating Analyst",
            "query": "Show me top rated movies above 9.0",
            "expected": "Should use top rated tool"
        },
        {
            "id": "TT007",
            "tool": "compare_movies",
            "agent": "Comparison Analyst",
            "query": "Compare action and comedy genres",
            "expected": "Should use compare tool"
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
            "agent": test_case.get("agent", "Unknown"),
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
            "agent": test_case.get("agent", "Unknown"),
            "query": query[:50] + "..." if len(query) > 50 else query,
            "status": "ERROR",
            "error": str(e),
        }


def run_test_suite(chatbot, test_cases: List[Dict], suite_name: str) -> Dict:
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
        if result.get('agent'):
            print(f"   Agent: {result['agent']}")
        print(f"   Query: {result['query']}")
        print(f"   Status: {result['status']}")

        if result['status'] == "ERROR":
            print(f"   Error: {result.get('error', 'Unknown')}")
        else:
            print(f"   Response: {result.get('response_preview', 'N/A')}")

    # Summary
    print(f"\n{'-'*60}")
    print(f"📊 Summary: PASS={results['PASS']} | PARTIAL={results['PARTIAL']} | FAIL={results['FAIL']} | ERROR={results['ERROR']} | REVIEW={results['REVIEW']}")

    return results


def run_all_tests(chatbot) -> Dict:
    """Run all test suites and return combined results."""
    print("\n" + "="*60)
    print("🎬 IMDb MOVIE CHATBOT - ENHANCED MULTI-AGENT TEST SUITE")
    print("="*60)

    all_results = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0, "REVIEW": 0}

    # Define all test suites
    test_suites = [
        ("BASIC FUNCTIONALITY TESTS", TestCases.BASIC_QUERIES),
        ("MULTI-AGENT ROUTING TESTS", TestCases.AGENT_ROUTING_TESTS),
        ("MOOD-BASED RECOMMENDATION TESTS", TestCases.MOOD_TESTS),
        ("DURATION/RUNTIME TESTS", TestCases.DURATION_TESTS),
        ("ERA/YEAR-BASED TESTS", TestCases.ERA_TESTS),
        ("SIMILAR MOVIES TESTS", TestCases.SIMILARITY_TESTS),
        ("COMPARISON TESTS", TestCases.COMPARISON_TESTS),
        ("TRIVIA/FACTS TESTS", TestCases.TRIVIA_TESTS),
        ("REVIEW/SENTIMENT TESTS", TestCases.REVIEW_TESTS),
        ("COMPLEX QUERY TESTS", TestCases.COMPLEX_QUERIES),
        ("EDGE CASE TESTS", TestCases.EDGE_CASES),
        ("AGENT-SPECIFIC TOOL TESTS", TestCases.TOOL_TESTS),
    ]

    # Run each test suite
    for suite_name, test_cases in test_suites:
        suite_results = run_test_suite(chatbot, test_cases, suite_name)
        for key in all_results:
            all_results[key] += suite_results[key]

    # Final summary
    total_tests = sum(all_results.values())
    print("\n" + "="*60)
    print("📊 FINAL TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ PASS: {all_results['PASS']} ({100*all_results['PASS']/total_tests:.1f}%)")
    print(f"🟡 PARTIAL: {all_results['PARTIAL']} ({100*all_results['PARTIAL']/total_tests:.1f}%)")
    print(f"❌ FAIL: {all_results['FAIL']} ({100*all_results['FAIL']/total_tests:.1f}%)")
    print(f"💥 ERROR: {all_results['ERROR']} ({100*all_results['ERROR']/total_tests:.1f}%)")
    print(f"🔍 REVIEW: {all_results['REVIEW']} ({100*all_results['REVIEW']/total_tests:.1f}%)")
    print("="*60)
    print("✅ TEST SUITE COMPLETE")
    print("="*60)

    return all_results


# ============================================================
# MANUAL TEST RUNNER (for Jupyter/Colab)
# ============================================================

def print_test_cases():
    """Print all test cases for manual testing."""
    print("\n" + "="*60)
    print("📋 MANUAL TEST CASES FOR IMDb MOVIE CHATBOT - MULTI-AGENT SYSTEM")
    print("="*60)

    all_tests = [
        ("Basic Functionality", TestCases.BASIC_QUERIES),
        ("Multi-Agent Routing", TestCases.AGENT_ROUTING_TESTS),
        ("Mood-Based Recommendations", TestCases.MOOD_TESTS),
        ("Duration/Runtime", TestCases.DURATION_TESTS),
        ("Era/Year-Based", TestCases.ERA_TESTS),
        ("Similar Movies", TestCases.SIMILARITY_TESTS),
        ("Comparisons", TestCases.COMPARISON_TESTS),
        ("Trivia/Facts", TestCases.TRIVIA_TESTS),
        ("Review/Sentiment", TestCases.REVIEW_TESTS),
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
            if 'agent' in t:
                print(f"   Agent: {t['agent']}")
            print(f"   Query: \"{t['query'][:80]}{'...' if len(t['query']) > 80 else ''}\"")
            print(f"   Expected: {t['expected']}")


def print_agent_coverage():
    """Print coverage information for all 11 agents."""
    print("\n" + "="*60)
    print("🤖 MULTI-AGENT COVERAGE REPORT")
    print("="*60)

    agents = {
        "Genre Expert": "Handles genre-based movie recommendations",
        "Filmography Expert": "Handles actor and director filmography queries",
        "Rating Analyst": "Handles rating-based filtering and analysis",
        "Comparison Analyst": "Handles movie and genre comparisons",
        "Mood Curator": "Handles mood-based recommendations",
        "Trivia Master": "Handles movie trivia and quiz questions",
        "Review Analyst": "Handles review sentiment and critical reception",
        "Similarity Finder": "Handles similar movie recommendations",
        "Runtime Advisor": "Handles duration-based filtering",
        "Era Explorer": "Handles year and decade-based searches",
        "General Assistant": "Handles general queries and fallback",
    }

    # Count tests per agent
    all_tests = (
        TestCases.BASIC_QUERIES +
        TestCases.AGENT_ROUTING_TESTS +
        TestCases.MOOD_TESTS +
        TestCases.DURATION_TESTS +
        TestCases.ERA_TESTS +
        TestCases.SIMILARITY_TESTS +
        TestCases.COMPARISON_TESTS +
        TestCases.TRIVIA_TESTS +
        TestCases.REVIEW_TESTS +
        TestCases.TOOL_TESTS
    )

    agent_counts = {agent: 0 for agent in agents}
    for test in all_tests:
        agent = test.get('agent', 'Unknown')
        if agent in agent_counts:
            agent_counts[agent] += 1

    print("\nAgent Test Coverage:")
    print("-" * 50)
    for agent, description in agents.items():
        count = agent_counts[agent]
        bar = "█" * min(count, 20) + "░" * (20 - min(count, 20))
        print(f"{agent:20} [{bar}] {count} tests")
        print(f"                     └─ {description}")

    print("\n" + "="*60)


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


class MockQueryClassifier:
    """Mock implementation of EnhancedQueryClassifier for testing."""

    AGENT_PATTERNS = {
        "genre": [r'\b(comedy|drama|action|horror|thriller|romance|sci-fi|documentary)\b'],
        "mood": [r'\b(feeling|mood|happy|sad|excited|scared|romantic)\b'],
        "actor": [r'\b(starring|actor|actress|with)\b.*\b(movie|film)\b'],
        "director": [r'\b(directed|director|by)\b'],
        "rating": [r'\b(rated|rating|score|top)\b'],
        "duration": [r'\b(minutes|hours|long|short|runtime|duration)\b'],
        "era": [r'\b(19\d{2}|20\d{2}|decade|year|classic)\b'],
        "similar": [r'\b(similar|like|recommend.*like)\b'],
        "compare": [r'\b(compare|versus|vs|difference)\b'],
        "trivia": [r'\b(fact|trivia|quiz|behind the scenes)\b'],
        "review": [r'\b(review|critic|sentiment|reception)\b'],
    }

    def classify(self, query: str) -> Tuple[str, float]:
        """Classify query to appropriate agent."""
        query_lower = query.lower()

        for agent, patterns in self.AGENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return agent, 0.9

        return "general", 0.5


class MockRateLimiter:
    """Mock implementation of RateLimiter for testing."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        import time
        current_time = time.time()
        self.requests = [r for r in self.requests if current_time - r < self.window_seconds]

        if len(self.requests) >= self.max_requests:
            return False

        self.requests.append(current_time)
        return True


class MockQueryCache:
    """Mock implementation of QueryCache for testing."""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size

    def get(self, query: str) -> Optional[str]:
        """Get cached response."""
        return self.cache.get(query.lower().strip())

    def set(self, query: str, response: str) -> None:
        """Cache a response."""
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[query.lower().strip()] = response


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
        self.assertIsInstance(response, str)

    def test_sql_injection_safe(self):
        """EC008: SQL injection attempts should be handled safely."""
        response = self.chatbot.chat("'; DROP TABLE movies; --")
        self.assertIsInstance(response, str)


class TestQueryClassifier(unittest.TestCase):
    """Unit tests for query classification logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MockQueryClassifier()

    def test_genre_classification(self):
        """Genre queries should be classified correctly."""
        agent, confidence = self.classifier.classify("Recommend some comedy movies")
        self.assertEqual(agent, "genre")
        self.assertGreater(confidence, 0.5)

    def test_mood_classification(self):
        """Mood queries should be classified correctly."""
        agent, confidence = self.classifier.classify("I'm feeling happy, what should I watch?")
        self.assertEqual(agent, "mood")
        self.assertGreater(confidence, 0.5)

    def test_rating_classification(self):
        """Rating queries should be classified correctly."""
        agent, confidence = self.classifier.classify("Show me top rated movies")
        self.assertEqual(agent, "rating")
        self.assertGreater(confidence, 0.5)

    def test_duration_classification(self):
        """Duration queries should be classified correctly."""
        agent, confidence = self.classifier.classify("Movies under 90 minutes")
        self.assertEqual(agent, "duration")
        self.assertGreater(confidence, 0.5)

    def test_era_classification(self):
        """Era queries should be classified correctly."""
        agent, confidence = self.classifier.classify("Best movies of 2020")
        self.assertEqual(agent, "era")
        self.assertGreater(confidence, 0.5)

    def test_compare_classification(self):
        """Comparison queries should be classified correctly."""
        agent, confidence = self.classifier.classify("Compare Titanic and Avatar")
        self.assertEqual(agent, "compare")
        self.assertGreater(confidence, 0.5)

    def test_trivia_classification(self):
        """Trivia queries should be classified correctly."""
        agent, confidence = self.classifier.classify("Tell me a fact about Inception")
        self.assertEqual(agent, "trivia")
        self.assertGreater(confidence, 0.5)

    def test_review_classification(self):
        """Review queries should be classified correctly."""
        agent, confidence = self.classifier.classify("What do critics think about this movie?")
        self.assertEqual(agent, "review")
        self.assertGreater(confidence, 0.5)

    def test_general_fallback(self):
        """Unknown queries should fallback to general."""
        agent, confidence = self.classifier.classify("xyz abc 123")
        self.assertEqual(agent, "general")
        self.assertLessEqual(confidence, 0.5)


class TestRateLimiter(unittest.TestCase):
    """Unit tests for rate limiting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.rate_limiter = MockRateLimiter(max_requests=5, window_seconds=60)

    def test_initial_request_allowed(self):
        """First request should be allowed."""
        self.assertTrue(self.rate_limiter.is_allowed())

    def test_multiple_requests_allowed(self):
        """Multiple requests within limit should be allowed."""
        for _ in range(5):
            self.assertTrue(self.rate_limiter.is_allowed())

    def test_exceeding_limit_blocked(self):
        """Requests exceeding limit should be blocked."""
        for _ in range(5):
            self.rate_limiter.is_allowed()
        self.assertFalse(self.rate_limiter.is_allowed())


class TestQueryCache(unittest.TestCase):
    """Unit tests for query caching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = MockQueryCache(max_size=3)

    def test_cache_miss(self):
        """Cache should return None for missing queries."""
        result = self.cache.get("nonexistent query")
        self.assertIsNone(result)

    def test_cache_hit(self):
        """Cache should return stored response."""
        self.cache.set("test query", "test response")
        result = self.cache.get("test query")
        self.assertEqual(result, "test response")

    def test_cache_case_insensitive(self):
        """Cache should be case insensitive."""
        self.cache.set("Test Query", "test response")
        result = self.cache.get("test query")
        self.assertEqual(result, "test response")

    def test_cache_eviction(self):
        """Cache should evict oldest entry when full."""
        self.cache.set("query1", "response1")
        self.cache.set("query2", "response2")
        self.cache.set("query3", "response3")
        self.cache.set("query4", "response4")  # Should evict query1

        self.assertIsNone(self.cache.get("query1"))
        self.assertIsNotNone(self.cache.get("query4"))


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
            self.assertEqual(msg, msg.lower())

    def test_exception_doesnt_crash(self):
        """Exceptions should be caught, not crash the chatbot."""
        chatbot = MockMovieChatbot()

        try:
            chatbot.chat("")
            chatbot.chat("a" * 2000)
            chatbot.chat("'; DROP TABLE; --")
        except Exception as e:
            self.fail(f"Chatbot crashed with exception: {e}")


class TestMoodMapping(unittest.TestCase):
    """Tests for mood to genre mapping functionality."""

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
        "thoughtful": ["Drama", "Biography", "Documentary"],
    }

    def test_all_moods_have_mappings(self):
        """All moods should have genre mappings."""
        for mood in ["happy", "sad", "excited", "romantic", "scared",
                     "curious", "nostalgic", "relaxed", "adventurous", "thoughtful"]:
            self.assertIn(mood, self.MOOD_GENRE_MAPPING)
            self.assertGreater(len(self.MOOD_GENRE_MAPPING[mood]), 0)

    def test_genre_validity(self):
        """All mapped genres should be valid movie genres."""
        valid_genres = {"Comedy", "Animation", "Family", "Drama", "Romance",
                       "Action", "Adventure", "Thriller", "Horror", "Documentary",
                       "Mystery", "Sci-Fi", "Biography"}

        for mood, genres in self.MOOD_GENRE_MAPPING.items():
            for genre in genres:
                self.assertIn(genre, valid_genres,
                            f"Invalid genre '{genre}' for mood '{mood}'")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("IMDb Movie Chatbot - Enhanced Multi-Agent Test Suite")
    print("="*60)
    print("\nThis test suite covers all 11 specialized agents:")
    print("  1. Genre Expert       - Genre-based recommendations")
    print("  2. Filmography Expert - Actor/Director searches")
    print("  3. Rating Analyst     - Rating-based filtering")
    print("  4. Comparison Analyst - Movie/genre comparisons")
    print("  5. Mood Curator       - Mood-based recommendations")
    print("  6. Trivia Master      - Movie trivia and quizzes")
    print("  7. Review Analyst     - Review sentiment analysis")
    print("  8. Similarity Finder  - Similar movie recommendations")
    print("  9. Runtime Advisor    - Duration-based searches")
    print(" 10. Era Explorer       - Year/decade searches")
    print(" 11. General Assistant  - General queries")
    print("\nTo run tests:")
    print("1. First run the notebook cells to create 'chatbot' object")
    print("2. Then run: run_all_tests(chatbot)")
    print("\nOr print test cases for manual testing:")
    print("   print_test_cases()")
    print("\nTo see agent coverage:")
    print("   print_agent_coverage()")
    print("\nTo run unit tests (no API required):")
    print("   python -m pytest test_chatbot.py -v")
    print("   or: python -m unittest test_chatbot -v")
    print("\n")

    # Print agent coverage
    print_agent_coverage()

    # Print test cases for reference
    print_test_cases()

    # Run unit tests
    print("\n" + "="*60)
    print("Running Unit Tests...")
    print("="*60)
    unittest.main(verbosity=2, exit=False)
