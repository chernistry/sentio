import os
import requests
import pytest
import time
import logging
import sys
import json
import argparse
from requests.exceptions import ConnectionError

# Pretty formatting of JSON answers
from src.core.llm_reply_extractor import format_llm_response

API_URL = os.getenv("SENTIO_API_URL", "http://localhost:8000")

# Command line testing functionality
def run_cli_test(question, verbose=False):
    """Run a test query via command line."""
    print(f"\n[CLI TEST] Query: {question}")
    
    try:
        start = time.time()
        payload = {"question": question}
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Response received in {elapsed:.2f}s")
            
            # Show raw response in verbose mode
            if verbose:
                print("\n[RAW RESPONSE]")
                print(json.dumps(data, indent=2))
            
            # Pre-process answer to remove any prompting artifacts
            answer = data.get("answer", "")
            if "-----" in answer:
                # Remove everything after the divider (often contains self-instructions)
                answer = answer.split("-----")[0].strip()
                data["answer"] = answer
            
            # Show formatted response
            print(f"\n{format_llm_response(data)}")
            
            # Print detailed source information
            if "sources" in data and data["sources"]:
                print(f"\n📚 Sources ({len(data['sources'])}):")
                for i, src in enumerate(data["sources"]):
                    source_file = src.get("source", "Unknown")
                    page = src.get("metadata", {}).get("page_label", "")
                    page_info = f"(page {page})" if page else ""
                    
                    print(f"  [{i+1}] Score: {src.get('score', 'N/A'):.4f}")
                    print(f"      File: {source_file} {page_info}")
                    
                    # Format title if exists
                    title = src.get("title", "") or src.get("metadata", {}).get("title", "")
                    if title:
                        print(f"      Title: {title}")
                    
                    # Show excerpt in a clean format
                    if "text" in src:
                        # Clean up the text
                        text = src["text"]
                        text = text.replace("\t", " ").replace("\n", " ")
                        text = " ".join(text.split()) # Normalize whitespace
                        
                        excerpt = text[:150] + "..." if len(text) > 150 else text
                        print(f"      Excerpt: \"{excerpt}\"")
                    print()
            return True
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

def main():
    """Command-line entry point for direct testing."""
    parser = argparse.ArgumentParser(description="Test the Sentio RAG API")
    parser.add_argument("--question", "-q", type=str, help="Question to ask")
    parser.add_argument("--preset", "-p", type=str, choices=["rag", "osint", "all"], 
                       help="Preset question to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    # Handle preset questions
    if args.preset == "rag":
        run_cli_test("What is RAG? Describe a typical RAG pipeline in a single line as 'stage1->stage2->…stageN'.", args.verbose)
    elif args.preset == "osint":
        run_cli_test("What does OSINT stand for and what is it used for?", args.verbose)
    elif args.preset == "all":
        run_cli_test("What is RAG? Describe a typical RAG pipeline in a single line as 'stage1->stage2->…stageN'.", args.verbose)
        print("\n" + "-" * 80 + "\n")
        run_cli_test("What does OSINT stand for and what is it used for?", args.verbose)
    # Custom question
    elif args.question:
        run_cli_test(args.question, args.verbose)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


def check_api_availability():
    """Check if the API is available and properly configured."""
    try:
        # Try to access the API docs endpoint which should be available
        response = requests.get(f"{API_URL}/docs", timeout=5)
        if response.status_code != 200:
            print(f"API is not responding correctly. Status code: {response.status_code}")
            return False
        return True
    except Exception as e:
        print(f"API is not available: {str(e)}")
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_api_available():
    """Fixture to ensure API is available before running tests."""
    if not check_api_availability():
        pytest.skip("API is not available or properly configured. Skipping tests.")


def test_chat_endpoint():
    """Basic end-to-end check that /chat returns answer with citations."""
    payload = {"question": "What's OSINT?"}
    print(f"\n[TEST] test_chat_endpoint: {payload}")
    
    # Add retry logic for connection issues
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
            elapsed = time.time() - start
            print(f"[RESPONSE] status={response.status_code} time={elapsed:.2f}s body={response.text[:200]}")

            # Validate HTTP status
            assert response.status_code == 200, f"Unexpected status {response.status_code}: {response.text}"

            data = response.json()

            # Pretty-print model reply for human readability
            print(format_llm_response(data))

            # Validate response structure
            assert "answer" in data and data["answer"].strip(), "Answer missing or empty"
            assert "sources" in data and isinstance(data["sources"], list), "Sources missing or invalid"
            assert len(data["sources"]) >= 1, "No citations returned"

            # Validate each citation score
            for src in data["sources"]:
                score = src.get("score", 0)
                assert score >= 0.05, f"Citation score below threshold: {score}"
            
            # If we get here, test passed
            break
            
        except ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Connection error: {str(e)}. Retrying in {retry_delay} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                # Last attempt failed, re-raise the exception
                print(f"[FAILED] All {max_retries} attempts failed with connection errors")
                raise


def test_relevance():
    """Test the relevance of the responses for specific queries."""
    query_relevance_tests = [
        {
            "question": "What's OSINT?", 
            "expected_terms": [
                "OSINT", "open source intelligence", "information", "sources", 
                "publicly", "available", "data", "collection", "analysis", 
                "intelligence", "gathering"
            ]
        },
        {
            "question": "How do RAG systems work?", 
            "expected_terms": [
                "retrieval", "generation", "documents", "vector", "embedding", 
                "search", "context", "language model", "LLM", "database", 
                "knowledge", "augmented", "RAG", "query"
            ]
        },
    ]
    
    # Add retry logic for connection issues
    max_retries = 3
    retry_delay = 5
    
    for test in query_relevance_tests:
        payload = {"question": test["question"]}
        print(f"\n[TEST] test_relevance: {payload}")
        
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                elapsed = time.time() - start
                print(f"[RESPONSE] status={response.status_code} time={elapsed:.2f}s body={response.text[:200]}")
                
                assert response.status_code == 200, f"Unexpected status {response.status_code}: {response.text}"
                
                data = response.json()

                # Pretty-print model reply
                print(format_llm_response(data))
                
                # Check the answer for expected terms
                answer = data["answer"].lower()
                found_terms = [term for term in test["expected_terms"] if term.lower() in answer]
                print(f"[DEBUG] found_terms={found_terms}")
                assert len(found_terms) >= 1, (
                    f"Answer doesn't contain any expected terms for query: {test['question']}\n"
                    f"Expected any of: {test['expected_terms']}\n"
                    f"Answer: {data['answer'][:150]}..."
                )
                
                # Check that at least one source has a high relevance score
                sources = data["sources"]
                high_score = [src for src in sources if src.get("score", 0) >= 0.3]
                print(f"[DEBUG] high_score_sources={len(high_score)}")
                assert any(src.get("score", 0) >= 0.3 for src in sources), (
                    f"No highly relevant sources (score >= 0.3) found for query: {test['question']}"
                )
                
                # If we get here, test passed
                break
                
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY] Connection error: {str(e)}. Retrying in {retry_delay} seconds... (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed, re-raise the exception
                    print(f"[FAILED] All {max_retries} attempts failed with connection errors")
                    raise


def test_citation_accuracy():
    """Test that citations are actually relevant to the query."""
    payload = {
        "question": "Tell me about OSINT techniques",
        "top_k": 5
    }
    print(f"\n[TEST] test_citation_accuracy: {payload}")
    
    # Add retry logic for connection issues
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
            elapsed = time.time() - start
            print(f"[RESPONSE] status={response.status_code} time={elapsed:.2f}s body={response.text[:200]}")
            
            assert response.status_code == 200
            
            data = response.json()

            # Pretty-print model reply
            print(format_llm_response(data))
            
            # Check that sources actually contain relevant text
            sources = data["sources"]
            assert len(sources) >= 1, "No citations returned"
            
            # Verify that at least one source mentions OSINT
            osint_sources = 0
            
            for src in sources:
                text = src.get("text", "").lower()
                if "osint" in text:
                    osint_sources += 1
            
            print(f"[DEBUG] osint_sources={osint_sources} of {len(sources)}")
            assert osint_sources >= 1, f"No sources containing 'OSINT' found in citations. Found {len(sources)} total sources."
            
            # Check that the answer contains OSINT
            assert "osint" in data["answer"].lower(), "Answer does not mention OSINT"
            
            # If we get here, test passed
            break
            
        except ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Connection error: {str(e)}. Retrying in {retry_delay} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                # Last attempt failed, re-raise the exception
                print(f"[FAILED] All {max_retries} attempts failed with connection errors")
                raise 