#!/usr/bin/env python3
"""
Test script for Paper Service.
Tests arXiv search, download, and management functionality.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "email": "test@example.com",
    "password": "testpassword123"
}

def get_auth_token():
    """Get authentication token."""
    print("Getting authentication token...")
    
    url = f"{BASE_URL}/auth/login"
    response = requests.post(url, json=TEST_USER)
    
    if response.status_code == 200:
        data = response.json()
        token = data['access_token']
        print(f"Token obtained: {token[:50]}...")
        return token
    else:
        print(f"Login failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_search_papers(token):
    """Test paper search functionality."""
    print("\nTesting paper search...")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/papers/search"
    
    # Test search for machine learning papers
    params = {
        "query": "machine learning transformers",
        "max_results": 5,
        "sort_by": "relevance"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("Paper search successful!")
        print(f"   Query: {data['query']}")
        print(f"   Results found: {data['total']}")
        
        if data['results']:
            first_paper = data['results'][0]
            print(f"   First paper: {first_paper['title'][:60]}...")
            print(f"   ArXiv ID: {first_paper['arxiv_id']}")
            print(f"   Authors: {', '.join(first_paper['authors'][:2])}")
            return first_paper['arxiv_id']  # Return first paper ID for download test
        
        return None
    else:
        print(f"Paper search failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_download_paper(token, arxiv_id):
    """Test paper download functionality."""
    print(f"\nTesting paper download for {arxiv_id}...")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/papers/download"
    
    download_data = {
        "arxiv_id": arxiv_id,
        "category": "test_papers"
    }
    
    response = requests.post(url, json=download_data, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("Paper download successful!")
        print(f"   Message: {data['message']}")
        print(f"   Paper title: {data['paper']['title'][:60]}...")
        print(f"   File size: {data['paper']['file_size']} bytes")
        print(f"   Processing: {data['processing']}")
        return True
    else:
        print(f"Paper download failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_list_downloaded_papers(token):
    """Test listing downloaded papers."""
    print("\nTesting list downloaded papers...")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/papers/downloaded"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("List downloaded papers successful!")
        print(f"   Total papers: {data['total']}")
        print(f"   User: {data['username']}")
        
        for paper in data['papers']:
            print(f"   - {paper['arxiv_id']}: {paper['title'][:50]}...")
        
        return data['papers']
    else:
        print(f"List papers failed: {response.status_code}")
        print(f"Error: {response.text}")
        return []

def test_chat_with_papers(token):
    """Test chatting with downloaded papers."""
    print("\nTesting chat with papers...")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/chat"
    
    chat_data = {
        "query": "What are the main contributions of these papers?",
        "category": "test_papers"
    }
    
    response = requests.post(url, json=chat_data, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("Chat with papers successful!")
        print(f"   Response: {data['response'][:200]}...")
        return True
    else:
        print(f"Chat failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def main():
    """Run all paper service tests."""
    print("Running Paper Service Tests")
    print("=" * 50)
    
    # Get authentication token
    token = get_auth_token()
    if not token:
        print("Cannot proceed without authentication")
        return
    
    # Test search
    arxiv_id = test_search_papers(token)
    
    if arxiv_id:
        # Test download
        download_success = test_download_paper(token, arxiv_id)
        
        if download_success:
            # Wait a bit for processing
            print("\nWaiting 10 seconds for paper processing...")
            time.sleep(10)
            
            # Test listing
            papers = test_list_downloaded_papers(token)
            
            # Test chat if papers exist
            if papers:
                test_chat_with_papers(token)
    else:
        print("Skipping download test - no papers found")
    
    print("\n" + "=" * 50)
    print("Paper Service tests completed!")

if __name__ == "__main__":
    main()