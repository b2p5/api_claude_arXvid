#!/usr/bin/env python3
"""
Test script for authentication system.
Tests user registration, login, and protected endpoints.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "email": "test@example.com",
    "password": "testpassword123",
    "full_name": "Test User"
}

def test_register():
    """Test user registration."""
    print("Testing user registration...")
    
    url = f"{BASE_URL}/auth/register"
    response = requests.post(url, json=TEST_USER)
    
    if response.status_code == 200:
        data = response.json()
        print("Registration successful!")
        print(f"   User: {data['user']['email']}")
        print(f"   Token: {data['access_token'][:50]}...")
        return data['access_token']
    else:
        print(f"Registration failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_login():
    """Test user login."""
    print("\nTesting user login...")
    
    url = f"{BASE_URL}/auth/login"
    login_data = {
        "email": TEST_USER["email"],
        "password": TEST_USER["password"]
    }
    response = requests.post(url, json=login_data)
    
    if response.status_code == 200:
        data = response.json()
        print("Login successful!")
        print(f"   User: {data['user']['email']}")
        print(f"   Token: {data['access_token'][:50]}...")
        return data['access_token']
    else:
        print(f"Login failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_protected_endpoint(token):
    """Test accessing protected endpoint."""
    print("\nTesting protected endpoint...")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/auth/me"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("Protected endpoint access successful!")
        print(f"   User info: {data['user']['email']}")
        return True
    else:
        print(f"Protected endpoint access failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_documents_endpoint(token):
    """Test documents listing endpoint."""
    print("\nTesting documents endpoint...")
    
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE_URL}/my-documents"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("Documents endpoint access successful!")
        print(f"   Documents: {data.get('total_documents', len(data.get('documents', [])))}")
        print(f"   Categories: {data.get('categories', [])}")
        return True
    else:
        print(f"Documents endpoint access failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_unauthorized_access():
    """Test accessing protected endpoint without token."""
    print("\nTesting unauthorized access...")
    
    url = f"{BASE_URL}/my-documents"
    response = requests.get(url)
    
    if response.status_code == 403:
        print("Unauthorized access properly blocked!")
        return True
    else:
        print(f"Unauthorized access not properly blocked: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def main():
    """Run all authentication tests."""
    print("Running Authentication Tests")
    print("=" * 50)
    
    # Test server availability
    try:
        response = requests.get(BASE_URL)
        if response.status_code != 200:
            print("Server not available")
            return
        print("Server is running")
    except requests.ConnectionError:
        print("Cannot connect to server")
        return
    
    # Run tests
    token = test_register()
    if not token:
        # Try login instead
        token = test_login()
    
    if token:
        test_protected_endpoint(token)
        test_documents_endpoint(token)
    
    test_unauthorized_access()
    
    print("\n" + "=" * 50)
    print("Tests completed!")

if __name__ == "__main__":
    main()