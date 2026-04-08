#!/usr/bin/env python3
"""Test Pinecone SDK 8.1.1 API."""

from pinecone import Pinecone
from financial_analyzer.config import settings

# Initialize Pinecone
pc = Pinecone(api_key=settings.google_api_key)

# Check what methods are available for getting an index
print("Methods to access index:")
print(f"  - pc.Index: {hasattr(pc, 'Index')}")
print(f"  - pc.index: {hasattr(pc, 'index')}")

# Check Index class
print("\nIndex class info:")
print(f"  - type(pc.Index): {type(pc.Index) if hasattr(pc, 'Index') else 'N/A'}")

# Try to list indexes
print("\nListing indexes:")
try:
    indexes = pc.list_indexes()
    for idx in indexes:
        print(f"  - {idx.name}")
except Exception as e:
    print(f"  Error listing indexes: {e}")

# Try to access an index
print("\nTrying to access index:")
try:
    if hasattr(pc, 'Index'):
        index = pc.Index("financial-documents")
        print(f"  Success! Index type: {type(index)}")
    else:
        print("  pc.Index not available")
except Exception as e:
    print(f"  Error: {e}")
