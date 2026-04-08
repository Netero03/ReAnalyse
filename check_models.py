#!/usr/bin/env python3
"""Check available embedding models in Google Generative AI API."""

import google.generativeai as genai
from financial_analyzer.config import settings

genai.configure(api_key=settings.google_api_key)

# List all available models
print("Available models:")
print("-" * 60)
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"  Display Name: {model.display_name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Methods: {model.supported_generation_methods}")
    if "embedding" in model.name.lower():
        print("  *** EMBEDDING MODEL FOUND ***")
    print()
