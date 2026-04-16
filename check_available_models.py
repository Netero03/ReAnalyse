#!/usr/bin/env python3
"""Check available Google Generative AI models."""

import google.generativeai as genai
from financial_analyzer.config import settings

genai.configure(api_key=settings.google_api_key)

print("Available models with generateContent support:")
print("-" * 60)
generate_models = []
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"  - {model.name}: {model.display_name}")
        generate_models.append(model.name)

print("\n" + "=" * 60)
print("Recommended model to use:")
# Find a gemini model
gemini_models = [m for m in generate_models if "gemini" in m.lower()]
if gemini_models:
    print(f"  {gemini_models[0]}")
else:
    print(f"  {generate_models[0] if generate_models else 'No models found'}")
