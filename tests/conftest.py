"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_pdf_path() -> Path:
    """Get path to sample PDF file."""
    return Path(__file__).parent / "fixtures" / "sample_docs"


@pytest.fixture
def sample_text() -> str:
    """Get sample financial text for testing."""
    return """
    Annual Report 2023
    
    Financial Performance
    
    Total Revenue
    The company reported total revenues of $156.3 billion in fiscal year 2023, 
    representing a 3.7% increase compared to the prior year period.
    
    Operating Income
    Operating income increased to $28.9 billion, up 8.2% year-over-year.
    
    Key Financial Metrics
    - Gross Profit Margin: 41.5%
    - Operating Margin: 18.4%
    - Net Profit Margin: 22.1%
    
    Business Segments
    1. Technology Services: 45% of revenue ($70.3B)
    2. Business Solutions: 35% of revenue ($54.7B)
    3. Enterprise Products: 20% of revenue ($31.3B)
    
    Risk Factors
    The company faces several key risks:
    - Intensifying market competition in emerging markets
    - Regulatory changes affecting data privacy
    - Potential supply chain disruptions in key regions
    - Foreign exchange volatility
    """
