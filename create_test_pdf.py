#!/usr/bin/env python
"""Create a test financial report PDF for testing."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create a test PDF
pdf_path = 'test_financial_report.pdf'
c = canvas.Canvas(pdf_path, pagesize=letter)

# Add content
y_position = 750
content = [
    'ANNUAL FINANCIAL REPORT 2024',
    '',
    'Company Overview:',
    'ReAnalyse Inc. is a leading financial analytics company.',
    '',
    'Financial Summary:',
    'Total Revenue: $50,000,000',
    'Net Income: $12,500,000',
    'Operating Expenses: $25,000,000',
    'EBITDA: $15,000,000',
    '',
    'Key Performance Indicators:',
    'Revenue Growth: +15% Year-over-Year',
    'Profit Margin: 25%',
    'Return on Assets: 18%',
    'Debt-to-Equity Ratio: 0.8',
    '',
    'Risk Factors:',
    'Market volatility may impact revenue',
    'Competition from larger firms',
    'Regulatory changes in the sector',
    '',
    'Future Outlook:',
    'Expected growth of 20% in next fiscal year',
    'Expansion into new markets planned',
    'Investment in R&D to continue',
]

for line in content:
    c.drawString(50, y_position, line)
    y_position -= 20

c.save()
print(f'Test PDF created: {pdf_path}')
