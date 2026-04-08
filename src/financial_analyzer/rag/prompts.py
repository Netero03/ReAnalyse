"""Prompt templates for financial document Q&A."""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


# System prompt for financial Q&A
SYSTEM_PROMPT = """You are an expert financial analyst assistant. Your role is to answer questions about financial documents such as annual reports, 10-K filings, earnings statements, and other financial disclosures.

Guidelines:
1. ALWAYS base your answers ONLY on the provided document excerpts. Do not use external knowledge not in the documents.
2. If the information requested is not in the documents, clearly state "This information is not provided in the documents."
3. Be precise with numbers, dates, and financial metrics. Always include units (dollars, percentages, etc.).
4. When quoting from documents, clearly indicate the direct quote.
5. For complex financial questions, break down your answer step by step.
6. If asked about calculations (ratios, changes, etc.), show your work.
7. Maintain a professional, analytical tone.
8. If relevant documents discuss conflicting information, mention both perspectives.

Answer concisely but comprehensively."""

# Main Q&A prompt template
QA_PROMPT_TEMPLATE = """Context from financial documents:
{context}

Question: {question}

Based ONLY on the context provided above, please answer the question comprehensively. If the answer is not in the context, state that clearly."""

# Few-shot examples for financial Q&A
FINANCIAL_EXAMPLES = [
    {
        "question": "What was the total revenue in 2023?",
        "context": "In fiscal year 2023, the company reported total revenues of $156.3 billion, representing a 3.7% increase compared to the prior year.",
        "answer": "According to the documents, the total revenue in 2023 was $156.3 billion, which represented a 3.7% increase compared to the prior year.",
    },
    {
        "question": "What are the main business segments?",
        "context": "The company operates through three primary segments: Technology Services (45% of revenue), Business Solutions (35% of revenue), and Enterprise Products (20% of revenue).",
        "answer": "The company operates through three main business segments: (1) Technology Services accounting for 45% of revenue, (2) Business Solutions accounting for 35% of revenue, and (3) Enterprise Products accounting for 20% of revenue.",
    },
    {
        "question": "What risks are mentioned in the document?",
        "context": "Key risk factors include: market competition intensifying in emerging markets, regulatory changes affecting data privacy, potential supply chain disruptions in key regions, and foreign exchange volatility.",
        "answer": "The document identifies several key risk factors: (1) intensifying market competition in emerging markets, (2) regulatory changes affecting data privacy, (3) potential supply chain disruptions in key regions, and (4) foreign exchange volatility.",
    }
]


def get_system_prompt() -> str:
    """Get system prompt for financial Q&A."""
    return SYSTEM_PROMPT


def get_question_prompt() -> PromptTemplate:
    """Get question answering prompt template."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE,
    )


def get_qa_chat_prompt() -> ChatPromptTemplate:
    """Get chat-style Q&A prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}\n\nContext from financial documents:\n{context}"),
    ])


def get_example_selector():
    """
    Get example selector for few-shot prompting.
    
    Note: LengthBasedExampleSelector is not available in LangChain 1.2+.
    This function is kept for backward compatibility but returns None.
    Few-shot examples are not used in the current implementation.
    """
    return None


def format_chat_history(messages: list) -> str:
    """
    Format chat history for context.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Formatted chat history string
    """
    formatted = []
    for msg in messages[-5:]:  # Keep last 5 messages
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def format_context(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks as context.

    Args:
        retrieved_chunks: List of retrieved document chunks

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        content = chunk.get("content", "")
        score = chunk.get("score", 0)

        context_parts.append(
            f"[Document {i} - {source} (Relevance: {score:.2f})]\n{content}\n"
        )

    return "\n".join(context_parts)
