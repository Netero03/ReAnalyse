"""LangChain RAG chain for financial document Q&A."""

from typing import List, Dict, Optional, Any

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from financial_analyzer.config import settings
from financial_analyzer.embeddings.gemini_embedder import GeminiEmbedder
from financial_analyzer.vector_store.pinecone_client import PineconeClient
from financial_analyzer.rag.prompts import (
    get_system_prompt,
    get_question_prompt,
    format_context,
    format_chat_history,
)
from financial_analyzer.utils.logger import logger
from financial_analyzer.utils.errors import RAGChainError


class RAGChain:
    """Retrieval-Augmented Generation chain for financial document Q&A."""

    def __init__(
        self,
        use_memory: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """
        Initialize RAG chain.

        Args:
            use_memory: Whether to maintain conversation memory
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        self.logger = logger
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize components
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_pinecone()
        self._initialize_chain(use_memory)

    def _initialize_llm(self) -> None:
        """Initialize Gemini LLM."""
        try:
            self.logger.info(f"Initializing LLM: {settings.model_name}")

            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=settings.google_api_key,
            )

            self.logger.info("LLM initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            raise RAGChainError(f"Failed to initialize LLM: {str(e)}") from e

    def _initialize_embeddings(self) -> None:
        """Initialize Gemini embeddings."""
        try:
            self.logger.info("Initializing embeddings")

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.google_api_key,
            )

            self.logger.info("Embeddings initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise RAGChainError(f"Failed to initialize embeddings: {str(e)}") from e

    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone vector store."""
        try:
            self.logger.info("Initializing Pinecone vector store")

            self.pinecone_client = PineconeClient()

            # Initialize LangChain wrapper
            self.vector_store = Pinecone.from_existing_index(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                namespace=settings.pinecone_namespace,
            )

            self.logger.info("Pinecone vector store initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise RAGChainError(f"Failed to initialize Pinecone: {str(e)}") from e

    def _initialize_chain(self, use_memory: bool = True) -> None:
        """Initialize RAG chain."""
        try:
            self.logger.info("Initializing RAG chain")

            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": settings.max_retrieved_chunks,
                }
            )

            # Create prompt template
            prompt = get_question_prompt()

            # Initialize memory (optional)
            self.memory = None
            if use_memory:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                )

            # Create RetrievalQA chain
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Combine documents into single prompt
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context",
                },
            )

            self.logger.info("RAG chain initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize chain: {str(e)}")
            raise RAGChainError(f"Failed to initialize chain: {str(e)}") from e

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a question to the RAG chain.

        Args:
            question: User question
            chat_history: Optional chat history for context

        Returns:
            Dictionary with 'answer' and 'source_documents'

        Raises:
            RAGChainError: If query fails
        """
        try:
            self.logger.info(f"Processing query: {question[:100]}...")

            # Add chat history context if provided
            input_text = question
            if chat_history:
                history_str = format_chat_history(chat_history)
                input_text = f"Previous conversation:\n{history_str}\n\nNew question: {question}"

            # Run chain
            result = self.chain({"query": input_text})

            # Format response
            response = {
                "answer": result.get("result", ""),
                "source_documents": result.get("source_documents", []),
                "sources": [
                    {
                        "source": str(doc.metadata.get("source", "Unknown")),
                        "content": doc.page_content[:200] + "...",
                    }
                    for doc in result.get("source_documents", [])
                ],
            }

            self.logger.info(f"Generated response from {len(response['sources'])} sources")
            return response

        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise RAGChainError(f"Failed to process query: {str(e)}") from e

    def add_to_memory(self, role: str, content: str) -> None:
        """
        Add message to conversation memory.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        if self.memory:
            from langchain.schema import HumanMessage, AIMessage

            if role.lower() == "user":
                self.memory.chat_memory.add_user_message(content)
            elif role.lower() == "assistant":
                self.memory.chat_memory.add_ai_message(content)

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            self.logger.info("Memory cleared")

    def get_memory(self) -> Optional[ConversationBufferMemory]:
        """Get conversation memory."""
        return self.memory
