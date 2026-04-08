"""LangChain RAG chain for financial document Q&A."""

from typing import List, Dict, Optional, Any

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone

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
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """
        Initialize RAG chain.

        Args:
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
        self._initialize_chain()

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

            # Initialize LangChain wrapper (handles Pinecone connection internally)
            self.vector_store = Pinecone.from_existing_index(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
                namespace=settings.pinecone_namespace,
            )

            self.logger.info("Pinecone vector store initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise RAGChainError(f"Failed to initialize Pinecone: {str(e)}") from e

    def _initialize_chain(self) -> None:
        """Initialize RAG chain using LCEL."""
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

            # Build the chain
            # Uses LCEL (LangChain Expression Language) to:
            # 1. Pass through the input question
            # 2. Retrieve documents using the retriever
            # 3. Format documents into context string
            # 4. Create prompt with question and context
            # 5. Run through LLM
            # 6. Parse the output as string
            
            def _format_docs(docs):
                """Format retrieved documents into context string."""
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Store docs retrieval for source extraction
            self.docs_retriever = self.retriever
            
            # Build the chain using LCEL
            self.chain = (
                RunnablePassthrough.assign(
                    context=self.retriever | _format_docs
                )
                | prompt
                | self.llm
                | StrOutputParser()
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

            # Run chain to get answer
            answer = self.chain.invoke({"question": input_text})

            # Retrieve source documents separately
            source_documents = self.docs_retriever.invoke(question)

            # Format response
            response = {
                "answer": answer,
                "source_documents": source_documents,
                "sources": [
                    {
                        "source": str(doc.metadata.get("source", "Unknown")),
                        "content": doc.page_content[:200] + "...",
                    }
                    for doc in source_documents
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
        
        Note: Memory management is now handled by the UI layer (Streamlit).
        This method is kept for backward compatibility but is a no-op.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        # Memory management is now handled at the UI layer
        pass

    def clear_memory(self) -> None:
        """
        Clear conversation memory.
        
        Note: Memory management is now handled by the UI layer (Streamlit).
        This method is kept for backward compatibility but is a no-op.
        """
        # Memory management is now handled at the UI layer
        pass

    def get_memory(self) -> None:
        """
        Get conversation memory.
        
        Note: Memory management is now handled by the UI layer (Streamlit).
        This method is kept for backward compatibility.
        
        Returns:
            None (memory is handled at UI layer)
        """
        return None
