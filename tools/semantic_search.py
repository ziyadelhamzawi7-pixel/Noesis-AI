"""
Semantic search tool using ChromaDB vector similarity.
Retrieves relevant document chunks for analyst queries.
Supports both local (fastembed) and OpenAI embeddings for query encoding.
"""

import sys
import os
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    logger.error("Required packages not installed. Run: pip install chromadb openai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", ".tmp/chroma_db")

# Resolve embedding provider
try:
    from app.config import settings as _app_settings
    _EMBEDDING_PROVIDER = _app_settings.embedding_provider
except ImportError:
    _EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# Thread-safe singleton cache for SemanticSearch instances
_instance_cache: Dict[str, "SemanticSearch"] = {}
_instance_lock = threading.Lock()


def _get_cached_instance(persist_directory: Optional[str] = None) -> "SemanticSearch":
    """Get or create a cached SemanticSearch instance for the given directory."""
    key = persist_directory or CHROMA_DB_PATH
    if key not in _instance_cache:
        with _instance_lock:
            if key not in _instance_cache:
                _instance_cache[key] = SemanticSearch(persist_directory=persist_directory)
    return _instance_cache[key]


class SemanticSearch:
    """
    Semantic search using ChromaDB and OpenAI embeddings.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize semantic search.

        Args:
            persist_directory: ChromaDB storage path
            openai_api_key: OpenAI API key (only needed when EMBEDDING_PROVIDER=openai)
            embedding_model: Model for query embeddings (OpenAI only)
        """
        self.persist_directory = persist_directory or CHROMA_DB_PATH
        self._use_local = _EMBEDDING_PROVIDER == "local"

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        if self._use_local:
            # Local embeddings — no OpenAI key needed for search
            self._local_generator = None  # Lazy init
            logger.info("Semantic search initialized with local embeddings (fastembed)")
        else:
            # OpenAI embeddings
            self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self._openai_api_key:
                raise ValueError("OpenAI API key not found")
            self._thread_local = threading.local()
            self.embedding_model = embedding_model
            logger.info(f"Semantic search initialized with OpenAI model: {embedding_model}")

    def _get_openai_client(self) -> OpenAI:
        """Get a thread-local OpenAI client (httpx.Client is not thread-safe)."""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = OpenAI(api_key=self._openai_api_key)
        return self._thread_local.client

    def search(
        self,
        query: str,
        data_room_id: str,
        top_k: int = 15,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using vector similarity.

        Args:
            query: Search query text
            data_room_id: Data room to search in
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"document_category": "financials"})

        Returns:
            List of relevant chunks with similarity scores
        """
        logger.info(f"Searching data room {data_room_id}: '{query[:100]}...'")

        try:
            # Get collection
            collection_name = f"data_room_{data_room_id}"
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except Exception:
                logger.warning(f"Collection '{collection_name}' not found — data room has no indexed documents")
                return []

            chunk_count = collection.count()
            logger.debug(f"Collection has {chunk_count} chunks")

            if chunk_count == 0:
                logger.warning(f"Collection '{collection_name}' exists but is empty — documents may have failed to process")
                return []

            # Generate query embedding
            query_embedding = self._embed_query(query)

            # Prepare where filter for ChromaDB
            where_filter = None
            if filters:
                where_filter = self._build_where_filter(filters)

            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, chunk_count),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = self._format_results(results, query)

            logger.success(f"Found {len(formatted_results)} relevant chunks")

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query using the configured provider."""
        if self._use_local:
            return self._embed_query_local(query)
        return self._embed_query_openai(query)

    def _embed_query_local(self, query: str) -> List[float]:
        """Generate query embedding using local fastembed model."""
        try:
            if self._local_generator is None:
                from tools.generate_embeddings import LocalEmbeddingGenerator
                self._local_generator = LocalEmbeddingGenerator()
            return self._local_generator.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query (local): {e}")
            raise

    def _embed_query_openai(self, query: str) -> List[float]:
        """Generate query embedding using OpenAI API."""
        try:
            response = self._get_openai_client().embeddings.create(
                model=self.embedding_model,
                input=query
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to embed query (OpenAI): {e}")
            raise

    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where filter from user filters.

        Args:
            filters: User-provided filters

        Returns:
            ChromaDB where clause
        """
        # ChromaDB where clause format
        where = {}

        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values (OR condition)
                where[key] = {"$in": value}
            else:
                # Single value (equality)
                where[key] = value

        return where

    def _format_results(self, results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Format ChromaDB results into clean dictionaries."""
        formatted = []

        # ChromaDB returns lists for batch queries
        ids = results['ids'][0] if results['ids'] else []
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []

        for i, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # Convert distance to similarity score (0-1, higher is better)
            # ChromaDB uses L2 distance, convert to similarity
            similarity_score = 1 / (1 + distance)

            formatted.append({
                'rank': i + 1,
                'chunk_id': chunk_id,
                'chunk_text': document,
                'similarity_score': round(similarity_score, 4),
                'distance': round(distance, 4),
                'metadata': metadata,
                'source': {
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'page_number': int(metadata.get('page_number', 0)) if metadata.get('page_number') else None,
                    'section_title': metadata.get('section_title', ''),
                    'document_category': metadata.get('document_category', ''),
                    # Excel sheet metadata
                    'sheet_name': metadata.get('sheet_name', '') or None,
                    'row_start': int(metadata.get('row_start', 0)) if metadata.get('row_start') else None,
                    'row_end': int(metadata.get('row_end', 0)) if metadata.get('row_end') else None,
                    'currency': metadata.get('currency', '') or None,
                }
            })

        return formatted

    def list_collections(self) -> List[str]:
        """List all data room collections."""
        collections = self.chroma_client.list_collections()
        return [c.name for c in collections]


def semantic_search(
    query: str,
    data_room_id: str,
    top_k: int = 15,
    filters: Optional[Dict[str, Any]] = None,
    persist_directory: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for semantic search.

    Args:
        query: Search query
        data_room_id: Data room ID
        top_k: Number of results
        filters: Optional metadata filters
        persist_directory: Optional ChromaDB path

    Returns:
        List of relevant chunks

    Example:
        >>> results = semantic_search(
        ...     query="What is the revenue model?",
        ...     data_room_id="deal_123",
        ...     top_k=10
        ... )
        >>> for result in results:
        ...     print(f"{result['source']['file_name']}: {result['chunk_text'][:100]}")
    """
    searcher = _get_cached_instance(persist_directory=persist_directory)
    return searcher.search(query, data_room_id, top_k, filters)


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search in data room")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--data-room-id", required=True, help="Data room ID")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--filter-category", help="Filter by document category")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Show full results")

    args = parser.parse_args()

    # Build filters
    filters = {}
    if args.filter_category:
        filters['document_category'] = args.filter_category

    # Search
    results = semantic_search(
        query=args.query,
        data_room_id=args.data_room_id,
        top_k=args.top_k,
        filters=filters if filters else None
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        logger.info(f"Saved {len(results)} results to {output_path}")
    else:
        print(f"\n{'='*60}")
        print(f"Query: {args.query}")
        print(f"Found: {len(results)} results")
        print(f"{'='*60}\n")

        for result in results:
            print(f"Rank {result['rank']}: Score {result['similarity_score']:.3f}")
            print(f"Source: {result['source']['file_name']}", end="")
            if result['source']['page_number']:
                print(f", Page {result['source']['page_number']}")
            else:
                print()

            if args.verbose:
                print(f"Text: {result['chunk_text'][:300]}...")
            else:
                print(f"Preview: {result['chunk_text'][:150]}...")

            print("-" * 60)


if __name__ == "__main__":
    main()
