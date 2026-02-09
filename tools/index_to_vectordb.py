"""
Vector database indexing tool using ChromaDB.
Stores embeddings with metadata for semantic search.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    from dotenv import load_dotenv
except ImportError:
    logger.error("Required packages not installed. Run: pip install chromadb python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", ".tmp/chroma_db")


class VectorDBIndexer:
    """
    ChromaDB indexer for document chunks.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Path to ChromaDB storage directory
        """
        self.persist_directory = persist_directory or CHROMA_DB_PATH

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )

        # Cache collections to avoid repeated lookups
        self._collection_cache: Dict[str, chromadb.Collection] = {}

        logger.info(f"ChromaDB client initialized at {self.persist_directory}")

    def create_or_get_collection(self, data_room_id: str) -> chromadb.Collection:
        """
        Create or get existing collection for a data room.
        Uses caching to avoid repeated ChromaDB lookups.

        Args:
            data_room_id: Unique identifier for data room

        Returns:
            ChromaDB collection
        """
        # Check cache first (fast path)
        if data_room_id in self._collection_cache:
            return self._collection_cache[data_room_id]

        collection_name = f"data_room_{data_room_id}"

        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            logger.debug(f"Retrieved existing collection: {collection_name}")
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"data_room_id": data_room_id}
            )
            logger.info(f"Created new collection: {collection_name}")

        # Cache the collection for future use
        self._collection_cache[data_room_id] = collection
        return collection

    def index_chunks(
        self,
        data_room_id: str,
        chunks_with_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index chunks with embeddings to ChromaDB.

        Args:
            data_room_id: Data room identifier
            chunks_with_embeddings: Chunks with 'embedding' field

        Returns:
            Indexing result summary
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks to index")
            return {"indexed": 0, "collection_id": None}

        logger.info(f"Indexing {len(chunks_with_embeddings)} chunks to data room: {data_room_id}")

        # Get or create collection
        collection = self.create_or_get_collection(data_room_id)

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        skipped_count = 0
        for chunk in chunks_with_embeddings:
            # Validate chunk has embedding
            if 'embedding' not in chunk:
                logger.warning(f"Chunk {chunk.get('id')} missing embedding, skipping")
                skipped_count += 1
                continue

            embedding = chunk['embedding']

            # Validate embedding is not empty
            if not embedding:
                logger.warning(f"Chunk {chunk.get('id')} has empty embedding, skipping")
                skipped_count += 1
                continue

            # Validate embedding is not all zeros (failed embedding generation)
            # Fast O(1) check instead of O(n) - sample first, last, and middle values
            if embedding[0] == 0 and embedding[-1] == 0 and embedding[len(embedding)//2] == 0:
                logger.warning(f"Chunk {chunk.get('id')} has all-zero embedding (likely failed generation), skipping")
                skipped_count += 1
                continue

            ids.append(chunk['id'])
            embeddings.append(chunk['embedding'])
            documents.append(chunk['chunk_text'])

            # Prepare metadata (ChromaDB requires flat dict, no nested objects)
            metadata = {
                'chunk_index': chunk.get('chunk_index', 0),
                'token_count': chunk.get('token_count', 0),
                'file_name': (chunk.get('metadata') or {}).get('file_name', ''),
                'file_type': (chunk.get('metadata') or {}).get('file_type', ''),
                'page_number': chunk.get('page_number') or (chunk.get('metadata') or {}).get('page_number', 0),
                'section_title': chunk.get('section_title') or (chunk.get('metadata') or {}).get('section_title', ''),
                'chunk_type': chunk.get('chunk_type') or (chunk.get('metadata') or {}).get('chunk_type', 'text'),
                'document_category': (chunk.get('metadata') or {}).get('document_category', ''),
                # Excel sheet metadata
                'sheet_name': (chunk.get('metadata') or {}).get('sheet_name', ''),
                'row_start': (chunk.get('metadata') or {}).get('row_start', 0),
                'row_end': (chunk.get('metadata') or {}).get('row_end', 0),
                'currency': (chunk.get('metadata') or {}).get('currency', ''),
            }

            # Remove None values and convert to strings (ChromaDB requirement)
            metadata = {k: str(v) if v is not None else '' for k, v in metadata.items()}

            metadatas.append(metadata)

        # Add to collection in batches (5000 is optimal for ChromaDB's WAL)
        batch_size = 5000
        total_indexed = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            try:
                # Use upsert to handle duplicates gracefully (faster than checking + add)
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_indexed += len(batch_ids)
                logger.debug(f"Indexed batch: {total_indexed}/{len(ids)}")

            except Exception as e:
                logger.error(f"Failed to index batch: {e}")

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} chunks with invalid embeddings")

        logger.success(f"Indexed {total_indexed} chunks to collection: {collection.name}")

        return {
            "indexed": total_indexed,
            "skipped": skipped_count,
            "collection_id": collection.name,
            "data_room_id": data_room_id,
            "total_chunks": collection.count()
        }

    def get_collection_stats(self, data_room_id: str) -> Dict[str, Any]:
        """Get statistics for a data room collection."""
        try:
            collection = self.create_or_get_collection(data_room_id)

            return {
                "collection_name": collection.name,
                "data_room_id": data_room_id,
                "total_chunks": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def delete_collection(self, data_room_id: str) -> bool:
        """Delete a collection (for cleanup)."""
        try:
            collection_name = f"data_room_{data_room_id}"
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False


def index_to_vectordb(
    data_room_id: str,
    chunks_with_embeddings: List[Dict[str, Any]],
    persist_directory: Optional[str] = None,
    indexer: Optional['VectorDBIndexer'] = None
) -> Dict[str, Any]:
    """
    Convenience function to index chunks to ChromaDB.

    Args:
        data_room_id: Data room identifier
        chunks_with_embeddings: Chunks with embeddings
        persist_directory: Optional ChromaDB storage path
        indexer: Optional pre-created VectorDBIndexer to reuse (avoids recreating client)

    Returns:
        Indexing result

    Example:
        >>> chunks = [...]  # Chunks with embeddings
        >>> result = index_to_vectordb("deal_123", chunks)
        >>> print(f"Indexed {result['indexed']} chunks")
    """
    if indexer is None:
        indexer = VectorDBIndexer(persist_directory=persist_directory)
    return indexer.index_chunks(data_room_id, chunks_with_embeddings)


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Index embeddings to ChromaDB")
    parser.add_argument("file", help="Path to chunks with embeddings JSON file")
    parser.add_argument("--data-room-id", required=True, help="Data room ID")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")

    args = parser.parse_args()

    # Initialize indexer
    indexer = VectorDBIndexer()

    if args.stats:
        # Show stats
        stats = indexer.get_collection_stats(args.data_room_id)
        print(f"\n{'='*60}")
        print(f"Collection Stats:")
        print(f"Data Room ID: {stats.get('data_room_id')}")
        print(f"Total Chunks: {stats.get('total_chunks')}")
        print(f"Collection Name: {stats.get('collection_name')}")
        print(f"{'='*60}\n")
    else:
        # Load chunks with embeddings
        with open(args.file, 'r') as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks from {args.file}")

        # Index to vector DB
        result = index_to_vectordb(args.data_room_id, chunks)

        print(f"\n{'='*60}")
        print(f"Indexing Complete:")
        print(f"Indexed: {result['indexed']} chunks")
        print(f"Collection ID: {result['collection_id']}")
        print(f"Total in Collection: {result['total_chunks']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
