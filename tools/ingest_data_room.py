"""
Multi-file data room ingestion tool.
Orchestrates the complete pipeline: parse → chunk → embed → index.
"""

import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.parse_pdf import parse_pdf
from tools.parse_excel import parse_excel
from tools.chunk_documents import chunk_document
from tools.generate_embeddings import generate_embeddings, generate_embeddings_streaming
from tools.index_to_vectordb import index_to_vectordb, VectorDBIndexer
from app import database as db
from app.config import settings


def _process_single_file(
    file_path: str,
    data_room_id: str,
    update_progress: bool
) -> Dict[str, Any]:
    """Parse and chunk a single file. Returns dict with 'chunks', 'error', 'file_name'."""
    file_path_obj = Path(file_path)
    result = {'file_name': file_path_obj.name, 'chunks': [], 'error': None}

    if not file_path_obj.exists():
        result['error'] = f"File not found: {file_path}"
        return result

    file_type = file_path_obj.suffix.lower()

    # Get document ID from database
    documents = db.get_documents_by_data_room(data_room_id)
    document_id = None
    for doc in documents:
        if doc['file_name'] == file_path_obj.name:
            document_id = doc['id']
            break

    if not document_id:
        result['error'] = f"Document record not found for {file_path_obj.name}"
        return result

    if update_progress:
        db.update_document_status(document_id, "parsing")

    # Parse based on file type
    if file_type == '.pdf':
        parsed = parse_pdf(file_path)
    elif file_type in ['.xlsx', '.xls', '.csv']:
        parsed = parse_excel(file_path)
    else:
        error_msg = f"Unsupported file type: {file_type}"
        if update_progress:
            db.update_document_status(document_id, "failed", error_message=error_msg)
        result['error'] = error_msg
        return result

    # Chunk the document
    logger.info(f"Chunking {file_path_obj.name}...")
    chunks = chunk_document(
        text=parsed['text'],
        document_id=document_id,
        data_room_id=data_room_id,
        metadata=parsed.get('metadata', {})
    )

    # Update document status to parsed
    if update_progress:
        db.update_document_status(
            document_id, "parsed",
            page_count=parsed.get('metadata', {}).get('page_count'),
            token_count=sum(c.get('token_count', 0) for c in chunks)
        )

    logger.success(f"Parsed {file_path_obj.name}: {len(chunks)} chunks")
    result['chunks'] = chunks
    return result


def ingest_data_room(
    data_room_id: str,
    file_paths: List[str],
    update_progress: bool = True
) -> Dict[str, Any]:
    """
    Ingest and process a complete data room.

    Pipeline:
    1. Parse all documents (PDF, Excel, etc.)
    2. Chunk text into semantic units
    3. Generate embeddings for all chunks
    4. Index to vector database
    5. Save metadata to SQLite

    Args:
        data_room_id: Unique identifier for the data room
        file_paths: List of file paths to process
        update_progress: Whether to update database progress

    Returns:
        Dictionary with processing results and statistics
    """
    start_time = time.time()
    results = {
        'data_room_id': data_room_id,
        'total_files': len(file_paths),
        'parsed_files': 0,
        'failed_files': 0,
        'total_chunks': 0,
        'errors': []
    }

    logger.info(f"Starting data room ingestion: {data_room_id}")
    logger.info(f"Files to process: {len(file_paths)}")

    # Update status to parsing
    if update_progress:
        db.update_data_room_status(data_room_id, "parsing", progress=5)
        db.log_processing_stage(
            "ingestion", "started",
            data_room_id=data_room_id,
            message=f"Processing {len(file_paths)} files"
        )

    all_chunks = []

    # Step 1: Parse all documents in parallel
    max_workers = min(settings.max_parse_workers, len(file_paths))
    logger.info(f"Step 1/4: Parsing documents ({max_workers} workers)...")

    completed_count = 0
    progress_lock = threading.Lock()

    def _on_file_done(file_result):
        nonlocal completed_count
        with progress_lock:
            completed_count += 1
            if update_progress:
                progress = 5 + (completed_count / len(file_paths)) * 40
                db.update_data_room_status(data_room_id, "parsing", progress=progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_process_single_file, fp, data_room_id, update_progress): fp
            for fp in file_paths
        }

        for future in as_completed(future_to_path):
            fp = future_to_path[future]
            try:
                file_result = future.result()
                _on_file_done(file_result)

                if file_result['error']:
                    logger.error(f"Failed: {file_result['file_name']}: {file_result['error']}")
                    results['errors'].append({
                        'file': file_result['file_name'],
                        'error': file_result['error']
                    })
                    results['failed_files'] += 1
                else:
                    all_chunks.extend(file_result['chunks'])
                    results['parsed_files'] += 1

            except Exception as e:
                _on_file_done({'file_name': Path(fp).name})
                error_msg = f"Failed to parse {fp}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append({
                    'file': Path(fp).name,
                    'error': str(e)
                })
                results['failed_files'] += 1

    if not all_chunks:
        error_msg = "No chunks extracted from any document"
        logger.error(error_msg)
        if update_progress:
            db.update_data_room_status(data_room_id, "failed")
            db.log_processing_stage(
                "ingestion", "failed",
                data_room_id=data_room_id,
                error_details=error_msg
            )
        raise Exception(error_msg)

    results['total_chunks'] = len(all_chunks)
    logger.info(f"Total chunks extracted: {len(all_chunks)}")

    # Steps 2-4: Stream embeddings → index → save in batches (pipelined)
    logger.info("Step 2/4: Generating embeddings + indexing (pipelined)...")

    if update_progress:
        db.update_data_room_status(data_room_id, "indexing", progress=50)

    collection_name = f"data_room_{data_room_id}"
    total_indexed = 0
    total_saved = 0
    total_embedding_failures = 0

    # FIX #3: Create indexer ONCE to reuse collection cache
    indexer = VectorDBIndexer()

    # FIX #1 & #4: Use thread pool for parallel writes and async status updates
    write_executor = ThreadPoolExecutor(max_workers=3)  # 2 for writes + 1 for status

    try:
        for batch in generate_embeddings_streaming(
            chunks=all_chunks,
            batch_size=settings.batch_size,
            max_concurrent=settings.embedding_max_concurrent
        ):
            # Separate successful and failed embeddings
            valid_chunks = [c for c in batch if c.get('embedding') is not None and not c.get('embedding_failed')]
            failed_chunks = [c for c in batch if c.get('embedding') is None or c.get('embedding_failed')]
            total_embedding_failures += len(failed_chunks)

            if failed_chunks:
                logger.warning(f"{len(failed_chunks)} chunks failed embedding generation in this batch")

            # FIX #1: Run VectorDB and SQLite writes IN PARALLEL (not sequential)
            futures = []

            # Index only chunks with valid embeddings
            if valid_chunks:
                futures.append(write_executor.submit(
                    indexer.index_chunks, data_room_id, valid_chunks
                ))

            # Save all chunks to SQLite (so we can re-embed failed ones later)
            futures.append(write_executor.submit(db.create_chunks, batch))

            # Wait for both writes to complete
            for future in futures:
                future.result()

            total_indexed += len(valid_chunks)
            total_saved += len(batch)

            # FIX #4: Update progress asynchronously (don't block main loop)
            if update_progress:
                progress = 50 + (total_saved / len(all_chunks)) * 35
                write_executor.submit(
                    db.update_data_room_status, data_room_id, "indexing", progress
                )

            logger.info(f"Indexed {total_indexed}, saved {total_saved}/{len(all_chunks)} chunks")
    finally:
        write_executor.shutdown(wait=True)

    if total_embedding_failures > 0:
        error_msg = f"{total_embedding_failures}/{len(all_chunks)} chunks failed embedding generation"
        logger.error(error_msg)
        results['errors'].append({'file': 'embeddings', 'error': error_msg})
        results['embedding_failures'] = total_embedding_failures

    logger.success(f"Indexed {total_indexed} chunks to {collection_name}")
    logger.success(f"Saved {total_saved} chunks to database")

    # Mark as complete or failed
    duration_ms = int((time.time() - start_time) * 1000)
    duration_sec = duration_ms / 1000

    if update_progress:
        from datetime import datetime
        if total_embedding_failures > 0 and total_indexed == 0:
            # All embeddings failed — mark as failed
            db.update_data_room_status(
                data_room_id,
                "failed",
                error_message=f"All {total_embedding_failures} chunks failed embedding generation"
            )
            db.log_processing_stage(
                "ingestion", "failed",
                data_room_id=data_room_id,
                error_details=f"Embedding generation failed for all {total_embedding_failures} chunks",
                duration_ms=duration_ms
            )
        else:
            status = "complete" if total_embedding_failures == 0 else "complete"
            db.update_data_room_status(
                data_room_id,
                status,
                progress=100,
                completed_at=datetime.now().isoformat()
            )
            message = f"Successfully processed {results['parsed_files']} files"
            if total_embedding_failures > 0:
                message += f" ({total_embedding_failures} chunks failed embedding)"
            db.log_processing_stage(
                "ingestion", "completed",
                data_room_id=data_room_id,
                message=message,
                duration_ms=duration_ms
            )

    results['duration_seconds'] = duration_sec
    results['collection_name'] = collection_name

    logger.success(f"Data room ingestion complete in {duration_sec:.2f}s")
    logger.info(f"Results: {results['parsed_files']} parsed, {results['failed_files']} failed, {results['total_chunks']} chunks")

    return results


def reembed_data_room(data_room_id: str) -> Dict[str, Any]:
    """
    Re-generate embeddings for chunks that failed embedding generation.
    Queries chunks without embedding_id from SQLite, generates embeddings,
    and indexes them to ChromaDB.

    Args:
        data_room_id: Data room ID to re-embed

    Returns:
        Dictionary with re-embedding results
    """
    logger.info(f"Starting re-embedding for data room: {data_room_id}")

    # Get chunks without embeddings
    chunks = db.get_chunks_without_embeddings(data_room_id)
    if not chunks:
        logger.info(f"No chunks need re-embedding for {data_room_id}")
        return {'reembedded': 0, 'total': 0, 'status': 'no_action_needed'}

    logger.info(f"Found {len(chunks)} chunks without embeddings for {data_room_id}")

    # Update status
    db.update_data_room_status(data_room_id, "indexing", progress=50)

    total_indexed = 0
    total_failed = 0

    for batch in generate_embeddings_streaming(
        chunks=chunks,
        batch_size=settings.batch_size,
        max_concurrent=settings.embedding_max_concurrent
    ):
        valid_chunks = [c for c in batch if c.get('embedding') is not None and not c.get('embedding_failed')]
        failed_in_batch = len(batch) - len(valid_chunks)
        total_failed += failed_in_batch

        if valid_chunks:
            result = index_to_vectordb(
                data_room_id=data_room_id,
                chunks_with_embeddings=valid_chunks
            )
            total_indexed += result.get('indexed', 0)

            # Update embedding_id in SQLite for successfully embedded chunks
            for chunk in valid_chunks:
                if chunk.get('id'):
                    db.update_chunk_embedding_id(chunk['id'], chunk['id'])

        logger.info(f"Re-embedded {total_indexed}/{len(chunks)} chunks")

    # Update status
    from datetime import datetime
    if total_failed == 0:
        db.update_data_room_status(
            data_room_id, "complete", progress=100,
            completed_at=datetime.now().isoformat()
        )
    else:
        db.update_data_room_status(
            data_room_id, "complete", progress=100,
            completed_at=datetime.now().isoformat(),
            error_message=f"{total_failed} chunks still failed embedding"
        )

    logger.success(f"Re-embedding complete: {total_indexed} indexed, {total_failed} failed")

    return {
        'reembedded': total_indexed,
        'failed': total_failed,
        'total': len(chunks),
        'status': 'complete' if total_failed == 0 else 'partial'
    }


def main():
    """CLI interface for data room ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest and process a data room with multiple documents"
    )
    parser.add_argument(
        "data_room_id",
        help="Data room ID (e.g., dr_abc123)"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Files to process (PDF, Excel, etc.)"
    )
    parser.add_argument(
        "--no-db-update",
        action="store_true",
        help="Don't update database progress (for testing)"
    )

    args = parser.parse_args()

    # Validate files exist
    for file_path in args.files:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)

    try:
        # Process data room
        results = ingest_data_room(
            data_room_id=args.data_room_id,
            file_paths=args.files,
            update_progress=not args.no_db_update
        )

        # Print summary
        print("\n" + "="*60)
        print("DATA ROOM INGESTION SUMMARY")
        print("="*60)
        print(f"Data Room ID:    {results['data_room_id']}")
        print(f"Total Files:     {results['total_files']}")
        print(f"Parsed Files:    {results['parsed_files']}")
        print(f"Failed Files:    {results['failed_files']}")
        print(f"Total Chunks:    {results['total_chunks']}")
        print(f"Duration:        {results['duration_seconds']:.2f}s")
        print(f"Collection:      {results['collection_name']}")

        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  - {error['file']}: {error['error']}")

        print("="*60)

        sys.exit(0 if results['failed_files'] == 0 else 1)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
