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
from tools.parse_docx import parse_docx
from tools.parse_pptx import parse_pptx
from tools.chunk_documents import chunk_documents, DocumentChunker
from tools.generate_embeddings import generate_embeddings, generate_embeddings_streaming
from tools.index_to_vectordb import index_to_vectordb, VectorDBIndexer
from app import database as db
from app.config import settings


def parse_file_by_type(
    file_path: str,
    use_ocr: bool = True,
    max_ocr_pages: int = 50,
    use_financial_excel: bool = False,
) -> Dict[str, Any]:
    """
    Parse a file using the appropriate parser based on extension.

    Single source of truth for file type dispatch. All processing pipelines
    (main.py, worker.py, ingest_data_room.py) should use this.

    Args:
        file_path: Path to the file
        use_ocr: Enable OCR for scanned PDFs
        max_ocr_pages: Max pages to run OCR on
        use_financial_excel: Use the enhanced financial Excel parser

    Returns:
        Parsed document dictionary with at minimum:
        file_name, file_path, text, file_type, method

    Raises:
        ValueError: For unsupported file types
        FileNotFoundError: If file does not exist
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = file_path_obj.suffix.lower()

    if file_type == '.pdf':
        return parse_pdf(file_path, use_ocr=use_ocr, max_ocr_pages=max_ocr_pages)

    elif file_type in ('.xlsx', '.xls', '.csv'):
        if use_financial_excel and file_type != '.csv':
            from tools.parse_excel_financial import parse_excel_financial
            return parse_excel_financial(file_path)
        return parse_excel(file_path)

    elif file_type in ('.docx', '.doc'):
        return parse_docx(file_path)

    elif file_type in ('.pptx', '.ppt'):
        return parse_pptx(file_path)

    elif file_type == '.txt':
        return {
            'file_name': file_path_obj.name,
            'file_path': str(file_path_obj.absolute()),
            'file_size': file_path_obj.stat().st_size,
            'text': file_path_obj.read_text(encoding='utf-8', errors='replace'),
            'file_type': 'txt',
            'method': 'plaintext',
        }

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _process_single_file(
    file_path: str,
    data_room_id: str,
    update_progress: bool,
    chunker=None,
) -> Dict[str, Any]:
    """Parse and chunk a single file. Returns dict with 'chunks', 'error', 'file_name'."""
    file_path_obj = Path(file_path)
    result = {'file_name': file_path_obj.name, 'chunks': [], 'error': None}

    if not file_path_obj.exists():
        result['error'] = f"File not found: {file_path}"
        return result

    # Find document ID with resilient matching and auto-create fallback
    document_id = db.find_document_by_filename(
        data_room_id=data_room_id,
        target_filename=file_path_obj.name,
        create_if_missing=True,
        file_path=str(file_path_obj),
        file_size=file_path_obj.stat().st_size,
    )

    if not document_id:
        result['error'] = f"Document record not found and auto-creation failed for {file_path_obj.name}"
        return result

    if update_progress:
        db.update_document_status(document_id, "parsing")

    # Parse using shared dispatcher
    try:
        parsed = parse_file_by_type(file_path)
    except ValueError as e:
        error_msg = str(e)
        if update_progress:
            db.update_document_status(document_id, "failed", error_message=error_msg)
        result['error'] = error_msg
        return result

    # Chunk the document
    logger.info(f"Chunking {file_path_obj.name}...")
    chunks = chunk_documents(parsed, chunker=chunker)

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

    Pipeline (streaming — processes files in groups to limit memory):
    1. Parse a group of documents (PDF, Excel, etc.)
    2. Chunk text into semantic units
    3. Generate embeddings for the group's chunks
    4. Index to vector database + save to SQLite
    5. Repeat for next group until all files are processed

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

    collection_name = f"data_room_{data_room_id}"
    total_indexed = 0
    total_saved = 0
    total_embedding_failures = 0
    completed_files = 0

    # Reuse indexer, chunker, and write executor across all groups
    indexer = VectorDBIndexer()
    shared_chunker = DocumentChunker()
    write_executor = ThreadPoolExecutor(max_workers=5)

    # Process files in groups to limit peak memory usage
    FILE_GROUP_SIZE = 20
    max_workers = min(settings.max_parse_workers, len(file_paths))

    try:
        for group_start in range(0, len(file_paths), FILE_GROUP_SIZE):
            group_paths = file_paths[group_start:group_start + FILE_GROUP_SIZE]
            group_num = group_start // FILE_GROUP_SIZE + 1
            total_groups = (len(file_paths) + FILE_GROUP_SIZE - 1) // FILE_GROUP_SIZE
            logger.info(f"Processing file group {group_num}/{total_groups} ({len(group_paths)} files)...")

            # --- Parse this group in parallel ---
            group_chunks = []
            group_workers = min(max_workers, len(group_paths))

            with ThreadPoolExecutor(max_workers=group_workers) as parse_executor:
                future_to_path = {
                    parse_executor.submit(_process_single_file, fp, data_room_id, update_progress, shared_chunker): fp
                    for fp in group_paths
                }

                for future in as_completed(future_to_path):
                    fp = future_to_path[future]
                    completed_files += 1
                    try:
                        file_result = future.result()

                        if update_progress:
                            progress = 5 + (completed_files / len(file_paths)) * 40
                            db.update_data_room_status(data_room_id, "parsing", progress=progress)

                        if file_result['error']:
                            logger.error(f"Failed: {file_result['file_name']}: {file_result['error']}")
                            results['errors'].append({
                                'file': file_result['file_name'],
                                'error': file_result['error']
                            })
                            results['failed_files'] += 1
                        else:
                            group_chunks.extend(file_result['chunks'])
                            results['parsed_files'] += 1

                    except Exception as e:
                        error_msg = f"Failed to parse {fp}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append({
                            'file': Path(fp).name,
                            'error': str(e)
                        })
                        results['failed_files'] += 1

            if not group_chunks:
                logger.warning(f"Group {group_num}: no chunks extracted, skipping embedding")
                continue

            results['total_chunks'] += len(group_chunks)
            logger.info(f"Group {group_num}: {len(group_chunks)} chunks, embedding + indexing...")

            if update_progress:
                progress = 45 + (completed_files / len(file_paths)) * 40
                db.update_data_room_status(data_room_id, "indexing", progress=progress)

            # --- Embed + index + save this group's chunks ---
            pending_write_futures = []
            for batch in generate_embeddings_streaming(
                chunks=group_chunks,
                batch_size=settings.batch_size,
                max_concurrent=settings.embedding_max_concurrent
            ):
                valid_chunks = [c for c in batch if c.get('embedding') is not None and not c.get('embedding_failed')]
                failed_chunks = [c for c in batch if c.get('embedding') is None or c.get('embedding_failed')]
                total_embedding_failures += len(failed_chunks)

                if failed_chunks:
                    logger.warning(f"{len(failed_chunks)} chunks failed embedding generation in this batch")

                # Pipeline VectorDB and SQLite writes — don't block between batches
                batch_futures = []
                if valid_chunks:
                    batch_futures.append(write_executor.submit(
                        indexer.index_chunks, data_room_id, valid_chunks
                    ))
                batch_futures.append(write_executor.submit(db.create_chunks, batch))
                pending_write_futures.append(batch_futures)

                total_indexed += len(valid_chunks)
                total_saved += len(batch)

                if update_progress:
                    progress = 45 + (completed_files / len(file_paths)) * 40 + (total_saved / max(results['total_chunks'], 1)) * 10
                    write_executor.submit(
                        db.update_data_room_status, data_room_id, "indexing", min(progress, 95)
                    )

            # Wait for all pipelined writes to complete before freeing group memory
            for batch_futures in pending_write_futures:
                for future in batch_futures:
                    future.result()

            # Free group memory before next iteration
            del group_chunks

            logger.info(f"Group {group_num} done. Running totals: indexed={total_indexed}, saved={total_saved}")

    finally:
        write_executor.shutdown(wait=True)

    if results['total_chunks'] == 0:
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

    if total_embedding_failures > 0:
        error_msg = f"{total_embedding_failures}/{results['total_chunks']} chunks failed embedding generation"
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
            db.update_data_room_status(
                data_room_id,
                "complete",
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

            # Only mark chunks as embedded if indexing actually succeeded
            if result.get('indexed', 0) > 0:
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
