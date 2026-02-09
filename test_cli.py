#!/usr/bin/env python3
"""
Simple CLI to test the PDF parsing and Q&A pipeline.

Usage:
    python test_cli.py <pdf_file> "<question>"

Example:
    python test_cli.py sample_pitch_deck.pdf "What is the revenue model?"
"""

import sys
import uuid
from pathlib import Path
from loguru import logger

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.parse_pdf import parse_pdf
from tools.chunk_documents import chunk_documents
from tools.generate_embeddings import generate_embeddings
from tools.index_to_vectordb import index_to_vectordb
from tools.answer_question import answer_question


def main():
    """Main CLI function."""

    if len(sys.argv) < 3:
        print("Usage: python test_cli.py <pdf_file> \"<question>\"")
        print("\nExample:")
        print("  python test_cli.py sample.pdf \"What is the revenue model?\"")
        sys.exit(1)

    pdf_file = sys.argv[1]
    question = sys.argv[2]

    # Validate file exists
    if not Path(pdf_file).exists():
        print(f"❌ File not found: {pdf_file}")
        sys.exit(1)

    print("\n" + "="*60)
    print("VC Due Diligence - PDF Q&A Test")
    print("="*60 + "\n")

    # Generate test data room ID
    data_room_id = f"test_{uuid.uuid4().hex[:8]}"

    try:
        # Step 1: Parse PDF
        print(f"📄 Step 1: Parsing PDF...")
        parsed = parse_pdf(pdf_file)

        if parsed.get('error'):
            print(f"❌ Parse error: {parsed['error']}")
            sys.exit(1)

        print(f"✅ Parsed {parsed['page_count']} pages, {parsed['total_chars']:,} characters")

        # Step 2: Chunk document
        print(f"\n📝 Step 2: Chunking document...")
        chunks = chunk_documents(parsed, chunk_size=800, overlap=100)
        print(f"✅ Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        print(f"\n🧠 Step 3: Generating embeddings...")
        print(f"   (This will use OpenAI API)")
        chunks_with_embeddings = generate_embeddings(chunks)
        print(f"✅ Generated {len(chunks_with_embeddings)} embeddings")

        # Step 4: Index to vector DB
        print(f"\n💾 Step 4: Indexing to vector database...")
        index_result = index_to_vectordb(data_room_id, chunks_with_embeddings)
        print(f"✅ Indexed {index_result['indexed']} chunks")

        # Step 5: Answer question
        print(f"\n❓ Step 5: Answering question...")
        print(f"   Question: {question}")
        print(f"   (This will use Claude API)")
        print()

        answer_result = answer_question(question, data_room_id)

        # Display answer
        print("="*60)
        print("ANSWER:")
        print("="*60)
        print()
        print(answer_result['answer'])
        print()
        print("="*60)
        print(f"SOURCES ({len(answer_result['sources'])}):")
        print("="*60)

        for i, source in enumerate(answer_result['sources'], 1):
            page_info = f", p.{source['page_number']}" if source['page_number'] else ""
            print(f"{i}. {source['file_name']}{page_info}")
            print(f"   Relevance: {source['relevance_score']:.3f}")
            print(f"   Excerpt: {source['excerpt'][:150]}...")
            print()

        # Display metadata
        print("="*60)
        print("METADATA:")
        print("="*60)
        print(f"Confidence: {answer_result['confidence_score']:.3f}")
        print(f"Response Time: {answer_result['response_time_ms']}ms")
        print(f"Tokens Used: {answer_result['tokens_used']:,}")
        print(f"Cost: ${answer_result['cost']:.4f}")
        print(f"Model: {answer_result['model']}")
        print("="*60 + "\n")

        # Cleanup prompt
        print("Note: Test data has been indexed to collection:", index_result['collection_id'])
        print("To clean up test data, you can manually delete the collection later.")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
