"""
PDF parsing tool with text extraction, table detection, and OCR fallback.
Supports multiple parsing strategies for different PDF types.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from loguru import logger

# Primary: PyMuPDF (fitz) — 5-10x faster than pdfplumber
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.debug("PyMuPDF not available. Install with: pip install PyMuPDF")

# Fallback: pdfplumber + PyPDF2
try:
    import PyPDF2
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    if not FITZ_AVAILABLE:
        logger.error("No PDF library available. Install PyMuPDF or pdfplumber+PyPDF2")
        sys.exit(1)

# OCR support (optional)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.debug("OCR packages not available. Install pytesseract and pdf2image for OCR support.")


class PDFParser:
    """PDF parsing with multiple extraction strategies."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def parse(self, use_ocr: bool = False, max_ocr_pages: int = 50) -> Dict[str, Any]:
        """
        Main parsing method with fallback strategies.

        Args:
            use_ocr: Whether to use OCR for scanned documents
            max_ocr_pages: Maximum pages to OCR (skips OCR for larger PDFs)

        Returns:
            Dictionary with extracted content
        """
        logger.info(f"Parsing PDF: {self.file_path.name}")

        result = {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path.absolute()),
            "file_size": self.file_path.stat().st_size,
            "pages": [],
            "text": "",
            "tables": [],
            "metadata": {},
            "method": "pypdf2",
            "page_count": 0,
            "has_images": False
        }

        # Pre-validation: Check file is readable and valid PDF
        try:
            if self.file_path.stat().st_size == 0:
                result["error"] = "PDF file is empty (0 bytes)"
                logger.error(f"PDF file is empty: {self.file_path}")
                return result

            # Check PDF magic bytes
            with open(self.file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    result["error"] = f"File is not a valid PDF (header: {header!r})"
                    logger.error(f"Invalid PDF header for {self.file_path}: {header!r}")
                    return result
        except Exception as e:
            result["error"] = f"Cannot read PDF file: {str(e)}"
            logger.error(f"Cannot read PDF file {self.file_path}: {e}")
            return result

        try:
            if FITZ_AVAILABLE:
                self._parse_with_fitz(result, use_ocr, max_ocr_pages)
            elif PDFPLUMBER_AVAILABLE:
                self._parse_with_pdfplumber(result, use_ocr, max_ocr_pages)
            else:
                result["error"] = "No PDF library available"
                return result

            # OCR fallback - only trigger when essentially zero text extracted
            if len(result["text"].strip()) == 0:
                if use_ocr:
                    logger.info("Low text content detected, attempting OCR...")
                    ocr_text = self._extract_text_with_ocr(max_pages=max_ocr_pages)
                    if ocr_text and len(ocr_text.strip()) > len(result["text"].strip()):
                        result["text"] = ocr_text
                        result["method"] = "ocr"
                        # Update page text as well
                        result["pages"] = [{"page_number": i+1, "text": part, "char_count": len(part)}
                                          for i, part in enumerate(ocr_text.split("\n\n"))]
                        logger.success(f"OCR extracted {len(ocr_text)} characters")
                    else:
                        result["needs_ocr"] = True
                        logger.warning("OCR did not improve text extraction")
                else:
                    result["needs_ocr"] = True
                    logger.info("Low text content - document may need OCR (image-based PDF)")

            # Calculate statistics
            result["total_chars"] = len(result["text"])
            result["total_tables"] = len(result["tables"])

            logger.success(f"Parsed {result['page_count']} pages, {result['total_chars']} chars, {result['total_tables']} tables")

            return result

        except UnicodeDecodeError as e:
            # Handle encoding errors that occur during PDF structure parsing
            logger.error(f"Encoding error parsing PDF structure: {e}")
            result["error"] = f"PDF contains unsupported character encoding: {str(e)}"
            return result
        except Exception as e:
            logger.error(f"Unexpected error parsing PDF: {e}")
            result["error"] = str(e)
            return result

    def _parse_with_fitz(self, result: Dict[str, Any], use_ocr: bool, max_ocr_pages: int):
        """Parse PDF using PyMuPDF (fitz) — fast path with parallel page extraction."""
        doc = fitz.open(self.file_path)
        try:
            result["page_count"] = len(doc)

            if doc.is_encrypted:
                logger.warning(f"PDF is encrypted: {self.file_path.name}")
                result["error"] = "PDF is encrypted and requires a password"
                return

            # Extract metadata
            meta = doc.metadata or {}
            result["metadata"] = {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "creation_date": meta.get("creationDate", ""),
            }

            skip_tables = os.getenv("SKIP_TABLE_EXTRACTION", "").lower() in ("1", "true", "yes")
            extract_tables = not skip_tables and len(doc) <= 50
            num_pages = len(doc)

            def _extract_page(page_num_0based):
                """Extract text (and optionally tables) from a single page."""
                page = doc[page_num_0based]
                page_num = page_num_0based + 1
                try:
                    page_text = page.get_text() or ""
                    page_data = {
                        "page_number": page_num,
                        "text": page_text,
                        "char_count": len(page_text)
                    }
                    page_tables = []
                    if extract_tables:
                        try:
                            tables = page.find_tables()
                            for table_idx, table in enumerate(tables):
                                extracted = table.extract()
                                if extracted:
                                    page_tables.append({
                                        "page_number": page_num,
                                        "table_index": table_idx,
                                        "rows": len(extracted),
                                        "columns": len(extracted[0]) if extracted else 0,
                                        "data": extracted,
                                        "headers": extracted[0] if extracted else []
                                    })
                        except Exception:
                            pass
                    return page_data, page_tables
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    return {"page_number": page_num, "text": "", "error": str(e)}, []

            # Process pages in parallel (PyMuPDF page access is thread-safe for text extraction)
            # Tier 3 optimization: increased from 8 to 16 workers for faster PDF processing
            workers = min(16, num_pages)
            page_results = [None] * num_pages

            if num_pages > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_extract_page, i): i for i in range(num_pages)}
                    for future in futures:
                        idx = futures[future]
                        page_results[idx] = future.result()
            else:
                page_results[0] = _extract_page(0)

            # Collect results in order
            all_text = []
            for page_data, page_tables in page_results:
                all_text.append(page_data.get("text", ""))
                result["pages"].append(page_data)
                result["tables"].extend(page_tables)

            result["text"] = "\n\n".join(all_text)
            result["method"] = "pymupdf"

            if not extract_tables and not skip_tables and len(doc) > 50:
                logger.info(f"Skipped table extraction for large PDF ({len(doc)} pages)")
            elif skip_tables:
                logger.info("Table extraction skipped (SKIP_TABLE_EXTRACTION=true)")
        finally:
            doc.close()

    def _parse_with_pdfplumber(self, result: Dict[str, Any], use_ocr: bool, max_ocr_pages: int):
        """Parse PDF using pdfplumber/PyPDF2 — fallback path."""
        with open(self.file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            result["page_count"] = len(pdf_reader.pages)
            result["metadata"] = self._extract_metadata(pdf_reader)

            if pdf_reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {self.file_path.name}")
                result["error"] = "PDF is encrypted and requires a password"
                return

        all_text = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                        all_text.append(page_text)
                        result["pages"].append({
                            "page_number": page_num,
                            "text": page_text,
                            "char_count": len(page_text)
                        })
                        if result["page_count"] <= 50:
                            page_tables = page.extract_tables()
                            for table_idx, table in enumerate(page_tables):
                                if table:
                                    result["tables"].append({
                                        "page_number": page_num,
                                        "table_index": table_idx,
                                        "rows": len(table),
                                        "columns": len(table[0]) if table else 0,
                                        "data": table,
                                        "headers": table[0] if table else []
                                    })
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
                        result["pages"].append({
                            "page_number": page_num,
                            "text": "",
                            "error": str(e)
                        })
        except Exception as e:
            logger.warning(f"pdfplumber failed, falling back to PyPDF2: {e}")
            all_text = []
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                        all_text.append(page_text)
                        result["pages"].append({
                            "page_number": page_num,
                            "text": page_text,
                            "char_count": len(page_text)
                        })
                    except Exception as ex:
                        logger.warning(f"Failed to extract page {page_num}: {ex}")
                        result["pages"].append({
                            "page_number": page_num,
                            "text": "",
                            "error": str(ex)
                        })

        result["text"] = "\n\n".join(all_text)
        if result["tables"]:
            result["method"] = "pdfplumber"
        if result["page_count"] > 50:
            logger.info(f"Skipped table extraction for large PDF ({result['page_count']} pages)")

    def _extract_metadata(self, pdf_reader) -> Dict[str, Any]:
        """Extract PDF metadata with encoding error handling."""
        try:
            metadata = pdf_reader.metadata
            if metadata:
                result = {}
                for key, field in [
                    ("title", "/Title"),
                    ("author", "/Author"),
                    ("subject", "/Subject"),
                    ("creator", "/Creator"),
                    ("producer", "/Producer"),
                    ("creation_date", "/CreationDate"),
                ]:
                    try:
                        value = metadata.get(field, "")
                        # Handle bytes or strings with potential encoding issues
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='replace')
                        elif value:
                            value = str(value)
                        result[key] = value
                    except (UnicodeDecodeError, UnicodeEncodeError) as e:
                        logger.debug(f"Encoding error for metadata field {field}: {e}")
                        result[key] = ""
                return result
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            logger.warning(f"Encoding error extracting metadata: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        return {}

    def _extract_tables_with_pdfplumber(self) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        tables = []

        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()

                    for table_idx, table in enumerate(page_tables):
                        if table:  # Skip empty tables
                            tables.append({
                                "page_number": page_num,
                                "table_index": table_idx,
                                "rows": len(table),
                                "columns": len(table[0]) if table else 0,
                                "data": table,
                                "headers": table[0] if table else []
                            })
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        return tables

    def _extract_text_with_ocr(self, max_pages: int = 50) -> str:
        """Extract text from PDF using OCR (for image-based PDFs).

        Args:
            max_pages: Maximum number of pages to OCR (default 50).
                       Skips OCR entirely for PDFs exceeding this limit.
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available. Install pytesseract and pdf2image, and Tesseract OCR.")
            return ""

        try:
            # Check page count before converting
            import PyPDF2
            with open(self.file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

            if total_pages > max_pages:
                logger.warning(f"Skipping OCR: {self.file_path.name} has {total_pages} pages (limit: {max_pages})")
                return ""

            logger.info(f"Running OCR on {self.file_path.name} ({total_pages} pages)...")
            text_parts = [None] * total_pages

            def _ocr_page_batch(start, end):
                """Convert and OCR a batch of pages."""
                images = convert_from_path(self.file_path, first_page=start, last_page=end)
                results = []
                for i, img in enumerate(images):
                    page_text = pytesseract.image_to_string(img)
                    results.append((start + i - 1, page_text))
                    del img
                del images
                return results

            # Process OCR in parallel batches
            batch_size = 10
            batches = [(s, min(s + batch_size - 1, total_pages))
                       for s in range(1, total_pages + 1, batch_size)]
            ocr_workers = min(4, len(batches))

            with ThreadPoolExecutor(max_workers=ocr_workers) as executor:
                futures = [executor.submit(_ocr_page_batch, s, e) for s, e in batches]
                for future in futures:
                    for idx, page_text in future.result():
                        text_parts[idx] = page_text

            result = "\n\n".join(text_parts)
            logger.success(f"OCR extracted {len(result)} characters from {total_pages} pages")
            return result

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""


def parse_pdf(file_path: str, use_ocr: bool = False, max_ocr_pages: int = 50) -> Dict[str, Any]:
    """
    Convenience function to parse a PDF file.

    Args:
        file_path: Path to PDF file
        use_ocr: Whether to use OCR for scanned documents
        max_ocr_pages: Maximum pages to OCR (skips OCR for larger PDFs)

    Returns:
        Dictionary with parsed content

    Example:
        >>> result = parse_pdf("pitch_deck.pdf")
        >>> print(f"Extracted {len(result['text'])} characters from {result['page_count']} pages")
    """
    parser = PDFParser(file_path)
    return parser.parse(use_ocr=use_ocr, max_ocr_pages=max_ocr_pages)


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse PDF documents")
    parser.add_argument("file", help="Path to PDF file")
    parser.add_argument("--ocr", action="store_true", help="Use OCR for scanned documents")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Parse PDF
    result = parse_pdf(args.file, use_ocr=args.ocr)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Saved results to {output_path}")
    else:
        # Print summary
        print(f"\n{'='*60}")
        print(f"File: {result['file_name']}")
        print(f"Pages: {result['page_count']}")
        print(f"Characters: {result['total_chars']:,}")
        print(f"Tables: {result['total_tables']}")
        print(f"Method: {result['method']}")

        if result.get('error'):
            print(f"Error: {result['error']}")

        if result.get('needs_ocr'):
            print("\nWarning: Low text content. Document may be scanned. Use --ocr flag.")

        print(f"{'='*60}\n")

        # Print first 500 chars
        if result['text']:
            print("First 500 characters:")
            print(result['text'][:500])
            print("...")


if __name__ == "__main__":
    main()
