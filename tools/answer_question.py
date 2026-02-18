"""
RAG-powered Q&A tool using semantic search and Claude API.
Answers analyst questions with citations from data room documents.
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from loguru import logger

try:
    from anthropic import Anthropic, APITimeoutError
    from dotenv import load_dotenv
except ImportError:
    logger.error("Required packages not installed. Run: pip install anthropic python-dotenv")
    sys.exit(1)

# Handle both import paths (from tools dir and from project root)
try:
    from semantic_search import semantic_search
except ImportError:
    from tools.semantic_search import semantic_search

# Load environment variables
load_dotenv()


class QuestionAnswerer:
    """
    RAG-powered question answering system.
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        max_context_chunks: int = 15
    ):
        """
        Initialize question answerer.

        Args:
            anthropic_api_key: Anthropic API key
            model: Claude model to use
            max_context_chunks: Maximum chunks to include in context
        """
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.haiku_model = "claude-3-5-haiku-20241022"
        self.max_tokens = 2048
        self.api_timeout = 90  # seconds

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        # Model pricing (per 1M tokens)
        self.pricing = {
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
            "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00}
        }

        logger.info(f"QuestionAnswerer initialized with model: {model}")

    def answer(
        self,
        question: str,
        data_room_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline with optional web search.

        Args:
            question: Analyst's question
            data_room_id: Data room to search
            conversation_history: Previous conversation turns
            filters: Optional filters for search

        Returns:
            Answer with sources and metadata
        """
        start_time = time.time()

        logger.info(f"Answering question for data room {data_room_id}")
        logger.debug(f"Question: {question}")

        # Determine if web search is enabled
        web_search_enabled = self._should_use_web_search()

        # Step 1: Multi-query semantic search for relevant chunks
        search_results = self._multi_query_search(
            question=question,
            data_room_id=data_room_id,
            top_k=self.max_context_chunks * 2,  # Get extra for reranking
            filters=filters
        )

        if not search_results and not web_search_enabled:
            logger.warning("No relevant context found")
            # Check if the collection exists and has documents
            no_results_reason = "I couldn't find relevant information in the data room documents to answer this question."
            try:
                from tools.semantic_search import SemanticSearch
                searcher = SemanticSearch()
                collection_name = f"data_room_{data_room_id}"
                try:
                    collection = searcher.chroma_client.get_collection(name=collection_name)
                    if collection.count() == 0:
                        no_results_reason = (
                            "This data room has no indexed documents. The uploaded files may have failed "
                            "to process during embedding. Please try re-processing the data room."
                        )
                except Exception:
                    no_results_reason = (
                        "This data room has not been indexed yet. Please ensure files have been "
                        "uploaded and processing has completed before asking questions."
                    )
            except Exception:
                pass

            return {
                "question": question,
                "answer": no_results_reason,
                "sources": [],
                "confidence_score": 0.0,
                "tokens_used": 0,
                "cost": 0.0,
                "response_time_ms": int((time.time() - start_time) * 1000)
            }

        # Step 2: Select top chunks for context
        context_chunks = search_results[:self.max_context_chunks] if search_results else []

        # Step 3: Build web search tool config
        tools = None
        if web_search_enabled:
            tools = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self._get_web_search_max_uses(),
            }]

        # Step 4: Build prompt with context
        prompt = self._build_prompt(question, context_chunks, conversation_history,
                                     web_search_enabled=web_search_enabled)

        # Step 5: Call Claude API
        try:
            max_tokens = 4096 if tools else self.max_tokens
            timeout = 150 if tools else self.api_timeout

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "timeout": timeout,
                "messages": [{"role": "user", "content": prompt}],
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

            total_input = response.usage.input_tokens
            total_output = response.usage.output_tokens
            all_content = list(response.content)

            # Handle pause_turn: Claude paused mid-turn (e.g. during web search)
            if response.stop_reason == "pause_turn":
                continuation_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content},
                ]
                kwargs["messages"] = continuation_messages
                response2 = self.client.messages.create(**kwargs)
                all_content.extend(response2.content)
                total_input += response2.usage.input_tokens
                total_output += response2.usage.output_tokens

            # Extract text (skip narration before tool calls when tools are used)
            if tools:
                answer_text = self._extract_text_after_tools(all_content)
            else:
                answer_text = response.content[0].text

            # Extract web sources and search count
            web_sources = []
            web_search_count = 0
            if tools:
                # Build a minimal object with .content for _extract_web_sources
                class _FakeMsg:
                    pass
                fake_msg = _FakeMsg()
                fake_msg.content = all_content
                web_sources = self._extract_web_sources(fake_msg)

                server_tool_use = getattr(response.usage, 'server_tool_use', None)
                if server_tool_use:
                    web_search_count = getattr(server_tool_use, 'web_search_requests', 0)

            # Track usage
            self.total_input_tokens += total_input
            self.total_output_tokens += total_output

            # Calculate cost (tokens + web search)
            pricing = self.pricing.get(self.model, {"input": 3.0, "output": 15.0})
            token_cost = (total_input * pricing["input"] / 1_000_000) + \
                         (total_output * pricing["output"] / 1_000_000)
            web_search_cost = web_search_count * 0.01
            cost = token_cost + web_search_cost
            self.total_cost += cost

            # Step 6: Extract and validate citations
            sources = self._extract_sources(context_chunks) if context_chunks else []

            # Calculate confidence based on similarity scores
            avg_similarity = (sum(c['similarity_score'] for c in context_chunks) / len(context_chunks)) if context_chunks else 0.0
            confidence_score = round(avg_similarity, 3)

            response_time_ms = int((time.time() - start_time) * 1000)

            result = {
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "web_sources": web_sources,
                "web_search_count": web_search_count,
                "confidence_score": confidence_score,
                "tokens_used": total_input + total_output,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost": round(cost, 6),
                "response_time_ms": response_time_ms,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }

            logger.success(
                f"Generated answer in {response_time_ms}ms, cost: ${cost:.4f}"
                + (f", {web_search_count} web searches" if web_search_count else "")
            )

            return result

        except APITimeoutError as e:
            logger.error(f"Claude API timed out after {self.api_timeout}s: {e}")
            return {
                "question": question,
                "answer": "The AI model took too long to respond. This can happen with complex questions or large context. Please try again or simplify your question.",
                "sources": [],
                "confidence_score": 0.0,
                "error": "timeout",
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence_score": 0.0,
                "error": str(e),
                "response_time_ms": int((time.time() - start_time) * 1000)
            }

    def _expand_query(self, question: str) -> List[str]:
        """
        Use Claude Haiku to generate 2-3 keyword-rich search queries
        from the user's question to improve retrieval recall.

        Returns empty list on failure, preserving single-query behavior.
        """
        prompt = f"""You are a search query expander for a venture capital due diligence system.
Given an analyst's question about a company, generate 2-3 alternative search queries that would better match the actual content in data room documents (pitch decks, financial models, cap tables, legal docs, etc.).

Rules:
- Each query should be keyword-rich, using terms that would actually appear in documents
- Focus on specific nouns, titles, metrics, and domain terminology
- Do NOT rephrase the question — generate queries that would MATCH document content
- Return ONLY the queries, one per line, no numbering, no bullets, no explanation

Example:
Question: "Give me highlights of the team"
Queries:
founders team CEO CTO background experience leadership
advisors board members management team hiring key personnel
co-founder education prior companies executive biography

Question: "What's the company's competitive advantage?"
Queries:
competitive advantage moat differentiation unique value proposition
competitors market positioning intellectual property patents technology

Question: "{question}"
Queries:"""

        try:
            response = self.client.messages.create(
                model=self.haiku_model,
                max_tokens=200,
                temperature=0.3,
                timeout=10,
                messages=[{"role": "user", "content": prompt}]
            )

            raw_text = response.content[0].text.strip()

            queries = [
                line.strip()
                for line in raw_text.split("\n")
                if line.strip() and not line.strip().startswith(("Question:", "Queries:"))
            ][:3]

            if queries:
                logger.debug(f"Query expansion generated {len(queries)} additional queries: {queries}")

            return queries

        except Exception as e:
            logger.warning(f"Query expansion failed (falling back to single query): {e}")
            return []

    def _keyword_search(
        self,
        question: str,
        expanded_queries: List[str],
        data_room_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search chunks by keyword matching in PostgreSQL.
        Returns results formatted to match semantic search output.
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from app.database import search_chunks_by_keywords
        except ImportError:
            logger.warning("Could not import database module for keyword search")
            return []

        # Extract meaningful keywords (4+ chars) from question and expanded queries
        all_text = question + " " + " ".join(expanded_queries)
        stop_words = {"what", "give", "tell", "about", "highlights", "with", "from",
                       "that", "this", "have", "does", "their", "they", "some", "more",
                       "been", "were", "will", "would", "could", "should", "which"}
        keywords = [
            w for w in all_text.lower().split()
            if len(w) >= 3 and w not in stop_words
        ]
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        if not unique_keywords:
            return []

        try:
            db_chunks = search_chunks_by_keywords(
                data_room_id=data_room_id,
                keywords=unique_keywords[:10],  # Limit keywords to avoid huge queries
                limit=top_k
            )
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

        # Format to match semantic search output
        formatted = []
        for i, chunk in enumerate(db_chunks):
            metadata_json = chunk.get('metadata', '{}')
            try:
                metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else (metadata_json or {})
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            formatted.append({
                'rank': i + 1,
                'chunk_id': chunk.get('id', f'kw_{i}'),
                'chunk_text': chunk.get('chunk_text', ''),
                'similarity_score': 0.5,  # Synthetic score for keyword matches
                'distance': 1.0,
                'metadata': metadata,
                'source': {
                    'file_name': chunk.get('file_name', metadata.get('file_name', 'Unknown')),
                    'page_number': chunk.get('page_number'),
                    'section_title': chunk.get('section_title', ''),
                    'document_category': chunk.get('document_category', metadata.get('document_category', '')),
                    'sheet_name': metadata.get('sheet_name'),
                    'row_start': metadata.get('row_start'),
                    'row_end': metadata.get('row_end'),
                    'currency': metadata.get('currency'),
                }
            })

        if formatted:
            logger.info(f"Keyword search found {len(formatted)} chunks for keywords: {unique_keywords[:5]}")

        return formatted

    def _multi_query_search(
        self,
        question: str,
        data_room_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: semantic search (with query expansion) + keyword search.
        All searches run in parallel using threads.
        Uses round-robin selection to guarantee each source contributes results.
        """
        search_start = time.time()
        per_query_results: List[List[Dict[str, Any]]] = []

        def _do_semantic(query: str) -> List[Dict[str, Any]]:
            try:
                return semantic_search(
                    query=query,
                    data_room_id=data_room_id,
                    top_k=top_k,
                    filters=filters,
                )
            except Exception as e:
                logger.warning(f"Search failed for query '{query[:50]}...': {e}")
                return []

        with ThreadPoolExecutor(max_workers=6) as pool:
            # Immediately start: original query search + keyword search + query expansion
            original_future = pool.submit(_do_semantic, question)
            keyword_future = pool.submit(
                self._keyword_search, question, [], data_room_id, top_k
            )
            expand_future = pool.submit(self._expand_query, question)

            # Wait for expansion to finish, then submit expanded query searches
            expanded_futures: List[Future] = []
            try:
                expanded_queries = expand_future.result(timeout=10)
            except Exception:
                expanded_queries = []

            for eq in expanded_queries:
                expanded_futures.append(pool.submit(_do_semantic, eq))

            # Re-submit keyword search with expanded queries if we got any
            if expanded_queries:
                keyword_future_2 = pool.submit(
                    self._keyword_search, question, expanded_queries, data_room_id, top_k
                )
            else:
                keyword_future_2 = None

            # Collect original query results
            per_query_results.append(original_future.result(timeout=15))

            # Collect expanded query results
            for ef in expanded_futures:
                try:
                    per_query_results.append(ef.result(timeout=15))
                except Exception as e:
                    logger.warning(f"Expanded search timed out: {e}")

            # Collect keyword results (prefer expanded version if available)
            kw_results: List[Dict[str, Any]] = []
            if keyword_future_2 is not None:
                try:
                    kw_results = keyword_future_2.result(timeout=15)
                except Exception:
                    pass
            if not kw_results:
                try:
                    kw_results = keyword_future.result(timeout=15)
                except Exception:
                    pass
            if kw_results:
                per_query_results.append(kw_results)

        # Round-robin: take top results from each source in turns to ensure diversity
        selected: List[Dict[str, Any]] = []
        seen_ids: set = set()
        max_rounds = top_k  # safety limit

        for round_idx in range(max_rounds):
            added_this_round = False
            for query_results in per_query_results:
                # Find the next unseen chunk from this query
                while query_results:
                    chunk = query_results.pop(0)
                    cid = chunk.get("chunk_id", chunk.get("rank", id(chunk)))
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        selected.append(chunk)
                        added_this_round = True
                        break
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k or not added_this_round:
                break

        search_ms = int((time.time() - search_start) * 1000)
        total_queries = 1 + len(expanded_queries)
        logger.info(
            f"Hybrid search: {total_queries} semantic queries + keyword search, "
            f"{len(selected)} unique chunks selected (round-robin) in {search_ms}ms"
        )

        return selected

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        web_search_enabled: bool = False
    ) -> str:
        """Build prompt for Claude with context and question."""

        # System instructions
        if web_search_enabled:
            system_prompt = """You are an AI assistant helping venture capital analysts with due diligence.

Your role:
- Answer questions using both the provided data room documents AND web research when appropriate
- For questions about the company's own documents (financials, pitch deck content, cap tables), rely primarily on the data room context
- For questions requiring external context (market comparables, industry benchmarks, competitor analysis, recent news, public market data), use web search to find current, relevant information
- Be concise but thorough
- Use professional VC analysis language
- If you need to make calculations or inferences, explain your reasoning
- Never make up information or hallucinate facts
- Do NOT include source citations or references to the uploaded data room documents unless the user explicitly asks for them.
- When you use information from web research, clearly label it as coming from external sources and include the source name.
- When citing monetary values, ALWAYS use the currency specified in the document context (e.g., NGN, USD, EUR). Different sheets or sections may use different currencies — respect each one. Never default to "$" unless the source explicitly uses USD.
- Financial models may have historical sheets in local currency (e.g., NGN) and projection sheets in a different currency (e.g., USD). Always prefix monetary values with the correct currency code for their specific sheet as indicated by the "Currency:" label in the context.
- Do not write any preamble or narration about your search process. Do not announce what you will search for. Write only the final analysis.

Quality standards:
- If context is ambiguous or contradictory, note this explicitly
- Prioritize accuracy over completeness
- When web research provides data that contradicts or supplements data room content, note both perspectives"""
        else:
            system_prompt = """You are an AI assistant helping venture capital analysts with due diligence.

Your role:
- Answer questions accurately based ONLY on the provided context from data room documents
- If information is not in the context, clearly state "This information is not available in the provided documents"
- Be concise but thorough
- Use professional VC analysis language
- If you need to make calculations or inferences, explain your reasoning
- Never make up information or hallucinate facts
- Do NOT include source citations or references to the uploaded data room documents unless the user explicitly asks for them.
- If you use any information from outside the uploaded documents (general industry knowledge, market benchmarks, etc.), clearly note it as an external source.
- When citing monetary values, ALWAYS use the currency specified in the document context (e.g., NGN, USD, EUR). Different sheets or sections may use different currencies — respect each one. Never default to "$" unless the source explicitly uses USD.
- Financial models may have historical sheets in local currency (e.g., NGN) and projection sheets in a different currency (e.g., USD). Always prefix monetary values with the correct currency code for their specific sheet as indicated by the "Currency:" label in the context.

Quality standards:
- If context is ambiguous or contradictory, note this explicitly
- Prioritize accuracy over completeness"""

        # Build context from chunks
        context_sections = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['source']

            # Build location info (page for PDFs, sheet/rows for Excel)
            location_parts = []
            if source.get('page_number'):
                location_parts.append(f"p.{source['page_number']}")
            if source.get('sheet_name'):
                location_parts.append(f"Sheet: {source['sheet_name']}")
                if source.get('row_start') and source.get('row_end'):
                    location_parts.append(f"Rows {source['row_start']}-{source['row_end']}")

            location_info = f", {', '.join(location_parts)}" if location_parts else ""

            currency_info = ""
            if source.get('currency') and source['currency'] != 'Unknown':
                currency_info = f"\nCurrency: {source['currency']}"

            context_sections.append(
                f"[Document {i}: {source['file_name']}{location_info}]{currency_info}\n{chunk['chunk_text']}"
            )

        context = "\n\n---\n\n".join(context_sections)

        # Build conversation history if exists
        history_text = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                history_parts.append(f"Q: {turn['question']}\nA: {turn['answer']}")
            history_text = "\n\n".join(history_parts)
            history_text = f"\n\nPrevious conversation:\n{history_text}\n"

        # Combine into final prompt
        prompt = f"""{system_prompt}

Context from data room documents:

{context}
{history_text}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        return prompt

    def _extract_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context chunks."""
        sources = []
        seen_sources = set()

        for chunk in context_chunks:
            source = chunk['source']
            # Include sheet_name in deduplication key for Excel files
            source_key = (
                source['file_name'],
                source.get('page_number'),
                source.get('sheet_name')
            )

            # Avoid duplicate sources
            if source_key not in seen_sources:
                sources.append({
                    'file_name': source['file_name'],
                    'page_number': source.get('page_number'),
                    'relevance_score': chunk['similarity_score'],
                    'excerpt': chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'],
                    # Excel sheet metadata
                    'sheet_name': source.get('sheet_name'),
                    'row_start': source.get('row_start'),
                    'row_end': source.get('row_end'),
                })
                seen_sources.add(source_key)

        return sources

    def _should_use_web_search(self) -> bool:
        """Check if web search is globally enabled via config."""
        try:
            from app.config import settings
            return settings.qa_web_search_enabled
        except ImportError:
            return os.getenv("QA_WEB_SEARCH_ENABLED", "True").lower() == "true"

    def _get_web_search_max_uses(self) -> int:
        """Get the max web searches per question from config."""
        try:
            from app.config import settings
            return settings.qa_web_search_max_uses
        except ImportError:
            return int(os.getenv("QA_WEB_SEARCH_MAX_USES", "10"))

    def _extract_web_sources(self, final_message) -> List[Dict[str, Any]]:
        """Extract web search source citations from Claude's response content blocks."""
        web_sources = []
        seen_urls = set()

        for block in final_message.content:
            if hasattr(block, 'type') and block.type == 'web_search_tool_result':
                for result in getattr(block, 'search_results', []):
                    url = getattr(result, 'url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        snippet = getattr(result, 'snippet', '') or ''
                        web_sources.append({
                            'type': 'web',
                            'title': getattr(result, 'title', ''),
                            'url': url,
                            'snippet': snippet[:200],
                        })

        return web_sources

    def _extract_text_after_tools(self, content_blocks) -> str:
        """Extract text from content blocks, skipping narration before tool calls.

        Web search responses interleave text + server_tool_use + web_search_tool_result blocks.
        Only keep text blocks after the last tool block (the final analysis).
        """
        last_tool_idx = -1
        for i, block in enumerate(content_blocks):
            if not hasattr(block, 'text'):
                last_tool_idx = i
        text_parts = []
        for i, block in enumerate(content_blocks):
            if hasattr(block, 'text') and i > last_tool_idx:
                text_parts.append(block.text)
        return "".join(text_parts)

    def answer_stream(
        self,
        question: str,
        data_room_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        """
        Stream an answer as SSE events.

        Yields SSE-formatted lines:
          data: {"type": "search_done", "sources": [...]}
          data: {"type": "web_search_status", "status": "searching", "count": N}
          data: {"type": "delta", "text": "..."}
          data: {"type": "done", "metadata": {...}}
        """
        start_time = time.time()

        # Determine if web search is enabled
        web_search_enabled = self._should_use_web_search()

        # Step 1: Search (same as non-streaming)
        search_results = self._multi_query_search(
            question=question,
            data_room_id=data_room_id,
            top_k=self.max_context_chunks * 2,
            filters=filters,
        )

        if not search_results and not web_search_enabled:
            yield f"data: {json.dumps({'type': 'done', 'metadata': {'answer': 'I could not find relevant information in the data room documents to answer this question.', 'sources': [], 'confidence_score': 0.0, 'response_time_ms': int((time.time() - start_time) * 1000)}})}\n\n"
            return

        context_chunks = search_results[:self.max_context_chunks] if search_results else []
        sources = self._extract_sources(context_chunks) if context_chunks else []

        # Emit document sources immediately so frontend can show them while answer streams
        yield f"data: {json.dumps({'type': 'search_done', 'sources': sources})}\n\n"

        # Step 2: Build web search tool config
        tools = None
        if web_search_enabled:
            tools = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self._get_web_search_max_uses(),
            }]

        # Step 3: Build prompt (with web search awareness)
        prompt = self._build_prompt(question, context_chunks, conversation_history,
                                     web_search_enabled=web_search_enabled)

        # Step 4: Stream Claude response
        try:
            answer_parts: List[str] = []
            input_tokens = 0
            output_tokens = 0
            web_search_count = 0
            web_sources: List[Dict[str, Any]] = []

            # Increase limits when web search is enabled
            max_tokens = 4096 if tools else self.max_tokens
            timeout = 150 if tools else self.api_timeout

            stream_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "timeout": timeout,
                "messages": [{"role": "user", "content": prompt}],
            }
            if tools:
                stream_kwargs["tools"] = tools

            with self.client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    # Detect web search tool use starting
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, 'type') and event.content_block.type == "server_tool_use":
                            web_search_count += 1
                            yield f"data: {json.dumps({'type': 'web_search_status', 'status': 'searching', 'count': web_search_count})}\n\n"

                    # Stream text deltas to frontend (skip tool JSON deltas)
                    elif event.type == "content_block_delta" and hasattr(event.delta, "text"):
                        text = event.delta.text
                        answer_parts.append(text)
                        yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"

                # Get final usage and web sources from the accumulated message
                final_message = stream.get_final_message()
                input_tokens = final_message.usage.input_tokens
                output_tokens = final_message.usage.output_tokens

                # Extract web search sources from response content blocks
                if tools:
                    web_sources = self._extract_web_sources(final_message)

                    # Get accurate web search count from usage metadata
                    server_tool_use = getattr(final_message.usage, 'server_tool_use', None)
                    if server_tool_use:
                        web_search_count = getattr(server_tool_use, 'web_search_requests', web_search_count)

            full_answer = "".join(answer_parts)

            # Calculate cost (tokens + web search)
            pricing = self.pricing.get(self.model, {"input": 3.0, "output": 15.0})
            token_cost = (input_tokens * pricing["input"] / 1_000_000) + \
                         (output_tokens * pricing["output"] / 1_000_000)
            web_search_cost = web_search_count * 0.01
            cost = token_cost + web_search_cost

            avg_similarity = (sum(c['similarity_score'] for c in context_chunks) / len(context_chunks)) if context_chunks else 0.0
            response_time_ms = int((time.time() - start_time) * 1000)

            done_metadata = {
                'answer': full_answer,
                'sources': sources,
                'web_sources': web_sources,
                'web_search_count': web_search_count,
                'confidence_score': round(avg_similarity, 3),
                'tokens_used': input_tokens + output_tokens,
                'cost': round(cost, 6),
                'response_time_ms': response_time_ms,
                'model': self.model,
            }
            yield f"data: {json.dumps({'type': 'done', 'metadata': done_metadata})}\n\n"

            logger.success(
                f"Streamed answer in {response_time_ms}ms, cost: ${cost:.4f}"
                + (f", {web_search_count} web searches" if web_search_count else "")
            )

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


class FinancialQuestionAnswerer(QuestionAnswerer):
    """
    Extended Q&A system with financial model context enrichment.
    Automatically includes extracted financial metrics when answering finance-related questions.
    """

    # Keywords that indicate a financial question
    FINANCIAL_KEYWORDS = [
        'revenue', 'growth', 'margin', 'burn', 'runway', 'arr', 'mrr',
        'cac', 'ltv', 'churn', 'profit', 'loss', 'cash', 'ebitda',
        'operating', 'expenses', 'financial', 'projections', 'forecast',
        'unit economics', 'gross', 'net', 'income', 'cost', 'roi',
        'payback', 'retention', 'arpu', 'acv', 'customer acquisition'
    ]

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        max_context_chunks: int = 15
    ):
        super().__init__(anthropic_api_key, model, max_context_chunks)

    def _is_financial_question(self, question: str) -> bool:
        """Check if the question is finance-related."""
        question_lower = question.lower()
        return any(kw in question_lower for kw in self.FINANCIAL_KEYWORDS)

    def _get_financial_context(self, data_room_id: str) -> Optional[str]:
        """Get extracted financial metrics as context."""
        try:
            # Import database module
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from app import database as db

            # Get financial analyses
            analyses = db.get_financial_analyses_by_data_room(data_room_id)

            if not analyses:
                return None

            context_parts = ["## Extracted Financial Data\n"]

            for analysis in analyses:
                if analysis.get('status') != 'complete':
                    continue

                context_parts.append(f"### From: {analysis.get('file_name', 'Unknown')}\n")

                # Determine the base currency for this analysis
                structure = analysis.get('model_structure', {})
                base_currency = structure.get('base_currency', '')

                if base_currency:
                    context_parts.append(f"**Currency: {base_currency}**\n")

                # Add metrics
                metrics = analysis.get('extracted_metrics', [])
                if metrics:
                    context_parts.append("**Key Metrics:**")
                    for m in metrics[:20]:  # Limit to top 20 metrics
                        confidence = m.get('confidence', 'medium')
                        # Use base_currency if available, fall back to per-metric unit
                        unit = base_currency if base_currency else m.get('unit', '')
                        context_parts.append(
                            f"- {m.get('name')}: {m.get('value')} {unit} "
                            f"({m.get('period', 'N/A')}) [{confidence} confidence]"
                        )
                    context_parts.append("")

                # Add model structure info
                if structure:
                    context_parts.append("**Model Information:**")
                    if structure.get('revenue_model_type'):
                        context_parts.append(f"- Revenue Model: {structure['revenue_model_type']}")
                    if structure.get('projection_years'):
                        context_parts.append(f"- Projection Years: {structure.get('historical_years', 0)} historical, {structure['projection_years']} projected")
                    context_parts.append("")

                # Add key insights
                insights = analysis.get('insights', [])[:5]
                if insights:
                    context_parts.append("**Key Insights:**")
                    for insight in insights:
                        context_parts.append(f"- [{insight.get('category')}] {insight.get('title')}: {insight.get('insight')[:200]}")
                    context_parts.append("")

            return "\n".join(context_parts) if len(context_parts) > 1 else None

        except Exception as e:
            logger.warning(f"Failed to get financial context: {e}")
            return None

    def answer(
        self,
        question: str,
        data_room_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question with optional financial context enrichment.
        """
        self._prepare_financial_context(question, data_room_id)
        return super().answer(question, data_room_id, conversation_history, filters)

    def _prepare_financial_context(self, question: str, data_room_id: str) -> None:
        """Set self._financial_context if the question is financial."""
        if self._is_financial_question(question):
            financial_context = self._get_financial_context(data_room_id)
            if financial_context:
                logger.info(f"Adding financial context to question: {question[:50]}...")
                self._financial_context = f"""

Note: This question appears to be about financials. I have extracted financial metrics available from analyzed Excel files in this data room. Please use these extracted values to provide precise answers with specific numbers when available.

{financial_context}

---

Now answer the following question using both the extracted financial data above AND the document context below. Prioritize the extracted metrics for specific numbers, but use document context for additional details and verification.
"""
            else:
                self._financial_context = None
        else:
            self._financial_context = None

    def answer_stream(
        self,
        question: str,
        data_room_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        """Stream with financial context enrichment."""
        self._prepare_financial_context(question, data_room_id)
        yield from super().answer_stream(question, data_room_id, conversation_history, filters)

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        web_search_enabled: bool = False
    ) -> str:
        """Build prompt with optional financial context."""

        # Get base prompt
        base_prompt = super()._build_prompt(question, context_chunks, conversation_history, web_search_enabled=web_search_enabled)

        # Inject financial context if available
        if hasattr(self, '_financial_context') and self._financial_context:
            # Insert financial context before the question
            parts = base_prompt.split("Question:")
            if len(parts) == 2:
                base_prompt = parts[0] + self._financial_context + "\nQuestion:" + parts[1]

        return base_prompt


# Keywords that indicate a question would benefit from rich analytics (charts/tables)
ANALYTICAL_KEYWORDS = [
    'compare', 'comparison', 'trend', 'trends', 'over time', 'year over year',
    'yoy', 'qoq', 'quarter over quarter', 'month over month',
    'growth rate', 'trajectory', 'chart', 'graph', 'plot', 'visualize',
    'visualization', 'show me', 'breakdown', 'distribution', 'composition',
    'top 5', 'top 10', 'ranking', 'rank', 'highest', 'lowest',
    'revenue by', 'expenses by', 'margins by', 'split by',
    'segment', 'cohort', 'per unit',
    'scenario', 'forecast', 'projection', 'sensitivity',
    'best case', 'worst case', 'base case',
]


def _is_analytical_question(question: str) -> bool:
    """Detect if a question would benefit from rich analytics (charts/tables)."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in ANALYTICAL_KEYWORDS)


class AnalyticalQuestionAnswerer(FinancialQuestionAnswerer):
    """
    Extended Q&A with inline chart generation for analytical questions.
    Uses Claude Opus for higher-quality data analysis and returns chart specs
    alongside the text answer.
    """

    ANALYTICS_PROMPT_ADDITION = """

## Charts & Visualizations
When your answer involves numerical data that would benefit from a chart or graph
(comparisons, trends, distributions, rankings), you MUST include a visualization.

Output chart data in this exact tag format AFTER your text answer:

<analytics_charts>
{"charts": [{"id": "unique_id", "title": "Chart Title", "type": "bar", "x_key": "label", "y_key": "value", "x_label": "", "y_label": "", "y_format": "number", "color_key": "", "colors": "#9d174d", "data": [{"label": "A", "value": 100}]}]}
</analytics_charts>

Chart spec fields:
- id: unique string (e.g. "revenue_trend", "margin_comparison")
- title: descriptive chart title
- type: "bar" | "horizontal_bar" | "line"
- x_key, y_key: keys matching data point objects
- x_label, y_label: axis labels (use empty string "" if not needed)
- y_format: "currency" | "number" | "percent"
- color_key: data key for per-bar coloring (use empty string "" if not needed)
- colors: single hex color string OR object mapping color_key values to hex colors
- data: array of objects with keys matching x_key, y_key, and color_key

Use these colors: #9d174d (primary), #be185d, #e11d48, #881337.
All numeric values must be raw numbers, not formatted strings (e.g. 1200000 not "$1.2M").
If the user explicitly asks for a chart or visualization, you MUST always generate one. Otherwise, include charts only when the data genuinely warrants visualization.
You may include multiple charts if the analysis covers different dimensions.
"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_context_chunks: int = 15
    ):
        # Default to Opus for analytical questions
        if model is None:
            try:
                from app.config import settings
                model = settings.claude_opus_model
            except Exception:
                model = os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-5-20251101")
        super().__init__(anthropic_api_key, model, max_context_chunks)
        self.max_tokens = 4096  # Larger limit for chart JSON output
        self.api_timeout = 120  # Opus needs more time

    def answer(
        self,
        question: str,
        data_room_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result = super().answer(question, data_room_id, conversation_history, filters)
        result = self._parse_charts(result)
        result['is_analytical'] = True
        return result

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        web_search_enabled: bool = False
    ) -> str:
        base_prompt = super()._build_prompt(question, context_chunks, conversation_history, web_search_enabled=web_search_enabled)

        # Inject analytics instructions before the question
        parts = base_prompt.split("Question:")
        if len(parts) == 2:
            base_prompt = parts[0] + self.ANALYTICS_PROMPT_ADDITION + "\nQuestion:" + parts[1]

        return base_prompt

    def _parse_charts(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse <analytics_charts> tags from answer and extract chart specs."""
        answer_text = result.get('answer', '')
        charts_match = re.search(
            r'<analytics_charts>\s*(.*?)\s*</analytics_charts>',
            answer_text, re.DOTALL
        )
        if charts_match:
            try:
                charts_json = json.loads(charts_match.group(1).strip())
                if isinstance(charts_json, dict) and 'charts' in charts_json:
                    result['charts'] = charts_json['charts']
                    # Strip the tag from the displayed answer
                    result['answer'] = answer_text[:charts_match.start()].strip()
                    logger.info(f"Parsed {len(result['charts'])} chart(s) from analytical response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse analytics charts JSON: {e}")
        return result


def answer_question(
    question: str,
    data_room_id: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    filters: Optional[Dict[str, Any]] = None,
    model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
    use_financial_context: bool = True,
    enable_analytics: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to answer a question.

    Args:
        question: Analyst's question
        data_room_id: Data room ID
        conversation_history: Previous Q&A turns
        filters: Optional search filters
        model: Claude model to use
        use_financial_context: Whether to include extracted financial data
        enable_analytics: Whether to detect analytical questions and use Opus with charts

    Returns:
        Answer with sources (and optional charts for analytical questions)

    Example:
        >>> answer = answer_question(
        ...     question="What is the company's burn rate?",
        ...     data_room_id="deal_123"
        ... )
        >>> print(answer['answer'])
        >>> for source in answer['sources']:
        ...     print(f"Source: {source['file_name']}")
    """
    if enable_analytics and _is_analytical_question(question):
        answerer = AnalyticalQuestionAnswerer()
        return answerer.answer(question, data_room_id, conversation_history, filters)

    if use_financial_context:
        answerer = FinancialQuestionAnswerer(model=model)
    else:
        answerer = QuestionAnswerer(model=model)

    return answerer.answer(question, data_room_id, conversation_history, filters)


def answer_question_stream(
    question: str,
    data_room_id: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    filters: Optional[Dict[str, Any]] = None,
    model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
    use_financial_context: bool = True,
) -> Generator[str, None, None]:
    """
    Streaming version of answer_question. Yields SSE-formatted events.
    Falls back to non-streaming for analytical questions (they need post-processing for charts).
    """
    if use_financial_context:
        answerer = FinancialQuestionAnswerer(model=model)
    else:
        answerer = QuestionAnswerer(model=model)

    yield from answerer.answer_stream(question, data_room_id, conversation_history, filters)


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Answer questions about data room")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--data-room-id", required=True, help="Data room ID")
    parser.add_argument("--model", default=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"), help="Claude model")
    parser.add_argument("--filter-category", help="Filter by document category")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    # Build filters
    filters = {}
    if args.filter_category:
        filters['document_category'] = args.filter_category

    # Answer question
    result = answer_question(
        question=args.question,
        data_room_id=args.data_room_id,
        filters=filters if filters else None,
        model=args.model
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Saved answer to {output_path}")
    else:
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"{'='*60}\n")

        print("Answer:")
        print(result['answer'])
        print()

        print(f"{'='*60}")
        print(f"Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            page_info = f", p.{source['page_number']}" if source['page_number'] else ""
            print(f"{i}. {source['file_name']}{page_info} (relevance: {source['relevance_score']:.3f})")

        print(f"\nConfidence: {result['confidence_score']:.3f}")
        print(f"Cost: ${result['cost']:.4f}")
        print(f"Response Time: {result['response_time_ms']}ms")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
