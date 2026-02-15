"""
RAG-powered Q&A tool using semantic search and Claude API.
Answers analyst questions with citations from data room documents.
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
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
        Answer a question using RAG pipeline.

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

        # Step 1: Semantic search for relevant chunks
        search_results = semantic_search(
            query=question,
            data_room_id=data_room_id,
            top_k=self.max_context_chunks * 3,  # Get extra for reranking
            filters=filters
        )

        if not search_results:
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
        context_chunks = search_results[:self.max_context_chunks]

        # Step 3: Build prompt with context
        prompt = self._build_prompt(question, context_chunks, conversation_history)

        # Step 4: Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.3,
                timeout=self.api_timeout,
                messages=[{"role": "user", "content": prompt}]
            )

            answer_text = response.content[0].text

            # Track usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # Calculate cost
            pricing = self.pricing.get(self.model, {"input": 3.0, "output": 15.0})
            cost = (input_tokens * pricing["input"] / 1_000_000) + \
                   (output_tokens * pricing["output"] / 1_000_000)
            self.total_cost += cost

            # Step 5: Extract and validate citations
            sources = self._extract_sources(context_chunks)

            # Calculate confidence based on similarity scores
            avg_similarity = sum(c['similarity_score'] for c in context_chunks) / len(context_chunks)
            confidence_score = round(avg_similarity, 3)

            response_time_ms = int((time.time() - start_time) * 1000)

            result = {
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "confidence_score": confidence_score,
                "tokens_used": input_tokens + output_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": round(cost, 6),
                "response_time_ms": response_time_ms,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }

            logger.success(f"Generated answer in {response_time_ms}ms, cost: ${cost:.4f}")

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

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build prompt for Claude with context and question."""

        # System instructions
        system_prompt = """You are an AI assistant helping venture capital analysts with due diligence.

Your role:
- Answer questions accurately based ONLY on the provided context from data room documents
- If information is not in the context, clearly state "This information is not available in the provided documents"
- Be concise but thorough
- Use professional VC analysis language
- If you need to make calculations or inferences, explain your reasoning
- Never make up information or hallucinate facts
- At the end of your answer, include a brief "Sources:" line listing the documents you referenced (e.g., Sources: filename.pdf (p.3), model.xlsx (Sheet: P&L)). Only list documents you actually drew information from.
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
        # Check if this is a financial question
        if self._is_financial_question(question):
            # Get financial context
            financial_context = self._get_financial_context(data_room_id)

            if financial_context:
                logger.info(f"Adding financial context to question: {question[:50]}...")

                # Enhance the question with context note
                enhanced_prompt_note = f"""

Note: This question appears to be about financials. I have extracted financial metrics available from analyzed Excel files in this data room. Please use these extracted values to provide precise answers with specific numbers when available.

{financial_context}

---

Now answer the following question using both the extracted financial data above AND the document context below. Prioritize the extracted metrics for specific numbers, but use document context for additional details and verification.
"""
                # Store the financial context to inject into the prompt
                self._financial_context = enhanced_prompt_note
            else:
                self._financial_context = None
        else:
            self._financial_context = None

        # Call parent answer method
        return super().answer(question, data_room_id, conversation_history, filters)

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build prompt with optional financial context."""

        # Get base prompt
        base_prompt = super()._build_prompt(question, context_chunks, conversation_history)

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
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        base_prompt = super()._build_prompt(question, context_chunks, conversation_history)

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
