"""
Investment memo generator using RAG pipeline.
Generates structured investment memos from data room documents.
"""

import os
import time
import signal
import threading
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

try:
    from anthropic import Anthropic, APIError, RateLimitError, APITimeoutError
    from dotenv import load_dotenv
except ImportError:
    logger.error("Required packages not installed. Run: pip install anthropic python-dotenv")
    # Define fallback exception classes
    APIError = Exception
    RateLimitError = Exception
    APITimeoutError = Exception

try:
    from semantic_search import semantic_search
except ImportError:
    from tools.semantic_search import semantic_search

load_dotenv()

SECTIONS = [
    {
        "key": "proposed_investment_terms",
        "label": "Proposed Investment Terms",
        "phase": 1,
        "search_queries": [
            "investment round funding terms SAFE convertible note valuation cap",
            "deal structure equity ownership investors round size date",
        ],
        "system_prompt": (
            "Write a concise Proposed Investment Terms section for a VC investment memo. "
            "Present the key deal terms in a clean markdown table with two columns (Term, Details). "
            "Include rows for: Round Type, Total Round Size, Investment Amount, Post-Money Valuation, "
            "Investor Ownership %, Investment Vehicle (SAFE/Convertible Note/Priced Round), and Lead Investor if known. "
            "If the analyst provided deal parameters (Investment Amount or Post-Money Valuation), "
            "use those exact values in the table instead of values from the data room. "
            "IMPORTANT: Include exactly ONE row for Post-Money Valuation showing only the single latest number. "
            "Data rooms often contain terms from multiple funding rounds — look at dates, round labels (e.g. Series A vs Seed), "
            "and document recency to identify the most recent round. Do not list prior-round valuations. "
            "Only include terms that are found in the data room documents or provided by the analyst. Keep it factual, no analysis."
        ),
        "use_opus": False,
    },
    {
        "key": "executive_summary",
        "label": "Executive Summary",
        "phase": 1,
        "search_queries": [
            "company overview mission product market business model",
            "traction key metrics customers revenue growth",
        ],
        "system_prompt": (
            "Write a concise executive summary for a VC investment memo. "
            "Cover: what the company does, the problem it solves, business model, "
            "key traction metrics, and why this could be a compelling investment opportunity. "
            "Use 2-3 paragraphs."
        ),
        "use_opus": False,
    },
    {
        "key": "market_analysis",
        "label": "Market Analysis",
        "phase": 1,
        "search_queries": [
            "total addressable market TAM SAM SOM market size opportunity",
            "competitors competitive landscape differentiation positioning",
        ],
        "system_prompt": (
            "Write the Market Analysis section of a VC investment memo. "
            "Cover: market size (TAM/SAM/SOM if available), growth trends, "
            "competitive landscape, key competitors, and the company's competitive advantages."
        ),
        "use_opus": False,
    },
    {
        "key": "team_assessment",
        "label": "Team Assessment",
        "phase": 1,
        "search_queries": [
            "founders team CEO CTO background experience leadership",
            "advisors board members management team hiring",
        ],
        "system_prompt": (
            "Write the Team Assessment section of a VC investment memo. "
            "Cover: founder backgrounds and relevant experience, team completeness, "
            "key hires, advisory board, and any concerns about the team."
        ),
        "use_opus": False,
    },
    {
        "key": "product_technology",
        "label": "Product & Technology",
        "phase": 1,
        "search_queries": [
            "product features technology platform architecture stack",
            "intellectual property patents technical moat innovation",
        ],
        "system_prompt": (
            "Write the Product & Technology section of a VC investment memo. "
            "Cover: product description, technology stack, technical moat/IP, "
            "product roadmap, and current stage of development."
        ),
        "use_opus": False,
    },
    {
        "key": "financial_analysis",
        "label": "Financial Analysis",
        "phase": 2,
        "search_queries": [
            "revenue growth MRR ARR burn rate runway profitability",
            "unit economics CAC LTV gross margin operating expenses",
            "funding raised valuation cap table investors round",
        ],
        "system_prompt": (
            "Write the Financial Analysis section of a VC investment memo. "
            "Cover: current revenue and growth trajectory, burn rate and runway, "
            "unit economics (CAC/LTV), margin profile, funding history, "
            "and financial projections. Use specific numbers where available."
        ),
        "use_opus": True,
    },
    {
        "key": "valuation_analysis",
        "label": "Valuation Analysis",
        "phase": 2,
        "search_queries": [
            "valuation revenue multiple ARR MRR growth rate comparable companies funding round",
            "financial projections cash flow forecast exit potential market multiples",
        ],
        "system_prompt": None,  # Dynamic - built based on selected valuation methods
        "use_opus": False,
    },
    {
        "key": "risks_concerns",
        "label": "Risks & Concerns",
        "phase": 2,
        "search_queries": [
            "risks challenges concerns regulatory competition threats",
            "weaknesses dependencies concentration customer churn",
        ],
        "system_prompt": (
            "Write the Risks & Concerns section of a VC investment memo. "
            "Cover: key risks (market, execution, technical, regulatory), "
            "dependencies, concentration risks, and any red flags identified. "
            "Be thorough and honest."
        ),
        "use_opus": False,
    },
    {
        "key": "outcome_scenario_analysis",
        "label": "Outcome Scenario Analysis",
        "phase": 3,
        "search_queries": [
            "revenue growth MRR ARR valuation exit potential IPO acquisition",
            "funding raised valuation cap table investors round terms",
            "market size TAM total addressable market opportunity upside",
        ],
        "system_prompt": (
            "Write an Outcome Scenario Analysis section for a VC investment memo. "
            "Produce a single markdown table modeling 6 potential outcomes. "
            "The table must have these columns: Outcome, Exit Value, Exit Date, Gross Proceeds, Mult., IRR.\n\n"
            "In the **Outcome** column, put the scenario name in bold followed by a period, "
            "then a 2-3 sentence description of what happens in that scenario — all within the same cell. "
            "Use <br> tags for line breaks within the cell.\n\n"
            "The 6 scenarios (from worst to best) are:\n"
            "1. **Wipeout** — Company fails, total loss.\n"
            "2. **No Growth** — Stagnates, sold at a loss or small return.\n"
            "3. **Slow Growth** — Modest traction, moderate exit.\n"
            "4. **Strong Growth** — Solid execution, good exit.\n"
            "5. **Market Leader** — Becomes a category leader, large exit.\n"
            "6. **Home Run** — Exceptional outcome, massive exit.\n\n"
            "Base exit values, multiples, and IRRs on the company's current metrics, "
            "market size, competitive position, and funding stage found in the data room. "
            "Use the investment terms from the Proposed Investment Terms section as the basis "
            "for calculating ownership, proceeds, and returns."
        ),
        "use_opus": True,
    },
    {
        "key": "investment_recommendation",
        "label": "Investment Recommendation",
        "phase": 3,
        "search_queries": [
            "investment opportunity valuation terms deal structure",
            "company strengths growth potential return thesis",
        ],
        "system_prompt": (
            "Write the Investment Recommendation section of a VC investment memo. "
            "Synthesize all the analysis from prior sections into a clear recommendation. "
            "Cover: investment thesis, key strengths, key concerns, "
            "and overall recommendation (invest / pass / more info needed). "
            "Do NOT include a 'Suggested Next Steps' subsection."
        ),
        "use_opus": True,
    },
]

class SonnetTokenTracker:
    """
    Thread-safe tracker for Sonnet API input token usage within a rolling
    60-second window. Calculates the minimum cooldown needed before making
    additional Sonnet calls without exceeding the rate limit.
    """

    WINDOW_SECONDS = 60
    RATE_LIMIT = 30_000          # input tokens per minute
    SAFETY_MARGIN_SECONDS = 5

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: List[Tuple[float, int]] = []

    def record(self, input_tokens: int) -> None:
        """Record a Sonnet API call that just completed."""
        with self._lock:
            self._calls.append((time.time(), input_tokens))

    def required_wait(self) -> float:
        """
        Return the number of seconds to sleep before the next Sonnet call.
        Prunes expired entries, then checks if the rolling window has capacity.
        """
        with self._lock:
            now = time.time()
            cutoff = now - self.WINDOW_SECONDS
            self._calls = [(ts, tok) for ts, tok in self._calls if ts > cutoff]

            if not self._calls:
                return 0.0

            total_in_window = sum(tok for _, tok in self._calls)
            if total_in_window < self.RATE_LIMIT:
                return 0.0

            # Find the earliest call whose expiry brings us under the limit
            calls_sorted = sorted(self._calls, key=lambda x: x[0])
            tokens_to_shed = total_in_window - self.RATE_LIMIT
            shed = 0
            wait_until = now

            for ts, tok in calls_sorted:
                shed += tok
                wait_until = ts + self.WINDOW_SECONDS
                if shed >= tokens_to_shed:
                    break

            wait = wait_until - now + self.SAFETY_MARGIN_SECONDS
            return max(wait, 0.0)


# Model pricing (per 1M tokens)
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
}


def _normalize_key(key: str) -> str:
    """Normalize a key for fuzzy comparison (lowercase, strip separators)."""
    return key.lower().replace("_", "").replace("-", "").replace(" ", "")


def _coerce_numeric(val: str):
    """Attempt to coerce a formatted string to a number."""
    cleaned = val.strip().lstrip("$")
    cleaned = cleaned.rstrip("xX%")
    cleaned = cleaned.replace(",", "")

    multiplier = 1
    if cleaned.upper().endswith("B"):
        multiplier = 1_000_000_000
        cleaned = cleaned[:-1]
    elif cleaned.upper().endswith("M"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    elif cleaned.upper().endswith("K"):
        multiplier = 1_000
        cleaned = cleaned[:-1]

    try:
        return float(cleaned) * multiplier
    except (ValueError, TypeError):
        return val  # Return original; caller will skip this point


def _resolve_key(declared_key: str, actual_keys: set, exclude: set, spec_id: str = "?") -> str:
    """
    If declared_key exists in actual_keys, return it.
    Otherwise attempt fuzzy matching, then fall back to the sole remaining numeric key.
    """
    if declared_key in actual_keys:
        return declared_key

    norm_declared = _normalize_key(declared_key)
    candidates = []
    for ak in actual_keys:
        if ak in exclude:
            continue
        norm_ak = _normalize_key(ak)
        if norm_ak == norm_declared:
            candidates.append(ak)
        elif norm_declared in norm_ak or norm_ak in norm_declared:
            candidates.append(ak)

    if len(candidates) == 1:
        logger.info(f"Chart '{spec_id}': auto-fixed key '{declared_key}' -> '{candidates[0]}'")
        return candidates[0]

    # Fallback: if only one non-excluded key remains, use it
    remaining = actual_keys - exclude - {declared_key}
    if len(remaining) == 1:
        fallback = remaining.pop()
        logger.info(f"Chart '{spec_id}': fallback key '{declared_key}' -> '{fallback}'")
        return fallback

    logger.warning(f"Chart '{spec_id}': could not resolve key '{declared_key}' in {actual_keys}")
    return declared_key


def _validate_and_fix_chart_specs(chart_data: dict) -> dict:
    """
    Validate and auto-repair chart specs generated by Claude.
    Fixes key mismatches, coerces string values to numbers,
    and filters out unfixable charts.
    """
    if not isinstance(chart_data, dict) or "charts" not in chart_data:
        return {}

    charts = chart_data.get("charts", [])
    if not isinstance(charts, list):
        return {}

    valid_charts = []

    for spec in charts:
        if not isinstance(spec, dict):
            continue

        required = ["id", "title", "type", "x_key", "y_key", "data"]
        if not all(k in spec for k in required):
            logger.warning(f"Chart '{spec.get('id', '?')}' missing required fields, skipping")
            continue

        data = spec["data"]
        if not isinstance(data, list) or len(data) == 0:
            continue

        x_key = spec["x_key"]
        y_key = spec["y_key"]

        # Key mismatch detection & repair
        sample = data[0]
        actual_keys = set(sample.keys())

        y_key = _resolve_key(y_key, actual_keys, exclude={x_key}, spec_id=spec.get("id"))
        x_key = _resolve_key(x_key, actual_keys, exclude={y_key}, spec_id=spec.get("id"))

        spec["x_key"] = x_key
        spec["y_key"] = y_key

        # Validate color_key
        color_key = spec.get("color_key", "")
        if color_key and color_key not in actual_keys:
            logger.info(f"Chart '{spec.get('id')}': color_key '{color_key}' not in data, clearing")
            spec["color_key"] = ""

        # Type coercion & filtering
        cleaned_data = []
        for point in data:
            if y_key not in point:
                continue
            val = point[y_key]
            if isinstance(val, str):
                val = _coerce_numeric(val)
            if isinstance(val, (int, float)):
                point[y_key] = val
                cleaned_data.append(point)

        if len(cleaned_data) < 2:
            logger.warning(f"Chart '{spec.get('id')}': only {len(cleaned_data)} valid data points after cleanup, skipping")
            continue

        spec["data"] = cleaned_data
        valid_charts.append(spec)

    if not valid_charts:
        return {}

    logger.info(f"Chart validation: {len(valid_charts)}/{len(charts)} charts passed")
    return {"charts": valid_charts}


def _generate_chart_data_via_claude(sections: Dict[str, str], api_key: str, model: str = "claude-sonnet-4-20250514") -> Dict[str, Any]:
    """
    Use Claude to generate structured chart specifications from memo sections.
    Returns a dict with a 'charts' list that the frontend can render dynamically.
    """
    import json as _json

    # Build context from the most chart-relevant sections
    context_parts = []
    for key in ("financial_analysis", "executive_summary", "proposed_investment_terms"):
        text = sections.get(key, "")
        if text:
            label = key.replace("_", " ").title()
            context_parts.append(f"## {label}\n\n{text}")

    if not context_parts:
        return {}

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a data visualization specialist. Analyze the investment memo sections below and "
        "produce 2-3 charts that visualize the most important financial and growth metrics.\n\n"
        "Output ONLY a valid JSON object with a single key \"charts\" containing an array of chart specs.\n"
        "Each chart spec must have exactly these fields:\n"
        "- \"id\": unique string identifier (e.g. \"revenue\", \"users\", \"growth\")\n"
        "- \"title\": chart title string\n"
        "- \"type\": one of \"bar\", \"horizontal_bar\", \"line\"\n"
        "- \"x_key\": the data key for x-axis\n"
        "- \"y_key\": the data key for y-axis (the numeric value)\n"
        "- \"x_label\": axis label for x-axis (optional, empty string if not needed)\n"
        "- \"y_label\": axis label for y-axis (optional, empty string if not needed)\n"
        "- \"y_format\": how to format y values — one of \"currency\", \"number\", \"percent\"\n"
        "- \"color_key\": optional data key used to color bars differently (e.g. \"label\" if data has Actual/Projected). Use empty string if not needed.\n"
        "- \"colors\": object mapping color_key values to hex colors, OR a single hex color string if no color_key. Use indigo palette (#6366f1 primary, #a5b4fc light, #4f46e5 dark).\n"
        "- \"data\": array of data point objects. Each object must have keys matching x_key and y_key, plus color_key if used.\n\n"
        "Rules:\n"
        "- Only include charts where you have actual numeric data from the memo.\n"
        "- For revenue/financial projections, use a \"bar\" chart with years on x-axis.\n"
        "- For user/platform metrics comparison, use a \"horizontal_bar\" chart.\n"
        "- All numeric values must be raw numbers (not formatted strings).\n"
        "- If data has actual vs. projected distinction, include a color_key \"label\" with values \"Actual\" and \"Projected\".\n"
        "- Do NOT include any text outside the JSON object. No markdown, no explanation.\n\n"
        f"Memo sections:\n\n{context}"
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
        if raw.startswith("json"):
            raw = raw[4:].lstrip()

        chart_data = _json.loads(raw)

        # Validate structure
        if not isinstance(chart_data, dict) or "charts" not in chart_data:
            logger.warning("Claude chart response missing 'charts' key")
            return {}
        if not isinstance(chart_data["charts"], list):
            logger.warning("Claude chart response 'charts' is not a list")
            return {}

        # Validate and auto-fix chart specs (key mismatches, string values, etc.)
        chart_data = _validate_and_fix_chart_specs(chart_data)
        if not chart_data:
            logger.warning("All chart specs failed validation")
            return {}

        logger.info(f"Chart generation: {len(chart_data['charts'])} charts validated")
        return chart_data

    except Exception as e:
        logger.warning(f"Failed to generate chart data via Claude: {e}")
        return {}


class MemoGenerator:
    """Generates investment memos from data room documents using RAG."""

    def __init__(self, data_room_id: str):
        self.data_room_id = data_room_id

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._api_key = api_key
        self._thread_local = threading.local()
        self.default_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.opus_model = os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-5-20251101")
        self.total_tokens = 0
        self.total_cost = 0.0
        self._stats_lock = threading.Lock()
        self._sonnet_tracker = SonnetTokenTracker()

    def _get_client(self) -> Anthropic:
        """Get a thread-local Anthropic client (httpx.Client is not thread-safe)."""
        if not hasattr(self._thread_local, 'client'):
            self._thread_local.client = Anthropic(api_key=self._api_key)
        return self._thread_local.client

    def generate_memo(self, memo_id: str, save_section_fn=None, update_status_fn=None, deal_params: Optional[Dict[str, Any]] = None, check_cancelled_fn=None, save_metadata_fn=None):
        """
        Generate memo sections in parallel phases with section-level error recovery.

        Phases:
            1 (parallel): proposed_investment_terms, executive_summary, market_analysis, team_assessment, product_technology
            2 (parallel): financial_analysis, valuation_analysis, risks_concerns
            3 (sequential): outcome_scenario_analysis, investment_recommendation

        Args:
            memo_id: Memo record ID
            save_section_fn: Callable(memo_id, section_key, content, tokens, cost)
            update_status_fn: Callable(memo_id, status, full_memo)
            deal_params: Optional dict with ticket_size and post_money_valuation
            check_cancelled_fn: Callable(memo_id) -> bool, checks if generation was cancelled
            save_metadata_fn: Callable(memo_id, metadata_dict), saves chart data to metadata column
        """
        logger.info(f"Starting memo generation for data room {self.data_room_id}, deal_params={deal_params}")
        completed_sections: Dict[str, str] = {}
        failed_sections: List[str] = []

        for phase in [1, 2, 3]:
            # Check if cancelled before starting each phase
            if check_cancelled_fn and check_cancelled_fn(memo_id):
                logger.info(f"Memo {memo_id} cancelled by user — stopping generation")
                if update_status_fn:
                    full_memo = self._compile_full_memo(completed_sections) if completed_sections else None
                    update_status_fn(memo_id, "cancelled", full_memo)
                return completed_sections

            phase_sections = [s for s in SECTIONS if s["phase"] == phase]
            logger.info(f"Starting phase {phase}: {[s['label'] for s in phase_sections]}")

            # Snapshot prior sections so parallel threads share the same context
            prior_snapshot = dict(completed_sections)

            # Split into Opus and Sonnet sections — they have separate API rate limits
            opus_sections = [s for s in phase_sections if s.get("use_opus")]
            sonnet_sections = [s for s in phase_sections if not s.get("use_opus")]

            # Start Opus sections in background immediately (separate rate limit)
            opus_futures = {}
            executor = ThreadPoolExecutor(max_workers=max(len(opus_sections), 1)) if opus_sections else None
            try:
                for section_def in opus_sections:
                    logger.info(f"Generating section (Opus, background): {section_def['label']}")
                    future = executor.submit(
                        self._generate_section, section_def, prior_snapshot, deal_params
                    )
                    opus_futures[future] = section_def

                # Smart cooldown: only wait if this phase has Sonnet sections
                # AND there are recent Sonnet tokens in the rolling window.
                if phase > 1 and sonnet_sections:
                    wait = self._sonnet_tracker.required_wait()
                    if wait > 0:
                        logger.info(f"Rate limit cooldown before phase {phase} Sonnet sections ({wait:.0f}s)")
                        time.sleep(wait)
                    else:
                        logger.info(f"No cooldown needed before phase {phase} — Sonnet token window has capacity")

                # Run Sonnet sections sequentially (rate limit too low for parallel)
                for section_def in sonnet_sections:
                    # Check cancellation between sequential sections
                    if check_cancelled_fn and check_cancelled_fn(memo_id):
                        logger.info(f"Memo {memo_id} cancelled by user — stopping generation")
                        if update_status_fn:
                            full_memo = self._compile_full_memo(completed_sections) if completed_sections else None
                            update_status_fn(memo_id, "cancelled", full_memo)
                        return completed_sections

                    key = section_def["key"]
                    label = section_def["label"]
                    logger.info(f"Generating section: {label}")
                    try:
                        content, tokens, cost = self._generate_section(
                            section_def, prior_snapshot, deal_params
                        )
                        completed_sections[key] = content
                        if save_section_fn:
                            save_section_fn(memo_id, key, content, tokens, cost)
                        logger.success(f"Completed section: {label} ({tokens} tokens, ${cost:.4f})")
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Section '{label}' failed: {error_msg}")
                        failed_sections.append(key)
                        placeholder = f"*Section generation failed: {error_msg}. Please retry memo generation.*"
                        completed_sections[key] = placeholder
                        if save_section_fn:
                            try:
                                save_section_fn(memo_id, key, placeholder, 0, 0.0)
                            except Exception:
                                pass

                # Wait for background Opus sections to complete
                for future in as_completed(opus_futures):
                    section_def = opus_futures[future]
                    key = section_def["key"]
                    label = section_def["label"]
                    try:
                        content, tokens, cost = future.result()
                        completed_sections[key] = content
                        if save_section_fn:
                            save_section_fn(memo_id, key, content, tokens, cost)
                        logger.success(f"Completed section: {label} ({tokens} tokens, ${cost:.4f})")
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Section '{label}' failed: {error_msg}")
                        failed_sections.append(key)
                        placeholder = f"*Section generation failed: {error_msg}. Please retry memo generation.*"
                        completed_sections[key] = placeholder
                        if save_section_fn:
                            try:
                                save_section_fn(memo_id, key, placeholder, 0, 0.0)
                            except Exception:
                                pass
            finally:
                if executor:
                    executor.shutdown(wait=False)

        # Check if cancelled during the last phase before overwriting status
        if check_cancelled_fn and check_cancelled_fn(memo_id):
            logger.info(f"Memo {memo_id} was cancelled during generation — preserving cancelled status")
            full_memo = self._compile_full_memo(completed_sections) if completed_sections else None
            if update_status_fn:
                update_status_fn(memo_id, "cancelled", full_memo)
            return completed_sections

        # Generate chart data from completed sections using Claude
        if save_metadata_fn:
            try:
                chart_data = _generate_chart_data_via_claude(
                    completed_sections,
                    api_key=self._api_key,
                    model=self.default_model,
                )
                if chart_data:
                    save_metadata_fn(memo_id, {"chart_data": chart_data})
                    logger.info(f"Saved chart data for memo {memo_id}")
            except Exception as e:
                logger.warning(f"Failed to generate chart data: {e}")

        # Compile full memo (including any failed sections with placeholders)
        full_memo = self._compile_full_memo(completed_sections)

        # Determine final status
        if len(failed_sections) == len(SECTIONS):
            status = "failed"
            logger.error(f"Memo generation failed: all sections failed")
        elif failed_sections:
            status = "complete"
            logger.warning(f"Memo generation complete with errors: {len(failed_sections)} sections failed: {failed_sections}")
        else:
            status = "complete"
            logger.success(
                f"Memo generation complete. Total: {self.total_tokens} tokens, ${self.total_cost:.4f}"
            )

        if update_status_fn:
            update_status_fn(memo_id, status, full_memo)

        return completed_sections

    def _call_claude_with_retry(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        max_retries: int = 3,
        timeout_seconds: int = 120
    ) -> Tuple[str, int, int]:
        """
        Call Claude API with retry logic, response validation, and timeout.

        Returns: (content_text, input_tokens, output_tokens)
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                start = time.time()

                response = self._get_client().messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    timeout=timeout_seconds,
                    messages=[{"role": "user", "content": prompt}],
                )

                elapsed = time.time() - start

                # Validate response structure
                if not response.content:
                    raise ValueError("Empty response.content from Claude API")

                if len(response.content) == 0:
                    raise ValueError("response.content is empty list")

                first_block = response.content[0]
                if not hasattr(first_block, 'text'):
                    raise ValueError(f"Unexpected response block type: {type(first_block)}")

                content = first_block.text
                if not content or not content.strip():
                    raise ValueError("Response text is empty or whitespace")

                logger.debug(f"Claude API call succeeded in {elapsed:.1f}s (attempt {attempt + 1})")

                return content, response.usage.input_tokens, response.usage.output_tokens

            except RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 15 + (attempt * 15)  # 15, 30 seconds (rate limits are per-minute)
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")

            except APITimeoutError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 3  # 3, 6, 12 seconds
                    logger.warning(f"API timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API timeout after {max_retries} attempts")

            except APIError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 2
                    logger.warning(f"API error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error after {max_retries} attempts: {e}")

            except ValueError as e:
                # Response validation errors - retry immediately
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Response validation failed, retrying (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Response validation failed after {max_retries} attempts: {e}")

            except Exception as e:
                # Unexpected errors - don't retry
                logger.error(f"Unexpected error calling Claude API: {e}")
                raise

        # All retries exhausted
        raise last_error or Exception(f"Claude API call failed after {max_retries} attempts")

    def _generate_section(
        self,
        section_def: Dict[str, Any],
        prior_sections: Dict[str, str],
        deal_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int, float]:
        """Generate a single memo section. Returns (content, tokens_used, cost)."""

        # 1. Semantic search with multiple queries IN PARALLEL
        all_chunks = []
        seen_ids = set()

        def _run_search(query: str):
            try:
                return semantic_search(
                    query=query,
                    data_room_id=self.data_room_id,
                    top_k=15,
                )
            except Exception as e:
                logger.warning(f"Semantic search failed for query '{query}': {e}")
                return []

        with ThreadPoolExecutor(max_workers=len(section_def["search_queries"])) as search_executor:
            futures = {
                search_executor.submit(_run_search, query): query
                for query in section_def["search_queries"]
            }
            for future in as_completed(futures):
                for chunk in future.result():
                    cid = chunk.get("chunk_id", chunk.get("rank", id(chunk)))
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_chunks.append(chunk)

        # Sort by similarity and take top 15
        all_chunks.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
        context_chunks = all_chunks[:15]

        if not context_chunks:
            return (
                f"*Insufficient data available in the data room to generate this section.*",
                0,
                0.0,
            )

        # 2. Build context text
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", {})
            fn = source.get("file_name", "Unknown")
            page = source.get("page_number")
            loc = f", p.{page}" if page else ""
            context_parts.append(
                f"[Document {i}: {fn}{loc}]\n{chunk['chunk_text']}"
            )
        context_text = "\n\n---\n\n".join(context_parts)

        # 3. Include prior section summaries for coherence
        prior_context = ""
        if prior_sections:
            summaries = []
            for s in SECTIONS:
                if s["key"] in prior_sections:
                    text = prior_sections[s["key"]][:500]
                    summaries.append(f"### {s['label']}\n{text}...")
            prior_context = (
                "\n\nPreviously generated memo sections (for context and coherence):\n\n"
                + "\n\n".join(summaries)
            )

        # 4. Build deal parameters context (for outcomes and financial sections)
        deal_context = ""
        sections_with_deal_params = ["proposed_investment_terms", "outcome_scenario_analysis", "financial_analysis", "investment_recommendation", "valuation_analysis"]
        if deal_params and section_def["key"] in sections_with_deal_params:
            deal_parts = []
            if deal_params.get("ticket_size"):
                deal_parts.append(f"- Investment Amount (Ticket Size): ${deal_params['ticket_size']:,.0f}")
            if deal_params.get("post_money_valuation"):
                deal_parts.append(f"- Post-Money Valuation: ${deal_params['post_money_valuation']:,.0f}")
            if deal_parts:
                deal_context = "\n\n**Deal Parameters (provided by analyst):**\n" + "\n".join(deal_parts) + "\n\nUse these parameters for your analysis and calculations."

        # 5. Build system prompt (dynamic for valuation_analysis)
        system_prompt = section_def['system_prompt']
        if section_def["key"] == "valuation_analysis":
            system_prompt = self._build_valuation_prompt(deal_params)

        # 6. Build prompt
        prompt = f"""{system_prompt}

You are writing an investment memo for a venture capital firm based on data room documents.
Be professional, analytical, and data-driven.
Do NOT include source citations or references to the uploaded data room documents unless the user explicitly asks for them.
If you use any information from outside the uploaded documents (general industry knowledge, market benchmarks, etc.), clearly note it as an external source.
{deal_context}

Context from data room documents:

{context_text}
{prior_context}

Write the "{section_def['label']}" section now."""

        # 7. Call Claude with retry logic (Opus gets more time, fewer retries since sections run in parallel)
        model = self.opus_model if section_def.get("use_opus") else self.default_model
        timeout = 180 if section_def.get("use_opus") else 90

        start = time.time()
        content, input_tokens, output_tokens = self._call_claude_with_retry(
            prompt=prompt,
            model=model,
            max_tokens=4096,
            max_retries=3,
            timeout_seconds=timeout
        )
        elapsed = time.time() - start

        # Track Sonnet token usage for rate-limit-aware cooldowns
        if not section_def.get("use_opus"):
            self._sonnet_tracker.record(input_tokens)

        total_tokens = input_tokens + output_tokens

        pricing = PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * pricing["input"] / 1_000_000) + (
            output_tokens * pricing["output"] / 1_000_000
        )

        with self._stats_lock:
            self.total_tokens += total_tokens
            self.total_cost += cost

        logger.debug(f"Section generated in {elapsed:.1f}s, {total_tokens} tokens, ${cost:.4f}")

        return content, total_tokens, cost

    def _build_valuation_prompt(self, deal_params: Optional[Dict[str, Any]] = None) -> str:
        """Build dynamic system prompt for valuation analysis based on selected methods."""

        # Default to all methods if none specified
        methods = deal_params.get("valuation_methods", ["vc_method"]) if deal_params else ["vc_method"]
        if not methods:
            methods = ["vc_method"]

        method_instructions = []

        if "vc_method" in methods:
            method_instructions.append("""
### VC Method
1. Estimate exit value from projected revenue/ARR at exit (5-7 years) with an appropriate exit multiple
2. Work backwards to implied valuation using a target IRR (25-35%)
3. Provide a valuation range based on different exit scenarios
""")

        if "revenue_multiple" in methods:
            method_instructions.append("""
### Revenue Multiple
1. Identify current ARR/Revenue and growth rate
2. Apply comparable SaaS/tech multiples (EV/Revenue), adjusted for growth and margins
3. Calculate low/mid/high valuation range and compare to the proposed valuation
""")

        if "dcf" in methods:
            method_instructions.append("""
### Discounted Cash Flow (DCF)
1. Project free cash flows for 5-7 years based on available financial projections
2. Apply a VC-appropriate discount rate (25-40%) and terminal value
3. Provide a valuation range with key assumption sensitivities
""")

        if "comparables" in methods:
            method_instructions.append("""
### Comparable Analysis
1. Identify comparable public companies, recent private rounds, and M&A transactions
2. Derive valuation multiples and apply to this company
3. Provide a valuation range noting premium/discount factors
""")

        methods_text = "\n".join(method_instructions)

        single_method = len(methods) == 1
        conclusion = (
            "Conclude with your recommended valuation range and whether the proposed terms are reasonable."
            if single_method else
            "Conclude with a summary table comparing valuations from each method, your recommended range, and whether the proposed terms are reasonable."
        )

        return f"""Write the Valuation Analysis section of a VC investment memo.
Perform a valuation analysis using the following method(s):
{methods_text}
For each method, show key assumptions and provide a valuation range (low / mid / high).

{conclusion}
"""

    def _compile_full_memo(self, sections: Dict[str, str]) -> str:
        """Compile all sections into a single markdown document."""
        parts = []
        for section_def in SECTIONS:
            key = section_def["key"]
            if key in sections:
                parts.append(f"## {section_def['label']}\n\n{sections[key]}")
        return "\n\n---\n\n".join(parts)
