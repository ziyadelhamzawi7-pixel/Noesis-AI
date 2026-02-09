"""
Financial Analysis Agent for Excel models.
Uses Claude to analyze financial models, extract metrics, validate consistency,
and generate insights for VC due diligence.
"""

import sys
import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from anthropic import Anthropic
    from dotenv import load_dotenv
except ImportError:
    logger.error("Required packages not installed. Run: pip install anthropic python-dotenv")
    sys.exit(1)

# Handle imports
try:
    from parse_excel_financial import parse_excel_financial
    from prompts.financial_analysis import (
        FINANCIAL_MODEL_SYSTEM_PROMPT,
        STRUCTURE_ANALYSIS_PROMPT,
        METRIC_EXTRACTION_PROMPT,
        VALIDATION_PROMPT,
        INSIGHT_GENERATION_PROMPT,
        format_prompt
    )
except ImportError:
    from tools.parse_excel_financial import parse_excel_financial
    from tools.prompts.financial_analysis import (
        FINANCIAL_MODEL_SYSTEM_PROMPT,
        STRUCTURE_ANALYSIS_PROMPT,
        METRIC_EXTRACTION_PROMPT,
        VALIDATION_PROMPT,
        INSIGHT_GENERATION_PROMPT,
        format_prompt
    )

# Load environment variables
load_dotenv()


@dataclass
class AnalysisCost:
    """Track costs for analysis."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, model: str):
        """Add tokens and calculate cost."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        # Pricing per 1M tokens
        pricing = {
            "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00}
        }

        rates = pricing.get(model, {"input": 3.00, "output": 15.00})
        cost = (input_tokens * rates["input"] / 1_000_000) + \
               (output_tokens * rates["output"] / 1_000_000)
        self.total_cost += cost


class FinancialAnalysisAgent:
    """
    AI agent specialized in analyzing Excel financial models.
    Uses Claude to understand structure, extract metrics, and generate insights.
    """

    def __init__(
        self,
        analysis_model: str = "claude-opus-4-5-20251101",
        extraction_model: str = os.getenv("FINANCIAL_EXTRACTION_MODEL", "claude-sonnet-4-20250514"),
        anthropic_api_key: Optional[str] = None,
        max_cost: float = 5.0
    ):
        """
        Initialize the financial analysis agent.

        Args:
            analysis_model: Model for deep analysis (structure, insights)
            extraction_model: Model for metric extraction and validation
            anthropic_api_key: API key (or from env)
            max_cost: Maximum cost per analysis in USD
        """
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")

        self.client = Anthropic(api_key=api_key)
        self.analysis_model = analysis_model
        self.extraction_model = extraction_model
        self.max_cost = max_cost
        self.cost = AnalysisCost()

        logger.info(f"FinancialAnalysisAgent initialized with analysis_model={analysis_model}")

    def analyze(
        self,
        file_path: str,
        data_room_id: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Perform complete financial model analysis.

        Args:
            file_path: Path to Excel file
            data_room_id: Data room ID
            document_id: Document ID

        Returns:
            Complete analysis results
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())

        logger.info(f"Starting financial analysis for {file_path}")

        result = {
            "analysis_id": analysis_id,
            "data_room_id": data_room_id,
            "document_id": document_id,
            "file_name": Path(file_path).name,
            "analysis_timestamp": datetime.now().isoformat(),
            "status": "in_progress",
            "model_structure": {},
            "extracted_metrics": [],
            "time_series": [],
            "validation_results": {},
            "insights": [],
            "follow_up_questions": [],
            "executive_summary": "",
            "analysis_cost": 0.0,
            "tokens_used": 0,
            "processing_time_ms": 0
        }

        try:
            # Step 1: Parse Excel with enhanced parser
            logger.info("Step 1: Parsing Excel file...")
            parsed_excel = parse_excel_financial(file_path)

            if parsed_excel.get("error"):
                raise Exception(f"Failed to parse Excel: {parsed_excel['error']}")

            result["parsed_data_summary"] = {
                "total_sheets": len(parsed_excel["sheets"]),
                "total_rows": parsed_excel["total_rows"],
                "total_formulas": len(parsed_excel.get("all_formulas", {})),
                "sheet_types": {s["sheet_name"]: s["sheet_type"] for s in parsed_excel["sheets"]}
            }

            # Step 2: Analyze model structure
            logger.info("Step 2: Analyzing model structure...")
            structure_result = self._analyze_structure(parsed_excel)
            result["model_structure"] = structure_result

            # Check cost
            if self.cost.total_cost >= self.max_cost:
                raise Exception(f"Cost limit exceeded: ${self.cost.total_cost:.2f} >= ${self.max_cost}")

            # Step 3: Extract metrics
            logger.info("Step 3: Extracting financial metrics...")
            metrics_result = self._extract_metrics(parsed_excel, structure_result)
            result["extracted_metrics"] = metrics_result.get("metrics", [])
            result["time_series"] = metrics_result.get("time_series", [])
            result["missing_metrics"] = metrics_result.get("missing_metrics", [])

            # Check cost
            if self.cost.total_cost >= self.max_cost:
                raise Exception(f"Cost limit exceeded: ${self.cost.total_cost:.2f} >= ${self.max_cost}")

            # Step 4: Validate model
            logger.info("Step 4: Validating financial model...")
            validation_result = self._validate_model(parsed_excel, metrics_result, structure_result)
            result["validation_results"] = validation_result

            # Check cost
            if self.cost.total_cost >= self.max_cost:
                raise Exception(f"Cost limit exceeded: ${self.cost.total_cost:.2f} >= ${self.max_cost}")

            # Step 5: Generate insights
            logger.info("Step 5: Generating insights...")
            insights_result = self._generate_insights(
                parsed_excel,
                structure_result,
                metrics_result,
                validation_result
            )
            result["insights"] = insights_result.get("insights", [])
            result["follow_up_questions"] = insights_result.get("follow_up_questions", [])
            result["executive_summary"] = insights_result.get("executive_summary", "")
            result["key_metrics_summary"] = insights_result.get("key_metrics_summary", {})
            result["risk_assessment"] = insights_result.get("risk_assessment", {})
            result["investment_thesis_notes"] = insights_result.get("investment_thesis_notes", {})

            result["status"] = "complete"

        except Exception as e:
            logger.error(f"Financial analysis failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        # Final stats
        result["analysis_cost"] = round(self.cost.total_cost, 4)
        result["tokens_used"] = self.cost.input_tokens + self.cost.output_tokens
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)

        logger.success(
            f"Analysis complete in {result['processing_time_ms']}ms, "
            f"cost: ${result['analysis_cost']:.4f}"
        )

        return result

    def _call_claude(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = FINANCIAL_MODEL_SYSTEM_PROMPT,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Call Claude API and parse JSON response.

        Args:
            prompt: User prompt
            model: Model to use (defaults to extraction_model)
            system_prompt: System prompt
            max_tokens: Max response tokens

        Returns:
            Parsed JSON response
        """
        model = model or self.extraction_model

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent extraction
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            # Track costs
            self.cost.add(
                response.usage.input_tokens,
                response.usage.output_tokens,
                model
            )

            # Parse response
            response_text = response.content[0].text

            # Try to extract JSON from response
            # Handle cases where JSON is wrapped in markdown code blocks
            if "```json" in response_text:
                json_match = response_text.split("```json")[1].split("```")[0]
                return json.loads(json_match.strip())
            elif "```" in response_text:
                json_match = response_text.split("```")[1].split("```")[0]
                return json.loads(json_match.strip())
            else:
                return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return raw text in a structure
            return {"raw_response": response_text, "parse_error": str(e)}

    def _analyze_structure(self, parsed_excel: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the financial model structure."""

        # Build sheet summaries
        sheet_summaries = []
        for sheet in parsed_excel["sheets"]:
            summary = f"""
### {sheet['sheet_name']} ({sheet['sheet_type']})
- Rows: {sheet['rows']}, Columns: {sheet['columns']}
- Headers: {', '.join(sheet['headers'][:10])}{'...' if len(sheet['headers']) > 10 else ''}
- Row Labels (first 15): {', '.join(sheet['row_labels'][:15])}{'...' if len(sheet['row_labels']) > 15 else ''}
- Formulas: {sheet['formula_count']}
- Time Periods Detected: {len(sheet.get('time_periods', []))}
"""
            if sheet.get("financial_metrics"):
                summary += f"- Pre-detected Metrics: {', '.join(sheet['financial_metrics'].keys())}\n"

            sheet_summaries.append(summary)

        # Sample formulas
        all_formulas = parsed_excel.get("all_formulas", {})
        sample_formulas = list(all_formulas.items())[:20]
        formulas_text = "\n".join([f"- {cell}: {formula}" for cell, formula in sample_formulas])

        # Cross-sheet references
        cross_refs = parsed_excel.get("cross_sheet_references", [])[:15]
        cross_refs_text = "\n".join([
            f"- {ref['source']} references {ref['target_sheet']}!{ref['target_cell']}"
            for ref in cross_refs
        ])

        prompt = format_prompt(
            STRUCTURE_ANALYSIS_PROMPT,
            file_name=parsed_excel["file_name"],
            sheet_names=", ".join(parsed_excel["sheet_names"]),
            sheet_summaries="\n".join(sheet_summaries),
            sample_formulas=formulas_text if formulas_text else "No formulas found",
            cross_sheet_references=cross_refs_text if cross_refs_text else "No cross-sheet references"
        )

        return self._call_claude(prompt, model=self.analysis_model, max_tokens=4096)

    def _extract_metrics(
        self,
        parsed_excel: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract financial metrics from the model."""

        # Build financial data summary for each relevant sheet
        financial_data_parts = []

        for sheet in parsed_excel["sheets"]:
            if sheet["sheet_type"] in ["p_and_l", "balance_sheet", "cash_flow", "saas_metrics", "assumptions"]:
                # Include headers and data
                data_preview = []
                data_preview.append(f"\n### {sheet['sheet_name']} ({sheet['sheet_type']})")
                data_preview.append(f"Headers: {sheet['headers']}")

                # Include row labels with their data
                for i, label in enumerate(sheet["row_labels"][:50]):  # Limit rows
                    if label and i < len(sheet["data"]):
                        row_data = sheet["data"][i]
                        # Only include rows with numeric data
                        numeric_values = [v for v in row_data if isinstance(v, (int, float)) and v is not None]
                        if numeric_values:
                            data_preview.append(f"Row {i+2} '{label}': {row_data[:15]}")  # Limit columns

                financial_data_parts.append("\n".join(data_preview))

        # Get time periods
        time_periods = []
        for sheet in parsed_excel["sheets"]:
            time_periods.extend(sheet.get("time_periods", []))

        time_periods_text = ", ".join(set([
            p.get("period_type", "") + " " + str(p.get("year", ""))
            for p in time_periods if p.get("year")
        ]))

        # Get revenue model type from structure
        revenue_model = structure.get("revenue_model_type", "Unknown")

        detected_currency = structure.get("base_currency") or "Unknown"

        prompt = format_prompt(
            METRIC_EXTRACTION_PROMPT,
            revenue_model_type=revenue_model,
            time_periods=time_periods_text if time_periods_text else "Not clearly detected",
            financial_data="\n".join(financial_data_parts),
            currency=detected_currency
        )

        return self._call_claude(prompt, model=self.extraction_model, max_tokens=8192)

    def _validate_model(
        self,
        parsed_excel: Dict[str, Any],
        metrics: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate financial model for consistency."""

        # Prepare metrics JSON
        metrics_json = json.dumps(metrics.get("metrics", [])[:50], indent=2)

        # Prepare model structure
        model_structure = json.dumps({
            "statement_types": structure.get("statement_types", {}),
            "time_range": structure.get("time_range", {}),
            "model_quality": structure.get("model_quality", {})
        }, indent=2)

        # Sample formulas
        all_formulas = parsed_excel.get("all_formulas", {})
        formulas_sample = dict(list(all_formulas.items())[:30])
        formulas_text = json.dumps(formulas_sample, indent=2)

        prompt = format_prompt(
            VALIDATION_PROMPT,
            metrics_json=metrics_json,
            model_structure=model_structure,
            formulas_sample=formulas_text
        )

        return self._call_claude(prompt, model=self.extraction_model, max_tokens=4096)

    def _generate_insights(
        self,
        parsed_excel: Dict[str, Any],
        structure: Dict[str, Any],
        metrics: Dict[str, Any],
        validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate VC-relevant insights from the analysis."""

        # Prepare metrics summary
        all_metrics = metrics.get("metrics", [])
        metrics_by_category = {}
        for m in all_metrics:
            cat = m.get("category", "other")
            if cat not in metrics_by_category:
                metrics_by_category[cat] = []
            metrics_by_category[cat].append(m)

        metrics_summary = json.dumps(metrics_by_category, indent=2)

        # Validation summary
        validation_score = validation.get("validation_results", {}).get("overall_score", "N/A")
        validation_issues = []
        for issue in validation.get("consistency_issues", [])[:5]:
            validation_issues.append(issue.get("description", ""))
        for flag in validation.get("red_flags", [])[:5]:
            validation_issues.append(flag.get("description", ""))

        # Model structure summary
        model_structure_summary = json.dumps({
            "revenue_model": structure.get("revenue_model_type", "Unknown"),
            "time_range": structure.get("time_range", {}),
            "key_drivers": [d.get("name") for d in structure.get("key_drivers", [])[:5]]
        }, indent=2)

        prompt = format_prompt(
            INSIGHT_GENERATION_PROMPT,
            file_name=parsed_excel["file_name"],
            revenue_model_type=structure.get("revenue_model_type", "Unknown"),
            metrics_summary=metrics_summary,
            validation_score=validation_score,
            validation_issues=", ".join(validation_issues) if validation_issues else "None found",
            model_structure_summary=model_structure_summary
        )

        return self._call_claude(prompt, model=self.analysis_model, max_tokens=6144)


def analyze_financial_model(
    file_path: str,
    data_room_id: str,
    document_id: str,
    analysis_model: str = "claude-opus-4-5-20251101",
    extraction_model: str = os.getenv("FINANCIAL_EXTRACTION_MODEL", "claude-sonnet-4-20250514"),
    max_cost: float = 5.0
) -> Dict[str, Any]:
    """
    Convenience function to analyze a financial model.

    Args:
        file_path: Path to Excel file
        data_room_id: Data room ID
        document_id: Document ID
        analysis_model: Model for deep analysis
        extraction_model: Model for extraction
        max_cost: Maximum cost in USD

    Returns:
        Complete analysis results

    Example:
        >>> result = analyze_financial_model(
        ...     "financials.xlsx",
        ...     "dr_123",
        ...     "doc_456"
        ... )
        >>> print(f"Found {len(result['extracted_metrics'])} metrics")
    """
    agent = FinancialAnalysisAgent(
        analysis_model=analysis_model,
        extraction_model=extraction_model,
        max_cost=max_cost
    )
    return agent.analyze(file_path, data_room_id, document_id)


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Excel financial models")
    parser.add_argument("file", help="Path to Excel file")
    parser.add_argument("--data-room-id", default="test_dr", help="Data room ID")
    parser.add_argument("--document-id", default="test_doc", help="Document ID")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--max-cost", type=float, default=5.0, help="Maximum cost in USD")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Run analysis
    result = analyze_financial_model(
        args.file,
        args.data_room_id,
        args.document_id,
        max_cost=args.max_cost
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info(f"Saved results to {output_path}")
    else:
        print(f"\n{'='*60}")
        print(f"Financial Model Analysis: {result['file_name']}")
        print(f"Status: {result['status']}")
        print(f"{'='*60}\n")

        if result.get("error"):
            print(f"Error: {result['error']}\n")

        print("Model Structure:")
        structure = result.get("model_structure", {})
        print(f"  Revenue Model: {structure.get('revenue_model_type', 'Unknown')}")
        print(f"  Model Quality: {structure.get('model_quality', {}).get('score', 'N/A')}/10")

        print(f"\nExtracted Metrics: {len(result.get('extracted_metrics', []))}")
        for metric in result.get("extracted_metrics", [])[:10]:
            print(f"  - {metric.get('name')}: {metric.get('value')} {metric.get('unit')} ({metric.get('confidence')})")

        print(f"\nValidation Score: {result.get('validation_results', {}).get('validation_results', {}).get('overall_score', 'N/A')}/10")

        print(f"\nInsights: {len(result.get('insights', []))}")
        for insight in result.get("insights", [])[:5]:
            print(f"  - [{insight.get('importance')}] {insight.get('title')}")

        print(f"\nFollow-up Questions: {len(result.get('follow_up_questions', []))}")
        for q in result.get("follow_up_questions", [])[:5]:
            print(f"  - [{q.get('priority')}] {q.get('question')}")

        if result.get("executive_summary"):
            print(f"\nExecutive Summary:")
            print(f"  {result['executive_summary'][:500]}...")

        print(f"\n{'='*60}")
        print(f"Cost: ${result['analysis_cost']:.4f}")
        print(f"Tokens: {result['tokens_used']:,}")
        print(f"Time: {result['processing_time_ms']}ms")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
