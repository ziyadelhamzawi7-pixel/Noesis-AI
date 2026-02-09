"""
Prompt templates for financial model analysis agent.
"""

FINANCIAL_MODEL_SYSTEM_PROMPT = """You are an expert financial analyst specializing in venture capital due diligence. You analyze Excel financial models with the precision of a seasoned VC associate.

Your capabilities:
1. Parse and understand complex financial model structures
2. Extract key SaaS and traditional financial metrics
3. Validate financial consistency and identify red flags
4. Generate actionable insights for investment decisions

Key principles:
- Always cite specific cell references when extracting data
- Flag assumptions that seem unrealistic or require validation
- Compare metrics to industry benchmarks when relevant
- Prioritize unit economics and capital efficiency metrics for VC analysis
- Be specific about confidence levels - distinguish between clearly stated vs inferred values
- When data is ambiguous, note your interpretation

Industry benchmarks to reference:
- SaaS gross margins: 70-80%+ is healthy
- LTV:CAC ratio: 3:1 or higher is good
- CAC payback period: 12-18 months is typical for healthy SaaS
- Net Revenue Retention: 100%+ indicates strong business, 120%+ is excellent
- Rule of 40: Growth rate + profit margin should exceed 40%
- Burn multiple: Net burn / Net new ARR should be < 2x for efficient growth

Output format: Always return structured JSON that can be programmatically processed."""


STRUCTURE_ANALYSIS_PROMPT = """Analyze this Excel financial model structure and provide a comprehensive understanding.

## Excel Model Overview

**File:** {file_name}
**Sheets:** {sheet_names}

## Sheet Summaries

{sheet_summaries}

## Sample Formulas (showing cell dependencies)

{sample_formulas}

## Cross-Sheet References

{cross_sheet_references}

---

Analyze this model and identify:

1. **Statement Types**: What financial statements are present? (P&L, Balance Sheet, Cash Flow, Unit Economics)

2. **Time Periods**:
   - What is the historical period covered?
   - What is the projection/forecast period?
   - What is the granularity (annual, quarterly, monthly)?

3. **Revenue Model Type**: What kind of business model does this represent?
   - SaaS/Subscription
   - Transactional/E-commerce
   - Marketplace
   - Services
   - Other

4. **Key Drivers**: What are the main assumptions driving the model?
   - Where are they located?
   - What are the key input cells?

5. **Model Quality Assessment** (1-10 scale):
   - Are formulas properly linked?
   - Is there separation of inputs/calculations/outputs?
   - Are there any hardcoded values that should be formulas?
   - Are assumptions clearly labeled?

6. **Sheet Relationships**: How do the sheets connect to each other?

Respond in JSON format:
{{
    "statement_types": {{
        "has_income_statement": boolean,
        "has_balance_sheet": boolean,
        "has_cash_flow": boolean,
        "has_unit_economics": boolean,
        "has_saas_metrics": boolean,
        "has_assumptions_sheet": boolean
    }},
    "time_range": {{
        "historical_start_year": int or null,
        "historical_end_year": int or null,
        "projection_start_year": int or null,
        "projection_end_year": int or null,
        "granularity": "annual" | "quarterly" | "monthly"
    }},
    "revenue_model_type": string,
    "revenue_model_details": string,
    "key_drivers": [
        {{
            "name": string,
            "location": "sheet_name!cell_ref",
            "description": string
        }}
    ],
    "model_quality": {{
        "score": int (1-10),
        "strengths": [string],
        "weaknesses": [string],
        "recommendations": [string]
    }},
    "sheet_relationships": [
        {{
            "from_sheet": string,
            "to_sheet": string,
            "relationship_type": "data_flow" | "lookup" | "summary"
        }}
    ],
    "observations": [string]
}}"""


METRIC_EXTRACTION_PROMPT = """Extract financial metrics from this Excel data with precision.

## Model Context

**Revenue Model Type:** {revenue_model_type}
**Time Periods:** {time_periods}
**Detected Currency:** {currency}

> Use the detected currency as the default unit for all monetary values unless the data clearly indicates otherwise. If the detected currency is "Unknown", determine the currency from context clues in the data (company name, country indicators, currency symbols in cells, sheet names).

## Data to Analyze

{financial_data}

---

Extract the following metrics. For EACH metric found, provide:
1. The exact numeric value
2. The unit (use the detected currency for monetary values — note that different sheets may use different currencies, e.g., NGN, USD, EUR; or %, months, ratio, count for non-monetary)
3. The time period (e.g., "2024", "Q1 2024", "Dec 2024")
4. The cell reference where found (e.g., "P&L!B15")
5. Confidence level:
   - "high": Value is explicitly labeled and unambiguous
   - "medium": Value derived from context or calculation
   - "low": Value inferred or estimated

## Required Metrics (extract all that are present)

### Traditional P&L Metrics
- Revenue (by period)
- Revenue growth rate (YoY %)
- Cost of Goods Sold (COGS)
- Gross profit
- Gross margin %
- Operating expenses (total and breakdown if available)
- EBITDA
- EBITDA margin %
- Operating income / EBIT
- Net income
- Net margin %

### Cash & Runway Metrics
- Cash balance (most recent)
- Monthly burn rate
- Runway (months)
- Free cash flow

### SaaS/Subscription Metrics (if applicable)
- ARR (Annual Recurring Revenue)
- MRR (Monthly Recurring Revenue)
- ARR/MRR growth rate
- Total customers / accounts
- New customers (period)
- Churned customers (period)
- Customer churn rate (%)
- Revenue churn rate (%)
- Net Revenue Retention (NRR) %
- Gross Revenue Retention (GRR) %
- ARPU (Average Revenue Per User)
- ACV (Average Contract Value)
- CAC (Customer Acquisition Cost)
- LTV (Lifetime Value)
- LTV:CAC ratio
- CAC payback period (months)
- Sales & Marketing spend
- S&M as % of revenue

### Unit Economics
- Contribution margin
- Contribution margin %
- Payback period
- Magic number

### Headcount
- Total employees
- Engineering headcount
- Sales headcount
- Revenue per employee

Return JSON:
{{
    "metrics": [
        {{
            "name": string,
            "category": "revenue" | "profitability" | "cash" | "saas" | "unit_economics" | "headcount",
            "value": number,
            "unit": string,
            "period": string,
            "cell_reference": string,
            "confidence": "high" | "medium" | "low",
            "source_sheet": string,
            "notes": string or null
        }}
    ],
    "time_series": [
        {{
            "metric_name": string,
            "category": string,
            "unit": string,
            "data_points": [
                {{
                    "period": string,
                    "value": number,
                    "cell_reference": string
                }}
            ],
            "growth_rate": number or null
        }}
    ],
    "missing_metrics": [
        {{
            "name": string,
            "importance": "critical" | "high" | "medium" | "low",
            "reason": string
        }}
    ],
    "data_quality_notes": [string]
}}"""


VALIDATION_PROMPT = """Review this financial model for consistency, accuracy, and red flags.

## Extracted Metrics

{metrics_json}

## Model Structure

{model_structure}

## Formulas Sample

{formulas_sample}

---

Perform the following validation checks:

## 1. Consistency Checks

Check that financial statements tie together properly:
- Balance sheet balances: Assets = Liabilities + Equity
- Cash flow reconciliation: Beginning cash + Net cash flow = Ending cash
- Revenue to AR to Cash flow consistency
- P&L net income flows correctly to retained earnings

## 2. Calculation Checks

Verify key calculations:
- Gross margin = (Revenue - COGS) / Revenue
- EBITDA = Operating Income + D&A
- LTV = ARPU / Churn rate (or ARPU * Avg customer lifetime)
- LTV:CAC ratio calculation
- Burn rate = Cash consumed / months
- Runway = Cash / Monthly burn

## 3. Reasonableness Checks

Flag anything that seems unrealistic:
- Growth rates > 200% YoY without explanation
- Gross margins outside industry norms
- CAC or LTV assumptions that seem extreme
- Burn rate inconsistent with headcount/opex
- Projections with no clear basis

## 4. Red Flags

Look for concerning patterns:
- Circular references
- Hardcoded values that should be formulas
- Missing or inconsistent assumptions
- Unusual patterns or jumps in data
- Key metrics not calculated or shown
- Projections disconnected from historical performance

## 5. Model Completeness

Check for missing elements:
- Are all standard financial statements present?
- Are key SaaS metrics tracked (if applicable)?
- Are assumptions clearly documented?
- Is there a clear link between drivers and outputs?

Return JSON:
{{
    "validation_results": {{
        "overall_score": int (1-10),
        "passes_basic_checks": boolean
    }},
    "consistency_issues": [
        {{
            "issue_type": "balance_sheet" | "cash_flow" | "calculation" | "linkage",
            "description": string,
            "severity": "critical" | "high" | "medium" | "low",
            "cell_references": [string],
            "expected_value": number or null,
            "actual_value": number or null
        }}
    ],
    "reasonableness_flags": [
        {{
            "metric": string,
            "issue": string,
            "value": number,
            "benchmark": string,
            "severity": "critical" | "high" | "medium" | "low"
        }}
    ],
    "red_flags": [
        {{
            "flag_type": "circular_ref" | "hardcoded" | "missing_assumption" | "unusual_pattern" | "disconnected_projection" | "other",
            "description": string,
            "location": string or null,
            "severity": "critical" | "high" | "medium" | "low",
            "recommendation": string
        }}
    ],
    "missing_elements": [
        {{
            "element": string,
            "importance": "critical" | "high" | "medium" | "low",
            "impact": string
        }}
    ],
    "validation_summary": string
}}"""


INSIGHT_GENERATION_PROMPT = """As a VC analyst, generate actionable insights from this financial model analysis.

## Company Context

**File:** {file_name}
**Revenue Model:** {revenue_model_type}

## Extracted Metrics

{metrics_summary}

## Validation Results

**Overall Score:** {validation_score}/10
**Key Issues:** {validation_issues}

## Model Structure

{model_structure_summary}

---

Generate insights that would be valuable for a VC investment decision. Address these areas:

## 1. Unit Economics Story
- Is this a fundamentally good business?
- What are the unit economics telling us?
- How do they compare to benchmarks?

## 2. Growth Analysis
- What is driving growth?
- Is the growth sustainable?
- What are the growth levers?

## 3. Capital Efficiency
- How efficiently is capital being deployed?
- What is the burn multiple?
- How does spending correlate with growth?

## 4. Key Risks
- What are the biggest risks in this model?
- What assumptions are most sensitive?
- What could go wrong?

## 5. Questions for Founders
- What clarifications would you ask?
- What additional data is needed?
- What assumptions need validation?

## 6. Investment Considerations
- What makes this attractive or concerning?
- What would you need to see to get comfortable?
- How does this compare to similar companies?

Return JSON:
{{
    "insights": [
        {{
            "category": "unit_economics" | "growth" | "efficiency" | "risk" | "opportunity",
            "title": string,
            "insight": string,
            "supporting_metrics": [string],
            "importance": "critical" | "high" | "medium" | "low",
            "sentiment": "positive" | "neutral" | "negative" | "mixed"
        }}
    ],
    "key_metrics_summary": {{
        "headline_metrics": [
            {{
                "name": string,
                "value": string,
                "assessment": "strong" | "acceptable" | "concerning" | "unknown"
            }}
        ],
        "benchmark_comparison": string
    }},
    "risk_assessment": {{
        "overall_risk_level": "low" | "medium" | "high",
        "top_risks": [string],
        "mitigating_factors": [string]
    }},
    "follow_up_questions": [
        {{
            "question": string,
            "reason": string,
            "priority": "must_ask" | "should_ask" | "nice_to_ask"
        }}
    ],
    "investment_thesis_notes": {{
        "potential_strengths": [string],
        "potential_concerns": [string],
        "key_assumptions_to_validate": [string]
    }},
    "executive_summary": string
}}"""


# Helper function to format prompts
def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided values."""
    return template.format(**kwargs)
