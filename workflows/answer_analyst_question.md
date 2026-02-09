# Workflow: Answer Analyst Question

## Objective

Provide accurate, cited answers to analyst questions about data rooms using RAG (Retrieval Augmented Generation).

## Inputs

- **Question**: Natural language question from analyst
- **Data Room ID**: ID of the data room to query
- **Filters** (Optional): Document type, date range, category filters
- **Conversation History** (Optional): Previous Q&A for context

## Required Tools

1. `tools/semantic_search.py` - Vector similarity search
2. `tools/answer_question.py` - RAG pipeline with Claude
3. `app/database.py` - Save query to history

## Process Steps

### 1. Validate Inputs

**Check:**
- Data room exists in database
- Data room processing is complete (status = "complete")
- Question is not empty
- If failed or still processing, return appropriate error message

**Actions:**
- Query database for data room status
- If not complete: "Data room is still processing. Please wait."
- If failed: "Data room processing failed. Please re-upload."

### 2. Semantic Search

**Goal:** Find relevant document chunks from vector database

**Run:** `semantic_search.py`

**Parameters:**
- query: The analyst's question
- data_room_id: Target data room
- top_k: 15 (retrieve top 15 most relevant chunks)
- filters: Apply any document filters

**Output:** List of relevant chunks with:
- chunk_text
- document_name
- page_number
- relevance_score
- metadata

**Edge Cases:**
- **No results found**: If top relevance score < 0.3
  - Response: "I couldn't find relevant information in the data room to answer that question."
  - Do not hallucinate or make up answers
  - Suggest rephrasing or ask if they meant something else

- **Low confidence results**: If top relevance score between 0.3-0.5
  - Proceed but flag low confidence in response
  - Include confidence score in answer

### 3. Load Cached Data (If Applicable)

**For certain question types, load cached extracted data:**

Question Type | Cached Data to Load
--- | ---
Financial questions (revenue, burn, runway) | `analysis_cache` where `analysis_type = 'financials'`
Team questions (founders, employees) | `analysis_cache` where `analysis_type = 'team'`
Market questions (TAM, competitors) | `analysis_cache` where `analysis_type = 'market'`

**Why:** Structured data extraction is more accurate than RAG for specific data points

**Implementation:**
```python
if any(keyword in question.lower() for keyword in ['revenue', 'burn', 'runway']):
    cached_financials = db.get_analysis_cache(data_room_id, 'financials')
    # Add to context for RAG
```

### 4. Run RAG Pipeline

**Goal:** Generate answer using Claude with retrieved context

**Run:** `answer_question.py`

**Parameters:**
- question: The analyst's question
- chunks: Retrieved chunks from semantic search
- cached_data: Any loaded cached analysis
- conversation_history: Previous Q&A for multi-turn conversations
- model: claude-sonnet-4.5 (balance of quality and cost)

**System Prompt Requirements:**
- Role: "You are a VC analyst assistant"
- Tone: Professional, data-driven, concise
- Citation requirement: All factual claims must cite source documents
- No hallucination: Only use information from provided chunks
- Confidence: Express uncertainty when information is incomplete

**Output:**
- answer: Markdown-formatted answer
- sources: List of cited sources with page numbers
- confidence: 0-1 score
- tokens_used: For cost tracking
- cost: API cost in USD

### 5. Validate Answer Quality

**Quality Checks:**

✅ **Citation Validation**
- Every factual claim has a source citation
- Source citations match actual chunks provided
- No "fake" citations to documents not in the data room

✅ **Answer Completeness**
- Directly addresses the question
- Includes relevant context
- Flags missing information if incomplete

✅ **Professional Tone**
- No informal language
- No hedging unless genuinely uncertain
- Clear, actionable insights

**If quality check fails:**
- Log warning
- Optionally retry with different prompt
- Flag for manual review

### 6. Save to Database

**Save query and answer:**

```python
db.save_query(
    data_room_id=data_room_id,
    question=question,
    answer=answer,
    sources=sources,
    confidence_score=confidence,
    response_time_ms=response_time,
    tokens_used=tokens_used,
    cost=cost
)
```

**Also track API usage:**

```python
db.track_api_usage(
    provider='anthropic',
    model='claude-sonnet-4.5',
    operation='qa',
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    cost=cost,
    data_room_id=data_room_id
)
```

### 7. Return Response

**Response Format:**

```json
{
  "answer": "Based on the pitch deck, the company's primary revenue model is...",
  "sources": [
    {
      "document": "pitch_deck.pdf",
      "page": 12,
      "excerpt": "Revenue model: SaaS subscription with tiered pricing...",
      "relevance": 0.89
    }
  ],
  "confidence": 0.85,
  "tokens_used": 2430,
  "cost": 0.023,
  "response_time_ms": 2340
}
```

## Edge Cases & Error Handling

### Empty Data Room
**Scenario:** Data room has no indexed documents

**Response:** "This data room appears to be empty. Please upload documents first."

### Ambiguous Question
**Scenario:** Question is too vague or has multiple interpretations

**Response:** Ask clarifying question
- "I found information about both monthly and annual revenue. Which would you like to know about?"
- Provide options for the analyst to choose

### Multi-Part Question
**Scenario:** "What is the revenue and burn rate and how long is the runway?"

**Approach:**
- Break into sub-questions
- Answer each part separately with citations
- Use structured format (bullet points or sections)

### Calculation Required
**Scenario:** "What is the monthly burn rate?" when only total expenses and timeframe are in docs

**Approach:**
1. Extract relevant numbers from documents
2. Show calculation explicitly: "Total expenses of $1.2M over 6 months = $200K/month burn rate"
3. Cite source documents for all numbers
4. Flag if assumptions were made

### Contradictory Information
**Scenario:** Different documents have conflicting data (e.g., different revenue figures)

**Approach:**
- Acknowledge the discrepancy
- Present both values with sources
- "The pitch deck (page 5) states $2M ARR, while the financial model (Sheet: Summary) shows $1.8M ARR as of Q3. This may reflect different time periods or updates."

### Out of Scope Question
**Scenario:** Question not related to the data room (e.g., "What's the weather?")

**Response:** "I can only answer questions about the data room you've uploaded. Could you rephrase your question to be about the company's documents?"

### Follow-up Question
**Scenario:** "What about last year?" (requires conversation context)

**Approach:**
- Use conversation_history to understand context
- Reference previous answer: "Following up on the revenue question..."
- Maintain conversation thread with conversation_id

## Quality Standards

All answers must meet these standards:

### ✅ Accuracy
- 100% of factual claims must be sourced from documents
- No hallucinations or invented facts
- Calculations must be shown and verified

### ✅ Citations
- Every claim has a source citation
- Citations include: document name, page number, relevant excerpt
- Citations are clickable/navigable in UI

### ✅ Relevance
- Directly answers the question asked
- Provides appropriate level of detail
- Includes context when helpful

### ✅ Professional Tone
- VC industry-appropriate language
- Data-driven, analytical perspective
- Balanced view (pros and cons when applicable)

### ✅ Clarity
- Well-structured (paragraphs, bullets, sections as appropriate)
- Markdown formatting for readability
- Key takeaways highlighted

## Performance Targets

- **Response Time:** <5 seconds for most questions
- **Relevance:** >80% of top-5 chunks should be actually relevant
- **Confidence:** Average confidence score >0.7
- **Cost:** <$0.05 per question on average

## Success Metrics

Track these metrics over time:

1. **Answer Quality Score** (analyst feedback)
2. **Citation Accuracy** (all citations valid)
3. **Response Time** (median and p95)
4. **Cost per Query**
5. **Retry Rate** (questions asked >1 time for same info)

## Example Scenarios

### Example 1: Simple Factual Question

**Question:** "What is the company's monthly recurring revenue?"

**Process:**
1. Search for chunks mentioning "MRR", "monthly recurring revenue", "revenue"
2. Find in financial_model.xlsx: "$450K MRR as of October 2024"
3. Generate answer with citation
4. Save to database

**Answer:**
"The company's monthly recurring revenue (MRR) is **$450K** as of October 2024.

**Source:** financial_model.xlsx, Sheet: Dashboard"

---

### Example 2: Complex Analysis Question

**Question:** "Is the burn rate sustainable given the current runway?"

**Process:**
1. Search for financial metrics
2. Load cached financials if available
3. Extract: burn rate, cash balance, runway
4. Generate analytical answer

**Answer:**
"Based on the financial data:

- **Monthly burn rate:** $180K
- **Current cash balance:** $2.4M
- **Estimated runway:** 13.3 months (until November 2025)

This runway is **concerning but not critical**. The company has enough time to reach their next milestone (Series A target of $3M ARR by Q2 2025), but there's limited buffer for delays or unexpected expenses.

**Recommendation:** Monitor burn closely and consider extending runway through:
- Revenue acceleration (currently at $80K MRR)
- Cost optimization
- Bridge funding if milestones are at risk

**Sources:**
- financial_model.xlsx, Sheet: Cash Flow (burn rate calculation)
- pitch_deck.pdf, Page 15 (bank balance as of Dec 1)
- board_deck_oct.pdf, Page 8 (Series A targets)"

---

### Example 3: Handling Missing Information

**Question:** "What is the customer acquisition cost (CAC)?"

**Process:**
1. Search for CAC, customer acquisition, marketing spend
2. No direct mention found
3. Check if can be calculated from available data
4. If not, be transparent

**Answer:**
"I couldn't find a specific CAC (Customer Acquisition Cost) metric in the data room.

However, I found related information that might be helpful:
- **Monthly marketing spend:** $45K (financial_model.xlsx)
- **New customers last month:** 23 (metrics_dashboard.xlsx)
- **Implied CAC:** ~$1,956 per customer (if all marketing spend attributed to acquisition)

**Note:** This is a rough calculation and may not reflect the actual CAC if marketing spend includes other activities (brand, retention, etc.).

Would you like me to look for more detailed customer acquisition data?"

---

## Workflow Diagram

```
┌─────────────────────┐
│ Analyst Question    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate Input      │
│ - Data room exists? │
│ - Processing done?  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Semantic Search     │
│ - Query vector DB   │
│ - Get top-15 chunks │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Load Cached Data    │
│ (if applicable)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Run RAG Pipeline    │
│ - Claude API call   │
│ - Generate answer   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate Quality    │
│ - Check citations   │
│ - Verify accuracy   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Save to Database    │
│ - queries table     │
│ - api_usage table   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Return Response     │
│ - Answer + sources  │
└─────────────────────┘
```

## Continuous Improvement

**Learn from failures:**
- Log questions that fail quality checks
- Analyze low-confidence answers
- Review questions that analysts rephrase multiple times

**Update workflow when you discover:**
- Better search strategies
- Common question patterns
- New edge cases
- Optimization opportunities

**Stay pragmatic, stay reliable, keep learning.**
