# Noesis AI - Testing Guide

Complete guide to test your VC Due Diligence Assistant.

## Prerequisites

Before testing, ensure you have:

1. **Python virtual environment activated**
2. **API keys configured** in `.env` file
3. **Database initialized**
4. **Sample documents** ready for testing

---

## Step 1: Environment Setup

### Activate Virtual Environment

```bash
cd /Users/ziyadel-hamzawi/Desktop/Noesis\ AI
source venv/bin/activate
```

### Verify API Keys

```bash
# Check that .env file exists and has keys
cat .env

# Should show:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

If `.env` doesn't exist:

```bash
cp .env.example .env
# Then edit .env and add your API keys
```

### Initialize Database

```bash
python tools/init_database.py
```

You should see:
```
Created directory: .tmp
Created directory: .tmp/chroma_db
Created directory: .tmp/data_rooms
Created directory: .tmp/logs
Created table: data_rooms
Created table: documents
...
Database initialized successfully
```

---

## Step 2: Test Backend API

### Start the FastAPI Server

```bash
python app/main.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Test Health Check

Open a new terminal and run:

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "database": true,
  "vector_db": true,
  "api_keys_configured": true
}
```

### Test API Documentation

Open in browser: **http://localhost:8000/docs**

You should see the interactive API documentation (Swagger UI).

---

## Step 3: Test CLI Tools (Recommended First)

Before testing the full web app, test individual tools to verify they work.

### 3.1 Test PDF Parsing

Create a test PDF or use an existing one:

```bash
# Test parsing a PDF
python tools/parse_pdf.py path/to/sample.pdf
```

Expected output:
```
Parsing PDF: sample.pdf
Extracted 5 pages
Total characters: 4523
Metadata: {'page_count': 5, 'file_size': 234567, ...}
```

### 3.2 Test Complete Pipeline (CLI)

Use the test CLI to process a single file:

```bash
python test_cli.py path/to/sample.pdf "What is this document about?"
```

This will:
1. Parse the PDF
2. Chunk the text
3. Generate embeddings
4. Index to vector DB
5. Answer your question

Expected output:
```
Parsing document...
Chunking text...
Generating embeddings...
Indexing to vector database...
Searching for relevant chunks...
Generating answer...

Question: What is this document about?

Answer: This document is about...

Sources:
- sample.pdf, Page 1: "..."
```

---

## Step 4: Test Multi-file Data Room (CLI)

Test the complete ingestion pipeline with multiple files.

### Prepare Test Files

Create a test directory with 2-3 sample files:

```bash
mkdir -p test_data_room
# Copy some PDF/Excel files into test_data_room/
```

### Run Ingestion Tool

```bash
python tools/ingest_data_room.py dr_test_001 test_data_room/*.pdf
```

Expected output:
```
Starting data room ingestion: dr_test_001
Files to process: 3
Step 1/4: Parsing documents...
Parsing [1/3]: document1.pdf
Chunking document1.pdf...
Parsed document1.pdf: 15 chunks
...
Step 2/4: Generating embeddings...
Generated 45 embeddings
Step 3/4: Indexing to vector database...
Indexed to collection: data_room_dr_test_001
Step 4/4: Saving chunks to database...
Saved 45 chunks to database

============================================================
DATA ROOM INGESTION SUMMARY
============================================================
Data Room ID:    dr_test_001
Total Files:     3
Parsed Files:    3
Failed Files:    0
Total Chunks:    45
Duration:        23.45s
Collection:      data_room_dr_test_001
============================================================
```

### Test Semantic Search

```bash
python tools/semantic_search.py "revenue model" --data-room-id dr_test_001
```

Expected output:
```
Searching for: "revenue model"
Data room: dr_test_001

Top 5 Results:
1. [Score: 0.89] document1.pdf, Page 3
   "The revenue model is based on SaaS subscriptions..."

2. [Score: 0.82] document2.pdf, Page 7
   "Monthly recurring revenue (MRR) has grown..."
...
```

### Test Q&A

```bash
python tools/answer_question.py "What is the burn rate?" --data-room-id dr_test_001
```

Expected output:
```
Question: What is the burn rate?

Answer: Based on the financial documents, the monthly burn rate is approximately $180K...

Sources:
- financials.xlsx, Sheet: Cash Flow
- pitch_deck.pdf, Page 15

Confidence: 0.87
Cost: $0.023
Response time: 2.3s
```

---

## Step 5: Test Web UI

Now test the complete web application.

### Start Frontend

Open a new terminal:

```bash
cd frontend
npm install  # First time only
npm run dev
```

Expected output:
```
  VITE v5.0.7  ready in 234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

### Open Browser

Navigate to: **http://localhost:3000**

---

## Step 6: Web UI Testing Checklist

### Test 1: View Data Rooms Page

**Expected:**
- See "Data Rooms" header
- If you ran CLI tests, see `dr_test_001` in the list
- If empty, see "No data rooms yet" message
- "New Data Room" button visible

**Actions:**
- Click on any existing data room → should navigate to chat
- Click "New Data Room" → should navigate to upload page

---

### Test 2: Upload a Data Room

1. **Click "Upload" in navigation** or "New Data Room" button

**Expected:**
- Upload form with fields:
  - Company Name (required)
  - Analyst Name (required)
  - Analyst Email (optional)
  - File dropzone

2. **Fill in the form:**
   - Company Name: "Test Startup Inc"
   - Analyst Name: "Your Name"
   - Analyst Email: "you@example.com"

3. **Drag and drop 2-3 files** (PDF, Excel, CSV)
   - Or click dropzone to browse

**Expected:**
- Files appear in the list with:
  - File icon
  - File name
  - File size
  - Remove (X) button

4. **Click "Create Data Room"**

**Expected:**
- Button shows "Uploading..." with loading spinner
- Redirects to chat page
- Status banner shows "Uploading files..." → "Parsing documents..."

---

### Test 3: Watch Processing

**Expected in Chat Interface:**

1. **Status Banner** at top showing:
   - Animated spinner (while processing)
   - Status: "Parsing documents..." → "Indexing to vector DB..."
   - Progress bar (0% → 100%)
   - Document count: "2/3 documents"

2. **Progress should update automatically** every few seconds

3. **When complete:**
   - Status badge turns green: "Ready"
   - No spinner
   - Progress bar at 100%
   - Input field becomes enabled

**Typical processing timeline:**
- 2-3 small PDFs: ~30-60 seconds
- 5-10 documents: 2-3 minutes
- 20+ documents: 5-10 minutes

---

### Test 4: Ask Questions

Once status shows "Ready":

1. **Type a question** in the input field:
   - "What is this company about?"
   - "What is their revenue model?"
   - "Who are the founders?"

2. **Click Send** (or press Enter)

**Expected:**
- Input clears
- "Generating answer..." spinner appears
- After 2-5 seconds, answer appears with:
  - **Question bubble** (blue, right-aligned)
  - **Answer bubble** (gray, left-aligned)
  - **Markdown formatting** (bold, bullets, etc.)
  - **Sources section** below answer showing:
    - Document name
    - Page number (if PDF)
    - Relevant excerpt
  - **Metadata** (confidence, response time, cost)

3. **Test multiple questions:**
   - Ask 3-5 different questions
   - Each should appear in chat history
   - Scroll should auto-scroll to latest

**Example Questions to Test:**
```
Simple factual: "What is the company name?"
Financial: "What is the monthly recurring revenue?"
Team: "Who are the founders?"
Market: "What is their target market?"
Complex: "What are the main risks facing this company?"
```

---

### Test 5: Navigate Between Pages

1. **Click "Data Rooms" in navigation**

**Expected:**
- Returns to data room list
- Your newly created data room appears
- Shows:
  - Company name
  - Status badge (Complete)
  - Analyst name
  - Document count
  - Created date
  - Cost (if any)

2. **Click "Open Chat" button**

**Expected:**
- Returns to chat interface
- Previous questions/answers are still there (loaded from history)

---

## Step 7: Verify Data Persistence

### Check Database

```bash
# View data rooms
sqlite3 .tmp/due_diligence.db "SELECT id, company_name, processing_status, total_documents FROM data_rooms;"
```

**Expected output:**
```
dr_test_001|Test Startup Inc|complete|3
dr_abc123xyz|Another Company|complete|5
```

### Check Vector Database

```bash
curl http://localhost:8000/api/collections
```

**Expected response:**
```json
{
  "collections": [
    {
      "name": "data_room_dr_test_001",
      "count": 45,
      "metadata": {}
    }
  ],
  "total": 1
}
```

### Check Query History

```bash
curl http://localhost:8000/api/data-room/dr_test_001/questions
```

**Expected response:**
```json
{
  "data_room_id": "dr_test_001",
  "questions": [
    {
      "id": "query_...",
      "question": "What is the revenue model?",
      "answer": "The revenue model is...",
      "sources": [...],
      "created_at": "2026-01-25T...",
      "cost": 0.023
    }
  ],
  "total": 1
}
```

### Check API Costs

```bash
curl http://localhost:8000/api/costs
```

**Expected response:**
```json
{
  "period_days": 30,
  "total_cost": 2.45,
  "total_input_tokens": 12500,
  "total_output_tokens": 3400,
  "total_calls": 8,
  "by_provider": [
    {
      "provider": "openai",
      "cost": 0.15,
      "calls": 3
    },
    {
      "provider": "anthropic",
      "cost": 2.30,
      "calls": 5
    }
  ]
}
```

---

## Step 8: Test Error Handling

### Test Invalid File Upload

1. Try uploading a .txt or .docx file (unsupported)

**Expected:**
- File should be rejected in dropzone, OR
- Upload succeeds but parsing fails with error message

### Test Empty Question

1. Try submitting an empty question

**Expected:**
- Button should be disabled
- No API call made

### Test Question Before Processing Complete

1. Upload a data room
2. Try asking a question while status is "Parsing..."

**Expected:**
- Input should be disabled
- Error message: "Please wait for data room processing to complete"

### Test Non-existent Data Room

```bash
curl http://localhost:8000/api/data-room/fake_id_999/status
```

**Expected:**
```json
{
  "detail": "Data room not found: fake_id_999"
}
```

---

## Step 9: Performance & Cost Testing

### Track Processing Time

For a 5-document data room:

```bash
time python tools/ingest_data_room.py dr_perf_test test_data_room/*.pdf
```

**Expected:**
- Small (5 docs, ~50 pages): 30-90 seconds
- Medium (15 docs, ~150 pages): 2-4 minutes
- Large (30 docs, ~300 pages): 5-10 minutes

### Verify Cost Estimates

Check actual costs match estimates:

```bash
curl http://localhost:8000/api/costs?data_room_id=dr_test_001
```

**Expected for small data room (~5 docs):**
- Embedding cost: ~$0.10-0.20 (OpenAI)
- Q&A cost: ~$0.02-0.05 per question (Claude)
- **Total per data room: $2-3**

---

## Step 10: Browser Console Testing

### Open Browser DevTools

In browser (F12), check Console tab:

**Expected:**
- No red errors
- Network requests should succeed (200 status)
- API calls should complete in reasonable time (<5s for Q&A)

### Check Network Tab

1. Ask a question
2. Watch Network tab

**Expected:**
- POST to `/api/data-room/{id}/question`
- Status: 200 OK
- Response time: 2-5 seconds
- Response body contains answer and sources

---

## Common Issues & Solutions

### Issue: "API keys not configured"

**Solution:**
```bash
# Check .env file exists
cat .env

# If missing, copy example and add keys
cp .env.example .env
nano .env  # Add your API keys
```

### Issue: "Database not found"

**Solution:**
```bash
python tools/init_database.py
```

### Issue: Frontend can't connect to API

**Solution:**
- Check backend is running on port 8000
- Check frontend proxy in `vite.config.ts`
- Try: `curl http://localhost:8000/api/health`

### Issue: "Module not found" errors

**Solution:**
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Issue: Parsing fails for PDF

**Solution:**
- Check PDF is not encrypted
- Check PDF is not scanned image (OCR not implemented yet)
- Try a different PDF

### Issue: Embeddings fail

**Solution:**
- Check OpenAI API key is valid
- Check you have credits in OpenAI account
- Check internet connection

### Issue: Q&A returns poor results

**Solution:**
- Check that documents actually contain relevant information
- Try more specific questions
- Check chunk count (should be >10 chunks for meaningful search)

---

## Success Criteria

Your system is working correctly if:

✅ **Backend:**
- Health check returns "healthy"
- Database has tables created
- API endpoints respond without errors

✅ **File Processing:**
- PDFs parse successfully
- Chunks are generated (15-30 per document typically)
- Embeddings are created
- Vector DB collection is created

✅ **Q&A:**
- Questions get relevant answers
- All answers have source citations
- Citations reference actual documents/pages
- Response time <5 seconds
- Cost tracking works

✅ **Web UI:**
- Upload works with multiple files
- Processing status updates in real-time
- Chat interface shows Q&A history
- Navigation between pages works
- No console errors

✅ **Data Persistence:**
- Data rooms are saved to database
- Questions are saved to history
- Reloading page preserves data

---

## Next Steps After Testing

Once testing is complete:

1. **Clean up test data** (optional):
   ```bash
   rm -rf .tmp/data_rooms/dr_test_*
   ```

2. **Ready for real data rooms:**
   - Upload actual VC data rooms
   - Test with real pitch decks and financial models

3. **Move to Milestone 3:**
   - Structured data extraction (financials, team, market)
   - Data dashboard
   - Export functionality

---

## Quick Test Script

For quick verification, run this script:

```bash
#!/bin/bash
echo "Testing Noesis AI..."

# 1. Health check
echo "1. Testing API health..."
curl -s http://localhost:8000/api/health | grep "healthy" && echo "✅ API healthy" || echo "❌ API unhealthy"

# 2. List data rooms
echo "2. Listing data rooms..."
curl -s http://localhost:8000/api/data-rooms | grep "data_rooms" && echo "✅ Can list data rooms" || echo "❌ Cannot list data rooms"

# 3. Check vector DB
echo "3. Checking vector database..."
curl -s http://localhost:8000/api/collections | grep "collections" && echo "✅ Vector DB accessible" || echo "❌ Vector DB not accessible"

# 4. Check costs
echo "4. Checking cost tracking..."
curl -s http://localhost:8000/api/costs | grep "total_cost" && echo "✅ Cost tracking works" || echo "❌ Cost tracking failed"

echo ""
echo "Testing complete!"
```

Save as `test_quick.sh`, make executable (`chmod +x test_quick.sh`), and run (`./test_quick.sh`).
