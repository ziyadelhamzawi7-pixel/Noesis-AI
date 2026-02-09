# Noesis AI - VC Due Diligence Assistant

AI-powered tool for venture capital analysts to analyze startup data rooms, extract insights, and generate investment memos.

**Built on the WAT Framework** (Workflows, Agents, Tools) - A reliable architecture for AI-driven automation that separates probabilistic reasoning from deterministic execution.

## Overview

This tool helps VC analysts by:
- 📄 **Parsing data rooms** - PDFs, Excel, Word documents
- 💬 **Answering questions** - RAG-powered Q&A with source citations
- 📊 **Extracting structured data** - Financials, team info, market analysis
- 📝 **Generating investment memos** - Industry-standard VC memos with AI

The WAT framework is built on a simple principle: AI handles intelligent coordination while deterministic code handles execution. This separation dramatically improves reliability compared to having AI attempt every step directly.

## Architecture

### Layer 1: Workflows (The Instructions)
- Markdown SOPs stored in `workflows/`
- Define objectives, required inputs, tool usage, expected outputs, and edge cases
- Written in plain language, like briefing a team member

### Layer 2: Agents (The Decision-Maker)
- AI coordinates workflow execution
- Reads workflows, runs tools in sequence, handles failures, asks clarifying questions
- Connects intent to execution without doing everything directly

### Layer 3: Tools (The Execution)
- Python scripts in `tools/` that perform actual work
- Handle API calls, data transformations, file operations, database queries
- Consistent, testable, and fast

## Project Structure

```
.
├── app/                    # FastAPI web application
│   ├── main.py            # API endpoints
│   ├── models.py          # Pydantic models
│   ├── config.py          # Settings
│   └── prompts/           # Prompt templates (future)
│
├── tools/                 # Python execution scripts
│   ├── init_database.py   # Database initialization
│   ├── parse_pdf.py       # PDF parsing
│   ├── parse_excel.py     # Excel parsing
│   ├── chunk_documents.py # Text chunking
│   ├── generate_embeddings.py  # OpenAI embeddings
│   ├── index_to_vectordb.py    # ChromaDB indexing
│   ├── semantic_search.py      # Vector search
│   └── answer_question.py      # RAG Q&A
│
├── workflows/             # Markdown SOPs (to be created)
│
├── .tmp/                  # Temporary files
│   ├── data_rooms/        # Uploaded data rooms
│   ├── chroma_db/         # Vector database
│   ├── due_diligence.db   # SQLite database
│   └── logs/              # Processing logs
│
├── requirements.txt       # Python dependencies
├── .env                   # API keys (create from .env.example)
├── setup.sh              # Setup automation script
└── test_cli.py           # Simple test CLI
```

## Features Status

### ✅ Milestone 1: Foundation (Complete)
- [x] Database initialization
- [x] PDF parsing (PyPDF2 + pdfplumber)
- [x] Excel parsing (pandas + openpyxl)
- [x] Text chunking with semantic boundaries
- [x] Embedding generation (OpenAI)
- [x] Vector database indexing (ChromaDB)
- [x] Semantic search
- [x] RAG-powered Q&A with Claude
- [x] FastAPI skeleton with basic endpoints

### ✅ Milestone 2: Web UI & Multi-file (Complete)
- [x] Multi-file data room upload
- [x] Background processing with progress tracking
- [x] React frontend with TypeScript
- [x] Chat interface for Q&A
- [x] Document list view
- [x] Real-time status updates
- [x] Source citation display
- [x] Database integration layer

### 🚧 Milestone 3: Structured Data Extraction (Next - Weeks 5-6)
- [ ] Financial metrics extraction
- [ ] Team information extraction
- [ ] Market analysis extraction
- [ ] Data visualization dashboard

### 📋 Milestone 4: Investment Memo Generation (Weeks 7-8)
- [ ] Memo section generation
- [ ] Complete memo compilation
- [ ] PDF/DOCX export

### 📋 Milestone 5: Production Polish (Weeks 9-10)
- [ ] Google Drive integration
- [ ] Cost tracking and optimization
- [ ] Error recovery
- [ ] User documentation

## Testing

### Quick System Check

Run the quick test script to verify all services are running:

```bash
./test_quick.sh
```

This checks:
- API health status
- Database connectivity
- Vector DB status
- Data room listings
- Cost tracking
- Frontend availability

### Comprehensive Testing

See **[TESTING_GUIDE.md](TESTING_GUIDE.md)** for complete testing instructions including:
- Step-by-step setup verification
- CLI tool testing
- Web UI testing checklist
- Error handling verification
- Performance benchmarks
- Common issues and solutions

### Quick Manual Test

1. **Start backend:**
   ```bash
   python app/main.py
   ```

2. **Start frontend** (new terminal):
   ```bash
   cd frontend && npm run dev
   ```

3. **Open browser:** http://localhost:3000

4. **Upload a data room:**
   - Click "Upload"
   - Add company info
   - Drag-and-drop 2-3 PDF/Excel files
   - Click "Create Data Room"

5. **Wait for processing** (30-60 seconds for small data room)

6. **Ask questions:**
   - "What is this company about?"
   - "What is their revenue model?"
   - "Who are the founders?"

---

## Quick Start

### Prerequisites

1. **Xcode Command Line Tools** (macOS)
   ```bash
   xcode-select --install
   ```

2. **Python 3.11+**
   ```bash
   brew install python@3.11
   ```

3. **API Keys**
   - [Anthropic API key](https://console.anthropic.com/) - for Claude
   - [OpenAI API key](https://platform.openai.com/) - for embeddings

### Installation

1. **Run setup script**
   ```bash
   ./setup.sh
   ```

   This automatically:
   - Creates virtual environment
   - Installs all dependencies
   - Initializes databases
   - Creates `.env` file

2. **Add API keys to `.env`**
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

3. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

---

## Usage

### Option 1: Simple CLI Test

Test the complete pipeline with a single PDF:

```bash
python test_cli.py sample.pdf "What is the revenue model?"
```

This will parse the PDF, chunk text, generate embeddings, index to vector DB, and answer your question with citations.

### Option 2: Web API (Recommended)

Start the FastAPI server:

```bash
python app/main.py
```

Open API documentation at http://localhost:8000/docs

**Upload data room:**
```bash
curl -X POST "http://localhost:8000/api/data-room/create" \
  -F "company_name=Acme Inc" \
  -F "analyst_name=John Doe" \
  -F "files=@pitch_deck.pdf" \
  -F "files=@financials.xlsx"
```

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/api/data-room/{id}/question" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the burn rate?"}'
```

### Option 3: Individual Tools

Run tools independently:

```bash
# Parse PDF
python tools/parse_pdf.py sample.pdf

# Parse Excel
python tools/parse_excel.py financials.xlsx

# Search data room
python tools/semantic_search.py "revenue model" --data-room-id test_123

# Answer question
python tools/answer_question.py "What is the burn rate?" --data-room-id test_123
```

## Operating Principles

### Look for existing tools first
Before building anything new, check `tools/` for existing scripts. Only create new ones when necessary.

### Learn and adapt when things fail
When errors occur:
1. Read the full error message and trace
2. Fix the script and retest
3. Document learnings in the workflow
4. Update the workflow to prevent future issues

### Keep workflows current
Workflows should evolve as you learn. Update them when you discover:
- Better methods
- New constraints
- Recurring issues
- Optimization opportunities

## The Self-Improvement Loop

Every failure strengthens the system:
1. Identify what broke
2. Fix the tool
3. Verify the fix works
4. Update the workflow
5. Move forward with a more robust system

## Core Principles

- **Local files are for processing only** - Final outputs go to cloud services
- **Everything in `.tmp/` is disposable** - Can be regenerated as needed
- **Secrets stay in `.env`** - Never store credentials elsewhere
- **Workflows are preserved** - Don't overwrite without explicit permission
- **Deterministic execution wins** - Offload actual work to Python scripts

---

## Cost Management

**Estimated costs per data room:**
- Small (20 docs): ~$2
- Medium (75 docs): ~$6
- Large (200 docs): ~$20

**Monthly target: <$50**
- 5 small data rooms: ~$10
- 2 medium + 1 large: ~$35

**Cost optimization:**
- Embeddings cached (never regenerated)
- Claude Haiku for classification tasks
- Claude Sonnet for analysis/memos
- Batch processing for API calls

---

## Troubleshooting

**Xcode Command Line Tools not installed:**
```bash
xcode-select --install
```

**Python not found:**
```bash
brew install python@3.11
```

**Virtual environment issues:**
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

**API keys not working:**
- Ensure `.env` file exists in project root
- Check keys have no extra spaces or quotes
- Verify keys are valid on provider websites

**Database errors:**
```bash
rm -rf .tmp/
python tools/init_database.py
```

---

## Development

**Run database initialization:**
```bash
python tools/init_database.py
```

**Start API server:**
```bash
uvicorn app.main:app --reload
```

**Check vector DB collections:**
```bash
curl http://localhost:8000/api/collections
```

**View API costs:**
```bash
curl http://localhost:8000/api/costs
```

---

## Why WAT Framework Works

When AI tries to handle every step directly, accuracy compounds poorly. At 90% accuracy per step, you're down to 59% success after just five steps. By offloading execution to deterministic Python scripts, the system stays reliable while AI focuses on orchestration where it excels.

**Architecture:**
- **Workflows**: Markdown SOPs defining processes
- **Agents**: AI (Claude) coordinates execution
- **Tools**: Deterministic Python scripts do the work

**Tech Stack:**
- **Backend**: FastAPI (async, WebSocket support)
- **AI**: Claude (analysis/memos) + OpenAI (embeddings)
- **Vector DB**: ChromaDB (local, persistent)
- **Database**: SQLite (metadata, structured data)
- **Document Processing**: PyPDF2, pdfplumber, pandas

---

## Next Steps

**To start using:**
1. Run `./setup.sh`
2. Add API keys to `.env`
3. Test with `python test_cli.py sample.pdf "your question"`

**To continue development:**
- See [implementation plan](/Users/ziyadel-hamzawi/.claude/plans/vivid-tinkering-harbor.md)
- Next milestone: Multi-file processing & Web UI

---

**Built with the WAT Framework**
*Stay pragmatic. Stay reliable. Keep learning.*
