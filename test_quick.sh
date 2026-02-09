#!/bin/bash

# Quick test script for Noesis AI
# Tests that all core services are running and accessible

echo "========================================"
echo "Noesis AI - Quick System Test"
echo "========================================"
echo ""

# Check if backend is running
echo "1. Testing API health..."
HEALTH=$(curl -s http://localhost:8000/api/health 2>/dev/null)
if echo "$HEALTH" | grep -q "healthy"; then
    echo "   ✅ API is healthy"
else
    echo "   ❌ API is not running or unhealthy"
    echo "   Start with: python app/main.py"
    exit 1
fi

# Check database
echo "2. Checking database..."
if [ -f ".tmp/due_diligence.db" ]; then
    echo "   ✅ Database exists"
else
    echo "   ❌ Database not found"
    echo "   Run: python tools/init_database.py"
fi

# Check vector DB
echo "3. Checking vector database..."
if [ -d ".tmp/chroma_db" ]; then
    echo "   ✅ Vector DB directory exists"
else
    echo "   ❌ Vector DB not initialized"
fi

# List data rooms
echo "4. Listing data rooms..."
DATA_ROOMS=$(curl -s http://localhost:8000/api/data-rooms 2>/dev/null)
if echo "$DATA_ROOMS" | grep -q "data_rooms"; then
    COUNT=$(echo "$DATA_ROOMS" | grep -o '"total":[0-9]*' | grep -o '[0-9]*')
    echo "   ✅ Found $COUNT data room(s)"
else
    echo "   ❌ Cannot list data rooms"
fi

# Check vector DB collections
echo "5. Checking vector DB collections..."
COLLECTIONS=$(curl -s http://localhost:8000/api/collections 2>/dev/null)
if echo "$COLLECTIONS" | grep -q "collections"; then
    COLL_COUNT=$(echo "$COLLECTIONS" | grep -o '"total":[0-9]*' | grep -o '[0-9]*')
    echo "   ✅ Found $COLL_COUNT collection(s)"
else
    echo "   ❌ Cannot access vector DB"
fi

# Check cost tracking
echo "6. Checking cost tracking..."
COSTS=$(curl -s http://localhost:8000/api/costs 2>/dev/null)
if echo "$COSTS" | grep -q "total_cost"; then
    TOTAL_COST=$(echo "$COSTS" | grep -o '"total_cost":[0-9.]*' | grep -o '[0-9.]*')
    echo "   ✅ Cost tracking active (Total: \$$TOTAL_COST)"
else
    echo "   ❌ Cost tracking not working"
fi

# Check frontend (if running)
echo "7. Checking frontend..."
FRONTEND=$(curl -s http://localhost:3000 2>/dev/null)
if [ -n "$FRONTEND" ]; then
    echo "   ✅ Frontend is running"
else
    echo "   ⚠️  Frontend not running (optional)"
    echo "   Start with: cd frontend && npm run dev"
fi

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "Backend API:     http://localhost:8000"
echo "API Docs:        http://localhost:8000/docs"
echo "Frontend:        http://localhost:3000"
echo ""
echo "For detailed testing, see: TESTING_GUIDE.md"
echo ""
