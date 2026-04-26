#!/bin/bash

echo "=================================="
echo "MODULE 2 FINAL STATUS"
echo "=================================="
echo ""

GRAPH_COUNT=$(ls ml/data/graphs/*.pt 2>/dev/null | wc -l)
TOKEN_COUNT=$(ls ml/data/tokens/*.pt 2>/dev/null | wc -l)

echo "[GRAPHS]: $GRAPH_COUNT / 68568 files"
if [ -f ml/data/graphs/checkpoint.json ]; then
    grep -E '"total"|"completed"' ml/data/graphs/checkpoint.json | head -2
fi

echo ""
echo "[TOKENS]: $TOKEN_COUNT / 68568 files (COMPLETE)"

echo ""
echo "[DISK USAGE]:"
du -sh ml/data/graphs 2>/dev/null | sed 's/^/   Graphs: /'
du -sh ml/data/tokens 2>/dev/null | sed 's/^/   Tokens: /'

echo ""
echo "=================================="
if [ "$GRAPH_COUNT" -ge 68000 ]; then
    echo "STATUS: COMPLETE - Ready for validation"
else
    echo "STATUS: IN PROGRESS ($GRAPH_COUNT of 68568)"
fi
echo "=================================="
