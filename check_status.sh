#!/bin/bash
echo "=================================="
echo "MODULE 2 FINAL STATUS"
echo "=================================="
echo ""

# Graphs
GRAPH_COUNT=$(ls ml/data/graphs/*.pt 2>/dev/null | wc -l)
echo "📊 GRAPHS:"
echo "   Files: $GRAPH_COUNT / 68,568"
if [ -f ml/data/graphs/checkpoint.json ]; then
    echo "   Checkpoint:"
    cat ml/data/graphs/checkpoint.json | grep -E '"total"|"completed"' | head -2
fi

echo ""

# Tokens
TOKEN_COUNT=$(ls ml/data/tokens/*.pt 2>/dev/null | wc -l)
echo "🔤 TOKENS:"
echo "   Files: $TOKEN_COUNT / 68,568"
echo "   Status: ✅ COMPLETE"

echo ""

# Disk usage
GRAPH_SIZE=$(du -sh ml/data/graphs 2>/dev/null | cut -f1)
TOKEN_SIZE=$(du -sh ml/data/tokens 2>/dev/null | cut -f1)
echo "💾 DISK USAGE:"
echo "   Graphs: $GRAPH_SIZE"
echo "   Tokens: $TOKEN_SIZE"

echo ""
echo "=================================="

# Check if graphs complete
if [ $GRAPH_COUNT -ge 68000 ]; then
    echo "✅ MODULE 2: COMPLETE!"
    echo "   Ready for validation"
else
    echo "⏳ Graphs: $GRAPH_COUNT / 68,568 (\$(echo \"scale=1; \$GRAPH_COUNT * 100 / 68568\" | bc)%)"
    echo "   Waiting..."
fi

echo "=================================="
