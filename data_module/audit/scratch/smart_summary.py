"""Smart per-contract summary for the agreed-set validation.

For each .sol file, produce a compact excerpt that highlights:
- All functions + their visibility + modifiers
- State-changing external calls (.call/.send/.transfer/delegatecall)
- Access-control modifiers (onlyOwner / auth / onlyRole / require(msg.sender==...))
- For EB: any function missing access control that should have it
- For RE: any function doing external call before state update
"""
import re
import sys
from pathlib import Path

def summarize(path: Path) -> str:
    src = path.read_text(errors='replace')
    lines = src.splitlines()
    n = len(lines)
    if n == 0:
        return "(empty)"
    out = []
    out.append(f"[{n} lines, {len(src)} chars]")
    # find all function definitions
    fn_re = re.compile(
        r'^\s*function\s+(\w+)\s*\(([^)]*)\)\s+([^{;]*?)\s*\{',
        re.MULTILINE,
    )
    fns = []
    for m in fn_re.finditer(src):
        name = m.group(1)
        args = m.group(2).strip()
        ret = m.group(3).strip()
        # strip body braces - just need first 5 lines of body
        body_start = m.end()
        body = src[body_start:body_start + 1200]  # up to 1200 chars into body
        # find first { - already past it; find matching } approximately
        depth = 1
        i = 0
        for i, ch in enumerate(body):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    break
        body_text = body[:i]
        # find line offset
        line_no = src[:m.start()].count('\n') + 1
        fns.append((line_no, name, args, ret, body_text))
    if not fns:
        # no functions found - return first 50 lines raw
        return "\n".join(out + lines[:80])
    # limit to ~12 functions shown
    for line_no, name, args, ret, body in fns[:15]:
        out.append(f"  L{line_no} fn {name}({args[:60]}) {ret[:60]}")
        # look for critical patterns in body
        body_lines = body.splitlines()[:18]
        for bl in body_lines:
            bl_s = bl.strip()
            if not bl_s or bl_s.startswith('//') or bl_s.startswith('*'):
                continue
            # highlight key patterns
            if any(p in bl for p in ['onlyOwner', 'onlyAdmin', 'onlyRole', 'auth', 'require(msg.sender', 'tx.origin']):
                out.append(f"      | {bl_s[:120]}")
            elif any(p in bl for p in ['.call', '.send', '.transfer', '.delegatecall', 'sendValue']):
                out.append(f"      | {bl_s[:120]}")
            elif any(p in bl for p in ['= ', 'mapping', 'balances[', 'state.', '.amount']):
                out.append(f"      | {bl_s[:120]}")
    if len(fns) > 15:
        out.append(f"  ... and {len(fns)-15} more functions")
    return "\n".join(out)

if __name__ == '__main__':
    p = Path(sys.argv[1])
    print(summarize(p))
