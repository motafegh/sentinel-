"""Target-contract character span helpers for bounded selector research."""

from __future__ import annotations

from sentinel_data.representation.target_selector import declarations


def _mask_strings(source: str) -> str:
    chars = list(source)
    quote: str | None = None
    i = 0
    while i < len(chars):
        ch = chars[i]
        if quote is None:
            if ch in {"'", '"'}:
                quote = ch
                chars[i] = " "
            i += 1
            continue
        if ch == "\\" and i + 1 < len(chars):
            chars[i] = " "
            chars[i + 1] = " "
            i += 2
            continue
        chars[i] = "\n" if ch == "\n" else " "
        if ch == quote:
            quote = None
        i += 1
    return "".join(chars)


def target_contract_char_spans(
    source: str,
    target_names: list[str] | tuple[str, ...],
) -> list[tuple[int, int]]:
    """Return exact declaration-body spans for requested file-graph targets.

    Every target must resolve exactly once and have balanced braces.  The helper
    is lexical and offset-preserving; compilation/Slither remain authoritative
    for Solidity semantics.
    """

    if not target_names:
        raise ValueError("target_names must not be empty")
    masked = _mask_strings(source)
    items = declarations(source)
    spans: list[tuple[int, int]] = []
    for target in target_names:
        matches = [item for item in items if item.name == target]
        if len(matches) != 1:
            raise ValueError(
                f"target declaration count for {target!r} is {len(matches)}"
            )
        start = matches[0].source_offset
        open_brace = masked.find("{", start)
        if open_brace < 0:
            raise ValueError(f"target {target!r} has no opening brace")
        depth = 0
        for index in range(open_brace, len(masked)):
            char = masked[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1))
                    break
        else:
            raise ValueError(f"target {target!r} has no matching closing brace")
    return spans


__all__ = ["target_contract_char_spans"]
