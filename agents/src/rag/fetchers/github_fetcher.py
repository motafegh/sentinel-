"""
github_fetcher.py

Fetches exploit knowledge from DeFiHackLabs repository.
Parses Solidity PoC files and extracts structured metadata from comments.

Three comment formats exist in the corpus (726 files):

  Format A (@Summary)  — step-by-step attack narrative (~25 files, high-profile)
  Format B (@KeyInfo)  — loss + addresses + @Analysis URLs (~473 files, dominant)
  Format C (free-form) — older files with plain text / bare URLs (~159 files)

CHANGES (2026-04-11):
  FIX-20: past/ directory is now scanned alongside src/test/.
          self.past_path was set in __init__ but fetch() never used it.
          Exploits stored in past/ (archived / older PoCs) were silently
          excluded from the knowledge base.

  FIX-21: fetch_since() no longer silently drops undated files.
          Old: if doc_date_str: guard skipped docs with no date directory.
          Older, historically significant exploits (the ones most likely to
          lack YYYY-MM directory structure) were systematically excluded from
          every incremental update.
          New: undated docs are always included in fetch_since() results.

  FIX-22b: _infer_vuln_type() no longer reads content[:1000].
          The first 1000 characters of a .sol file are always
          SPDX-License-Identifier, pragma solidity, and import statements —
          never exploit logic. Using content[:1000] added noise, not signal.
          Now uses only already-parsed root_cause + summary_block + keyinfo.
"""

import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from .base_fetcher import BaseFetcher, Document


# ── Single-line extraction patterns ──────────────────────────────────────────
PATTERN_TX       = re.compile(r'//\s*@TX\s+(https?://\S+)', re.IGNORECASE)
PATTERN_ROOT     = re.compile(r'//\s*[Rr]oot\s+cause[:\s]+(.+?)(?:\n|$)')
PATTERN_DATE_DIR = re.compile(r'(\d{4})-(\d{2})')
PATTERN_URL      = re.compile(r'https?://\S+')
PATTERN_ANY_TAG  = re.compile(r'^@\w+')

PATTERN_LOSS_KEYINFO = re.compile(
    r'@KeyInfo[^\n]*Total\s+Lost\s*:?\s*[~\$\s]*([0-9][0-9,\.]*)\s*([MKBmkb])?',
    re.IGNORECASE,
)
PATTERN_LOSS_OLD = re.compile(
    r'//\s*(?:Loss|Profit)[:\s]*[\$~]*([0-9,\.]+)\s*([MKBmkb])?',
    re.IGNORECASE,
)

PATTERN_SUMMARY_TAG  = re.compile(r'//\s*@Summary\b',  re.IGNORECASE)
PATTERN_KEYINFO_TAG  = re.compile(r'//\s*@KeyInfo\b',  re.IGNORECASE)
PATTERN_ANALYSIS_TAG = re.compile(r'//\s*@Analysis\b', re.IGNORECASE)


class DeFiHackLabsFetcher(BaseFetcher):
    """
    Fetches exploit PoCs from locally cloned DeFiHackLabs repository.

    Each .sol file → one Document with:
      content:  human-readable description for embedding
      metadata: date, protocol, vuln_type, root_cause, loss_usd, analysis_urls
    """

    def __init__(self, repo_path: Path, data_dir: Path):
        super().__init__(data_dir)
        self.repo_path = Path(repo_path)
        self.src_path  = self.repo_path / "src" / "test"
        self.past_path = self.repo_path / "past"   # FIX-20: now actually used in fetch()

        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"DeFiHackLabs repo not found at {repo_path}. "
                f"Run: git clone --depth=1 "
                f"https://github.com/SunWeb3Sec/DeFiHackLabs.git {repo_path}"
            )

    @property
    def source_name(self) -> str:
        return "DeFiHackLabs"

    def fetch(self) -> list[Document]:
        """
        Parse all .sol exploit files in the repository.

        FIX-20: Scans both src/test/ (main corpus) AND past/ (archived exploits).
                Old code only scanned src/test/ — past_path was declared but unused.

        Returns one Document per exploit file.
        """
        logger.info(f"Fetching from DeFiHackLabs: {self.src_path}")
        documents = []

        # FIX-20: Collect from both directories.
        # past/ may not exist in all repo versions — guard with exists() check.
        sol_files = list(self.src_path.rglob("*.sol"))
        if self.past_path.exists():
            past_files = list(self.past_path.rglob("*.sol"))
            logger.info(f"Found {len(sol_files)} files in src/test/ + {len(past_files)} in past/")
            sol_files += past_files
        else:
            logger.info(f"Found {len(sol_files)} Solidity exploit files (past/ not present)")

        for sol_file in sol_files:
            try:
                doc = self._parse_sol_file(sol_file)
                if doc is not None:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse {sol_file.name}: {e}")

        logger.info(f"Successfully parsed {len(documents)} exploit documents")
        return documents

    def fetch_since(self, since: datetime) -> list[Document]:
        """
        Return documents from files dated on or after `since`.

        FIX-21: Undated files are now always included.
                Old code: if doc_date_str: guard silently dropped any document
                whose path had no YYYY-MM directory. Older, historically
                significant exploits (Format C, free-form) were excluded from
                every incremental update — they only appeared on full fetch().
                New code: undated docs pass through unconditionally, with a
                debug log so the operator can see how many were included.
        """
        logger.info(f"Fetching DeFiHackLabs documents since {since.date()}")
        all_docs = self.fetch()

        dated_new  = []
        undated    = []
        dated_skip = 0

        for doc in all_docs:
            date_str = doc.metadata.get("date", "")
            if not date_str:
                # FIX-21: Include undated documents — can't know if they're old.
                undated.append(doc)
            else:
                try:
                    doc_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if doc_date >= since:
                        dated_new.append(doc)
                    else:
                        dated_skip += 1
                except ValueError:
                    undated.append(doc)   # malformed date → treat as undated

        filtered = dated_new + undated
        logger.info(
            f"fetch_since results: {len(dated_new)} dated-new, "
            f"{len(undated)} undated (always included), "
            f"{dated_skip} dated-old (skipped)"
        )
        return filtered

    def _parse_sol_file(self, sol_file: Path) -> Optional[Document]:
        """
        Parse a single .sol exploit file into a Document.

        Extraction priority:
          1. @Summary block  → step-by-step attack (Format A)
          2. @KeyInfo block  → loss + addresses (Format B)
          3. @Analysis URLs  → post-mortem references (Format B)
          4. Comment lines   → fallback for old files (Format C)
        """
        content = sol_file.read_text(encoding="utf-8", errors="ignore")

        if "Poc-template" in sol_file.name or "RPCS_alive" in sol_file.name:
            return None

        date_str = self._extract_date(sol_file)
        protocol = sol_file.stem.replace("_exp", "").replace("_hack", "")

        summary_block = self._extract_summary_block(content)
        keyinfo_block = self._extract_keyinfo_block(content)
        analysis_urls = self._extract_all_analysis_urls(content)
        tx_url        = self._extract_first(PATTERN_TX, content)
        root_cause    = self._extract_root_cause(content)
        loss_usd      = self._extract_loss(content)
        vuln_type     = self._infer_vuln_type(root_cause, summary_block, keyinfo_block)

        content_text = self._build_content(
            protocol=protocol,
            date=date_str,
            root_cause=root_cause,
            vuln_type=vuln_type,
            loss_usd=loss_usd,
            analysis_urls=analysis_urls,
            tx_url=tx_url,
            summary_block=summary_block,
            keyinfo_block=keyinfo_block,
            raw_content=content,
        )

        if not content_text.strip():
            return None

        doc_id = hashlib.sha256(str(sol_file).encode()).hexdigest()[:16]

        return Document(
            content=content_text,
            source=self.source_name,
            doc_id=doc_id,
            metadata={
                "protocol":      protocol,
                "date":          date_str,
                "vuln_type":     vuln_type,
                "root_cause":    root_cause or "unknown",
                "loss_usd":      loss_usd,
                "analysis_urls": analysis_urls,
                "tx_url":        tx_url or "",
                "file_path":     str(sol_file),
                "source":        self.source_name,
                "has_summary":   summary_block is not None,
                "has_keyinfo":   bool(keyinfo_block),
            }
        )

    def _build_content(
        self,
        protocol: str,
        date: str,
        root_cause: Optional[str],
        vuln_type: str,
        loss_usd: Optional[int],
        analysis_urls: list[str],
        tx_url: Optional[str],
        summary_block: Optional[str],
        keyinfo_block: dict,
        raw_content: str,
    ) -> str:
        parts = [f"DeFi Security Incident: {protocol}"]
        if date:
            parts.append(f"Date: {date}")
        parts.append(f"Vulnerability type: {vuln_type}")
        if loss_usd:
            parts.append(f"Estimated loss: ${loss_usd:,}")
        if root_cause:
            parts.append(f"Root cause: {root_cause}")

        if summary_block:
            parts.append("\nAttack summary:")
            parts.append(summary_block)

        if keyinfo_block:
            parts.append("\nIncident details:")
            if keyinfo_block.get("keyinfo_line"):
                parts.append(keyinfo_block["keyinfo_line"])
            for key in ("attacker", "attack_contract", "vulnerable_contract", "tx_line"):
                if keyinfo_block.get(key):
                    parts.append(keyinfo_block[key])

        # tx_url from @TX tag — add if not already covered by keyinfo_block tx_line
        if tx_url and not keyinfo_block.get("tx_line"):
            parts.append(f"Transaction: {tx_url}")

        if analysis_urls:
            parts.append("\nPost-mortem references:")
            parts.extend(analysis_urls[:5])

        # Fallback for Format C — raw comment lines when no structured data exists
        if not summary_block and not keyinfo_block and not analysis_urls:
            comment_lines = [
                line.strip().lstrip("//").strip()
                for line in raw_content.split("\n")
                if line.strip().startswith("//")
                and len(line.strip()) > 5
                and "SPDX" not in line
                and "pragma" not in line.lower()
            ]
            if comment_lines:
                parts.append("\nDetailed notes:")
                parts.extend(comment_lines[:30])

        return "\n".join(parts)

    # ── Block extractors ──────────────────────────────────────────────────────

    def _extract_block_lines(self, content: str, start_pattern: re.Pattern) -> list[str]:
        """Extract consecutive // comment lines after a tag line."""
        lines    = content.split("\n")
        result   = []
        in_block = False
        for line in lines:
            stripped = line.strip()
            if not in_block:
                if start_pattern.search(stripped):
                    in_block = True
                continue
            if stripped.startswith("//"):
                text = stripped.lstrip("/").strip()
                if PATTERN_ANY_TAG.match(text):
                    break
                if text and "SPDX" not in text:
                    result.append(text)
            else:
                break
        return result

    def _extract_summary_block(self, content: str) -> Optional[str]:
        lines = self._extract_block_lines(content, PATTERN_SUMMARY_TAG)
        return "\n".join(lines) if lines else None

    def _extract_keyinfo_block(self, content: str) -> dict:
        info: dict = {}

        for line in content.split("\n"):
            if PATTERN_KEYINFO_TAG.search(line):
                info["keyinfo_line"] = line.strip().lstrip("/").strip()
                break

        if not info:
            return {}

        for line in self._extract_block_lines(content, PATTERN_KEYINFO_TAG):
            lower = line.lower()
            if "attacker" in lower and "contract" not in lower:
                info.setdefault("attacker", line)
            elif "attack contract" in lower or "exploit contract" in lower:
                info.setdefault("attack_contract", line)
            elif "vulnerable contract" in lower:
                info.setdefault("vulnerable_contract", line)
            elif "tx" in lower or "transaction" in lower:
                info.setdefault("tx_line", line)

        return info

    def _extract_all_analysis_urls(self, content: str) -> list[str]:
        lines = self._extract_block_lines(content, PATTERN_ANALYSIS_TAG)
        urls: list[str] = []
        for line in lines:
            urls.extend(PATTERN_URL.findall(line))
        seen: set[str] = set()
        unique: list[str] = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    # ── Single-value extractors ─────────────────────────────────��─────────────

    def _extract_date(self, sol_file: Path) -> str:
        """Extract date from directory name like 2023-03 → '2023-03-01'."""
        for part in sol_file.parts:
            match = PATTERN_DATE_DIR.match(part)
            if match:
                year, month = match.groups()
                return f"{year}-{month}-01"
        return ""

    def _extract_root_cause(self, content: str) -> Optional[str]:
        match = PATTERN_ROOT.search(content)
        return match.group(1).strip() if match else None

    def _extract_first(self, pattern: re.Pattern, content: str) -> Optional[str]:
        match = pattern.search(content)
        return match.group(1).strip() if match else None

    def _extract_loss(self, content: str) -> Optional[int]:
        """Parse loss amount. Handles: $197M, ~59643 USD, 1.4M, 15k."""
        match = PATTERN_LOSS_KEYINFO.search(content)
        if not match:
            match = PATTERN_LOSS_OLD.search(content)
        if not match:
            return None
        try:
            amount = float(match.group(1).replace(",", ""))
            suffix = (match.group(2) or "").upper()
            mult   = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
            return int(amount * mult)
        except (ValueError, TypeError):
            return None

    def _infer_vuln_type(
        self,
        root_cause: Optional[str],
        summary_block: Optional[str],
        keyinfo_block: dict,
    ) -> str:
        """
        Infer vulnerability type from already-parsed structured fields only.

        FIX-22b: Removed content[:1000] from type inference.
                 The first 1000 characters of any .sol file are always:
                   // SPDX-License-Identifier: MIT
                   pragma solidity ^0.8.0;
                   import "forge-std/Test.sol";
                   ...
                 None of that is exploit logic. Using it added noise and could
                 produce false positives (e.g. "pragma" doesn't mean anything;
                 a "Test" import doesn't mean "logic_error").
                 root_cause + summary_block + keyinfo are the semantic signal.

        Args:
            root_cause:    extracted root cause string (or None)
            summary_block: extracted @Summary block text (or None)
            keyinfo_block: extracted @KeyInfo dict (keys: keyinfo_line, attacker, ...)
        """
        # Combine all structured text sources — no raw file content
        keyinfo_text = " ".join(str(v) for v in keyinfo_block.values()) if keyinfo_block else ""
        text = " ".join([
            root_cause    or "",
            summary_block or "",
            keyinfo_text,
        ]).lower()

        # Most specific patterns first — order matters for multi-type incidents
        if "reentrancy" in text or "reentrant" in text:
            return "reentrancy"
        elif "flash loan" in text or "flashloan" in text:
            return "flash_loan"
        elif "price manip" in text or "oracle manip" in text:
            return "oracle_manipulation"
        elif "access control" in text or "privilege" in text or "unauthorized" in text:
            return "access_control"
        elif "overflow" in text or "underflow" in text:
            return "integer_overflow"
        elif "front.run" in text or "frontrun" in text or "mev" in text:
            return "front_running"
        elif "logic" in text or "business logic" in text:
            return "logic_error"
        elif "timestamp" in text:
            return "timestamp_dependence"
        elif "delegatecall" in text:
            return "delegatecall"
        elif "dos" in text or "denial" in text:
            return "denial_of_service"
        else:
            return "other"


if __name__ == "__main__":
    _agents_dir = Path(__file__).parent.parent.parent.parent

    fetcher = DeFiHackLabsFetcher(
        repo_path=_agents_dir / "data" / "defihacklabs",
        data_dir=_agents_dir / "data" / "exploits",
    )
    docs = fetcher.fetch()

    total       = len(docs)
    with_summary = sum(1 for d in docs if d.metadata.get("has_summary"))
    with_keyinfo = sum(1 for d in docs if d.metadata.get("has_keyinfo"))
    with_vuln    = sum(1 for d in docs if d.metadata.get("vuln_type") != "other")
    with_loss    = sum(1 for d in docs if d.metadata.get("loss_usd"))
    with_urls    = sum(1 for d in docs if d.metadata.get("analysis_urls"))

    logger.info(f"\n{'='*50}")
    logger.info(f"Total documents:       {total}")
    logger.info(f"With @Summary block:   {with_summary} ({with_summary/total*100:.1f}%)")
    logger.info(f"With @KeyInfo block:   {with_keyinfo} ({with_keyinfo/total*100:.1f}%)")
    logger.info(f"With vuln type:        {with_vuln} ({with_vuln/total*100:.1f}%)")
    logger.info(f"With loss amount:      {with_loss} ({with_loss/total*100:.1f}%)")
    logger.info(f"With analysis URLs:    {with_urls} ({with_urls/total*100:.1f}%)")
    logger.info(f"{'='*50}")

    euler_docs = [d for d in docs if "Euler" in d.metadata.get("protocol", "")]
    if euler_docs:
        e = euler_docs[0]
        logger.info(f"\n--- Euler Finance (smoke test) ---")
        logger.info(f"Protocol:    {e.metadata['protocol']}")
        logger.info(f"Date:        {e.metadata['date']}")
        logger.info(f"Vuln type:   {e.metadata['vuln_type']}")
        if e.metadata["loss_usd"]:
            logger.info(f"Loss:        ${e.metadata['loss_usd']:,}")
        logger.info(f"Has summary: {e.metadata['has_summary']}")
        logger.info(f"URLs:        {len(e.metadata['analysis_urls'])}")
        logger.info(f"\nContent preview:\n{e.content[:600]}")
