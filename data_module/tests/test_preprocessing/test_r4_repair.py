"""Repository-safe regression tests for the Phase-8 DATA repair primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel_data.preprocessing.compiler import build_solc_command
from sentinel_data.preprocessing.deduplicator import Deduplicator
from sentinel_data.preprocessing.normalizer import normalize
from sentinel_data.preprocessing.r4_pipeline import PreparedRecord, _materialize
from sentinel_data.preprocessing.r4_versions import (
    PREPROCESSING_ARTIFACT_VERSION,
    PREPROCESSING_META_SCHEMA_VERSION,
)


class TestR4LexicalNormalization:
    def test_comment_markers_inside_strings_are_not_removed(self):
        src = (
            'pragma solidity ^0.8.0;\n'
            'contract C {\n'
            '  string a = "https://example.test/a//b"; // real comment\n'
            '  string b = "/* not a comment */";\n'
            '}\n'
        )
        result = normalize(src, preserve_line_structure=True)
        assert 'https://example.test/a//b' in result.content
        assert '/* not a comment */' in result.content
        assert 'real comment' not in result.content

    def test_escaped_quote_does_not_end_string_state(self):
        src = 'contract C { string s = "quote: \\\" // still string"; /* gone */ }\n'
        result = normalize(src, preserve_line_structure=True)
        assert '// still string' in result.content
        assert 'gone' not in result.content

    def test_block_comment_newlines_are_preserved(self):
        src = 'contract C {\n/* one\ntwo\nthree */\nuint x;\n}\n'
        result = normalize(src, preserve_line_structure=True)
        assert result.content.count('\n') == src.count('\n')
        assert result.line_structure_preserved is True

    def test_repaired_normalization_is_idempotent(self):
        src = 'contract C {\n  string u = "https://x/y"; // comment\n\n  uint x;\n}\n'
        once = normalize(src, preserve_line_structure=True).content
        twice = normalize(once, preserve_line_structure=True).content
        assert twice == once

    def test_legacy_mode_keeps_blank_line_compaction_contract(self):
        src = 'uint x;\n\n\n\nuint y;\n'
        assert '\n\n\n' not in normalize(src).content


class TestR4DedupSemantics:
    ADDRESS = '0x1111111111111111111111111111111111111111'

    def test_same_address_distinct_content_survives(self):
        dedup = Deduplicator()
        first = dedup.process(
            f'contract A {{ address a = {self.ADDRESS}; uint x = 1; }}\n',
            Path('a.sol'),
        )
        second = dedup.process(
            f'contract B {{ address a = {self.ADDRESS}; uint x = 2; }}\n',
            Path('b.sol'),
        )
        assert first.address_literals == (self.ADDRESS,)
        assert second.address_literals == (self.ADDRESS,)
        assert not first.is_duplicate
        assert not second.is_duplicate
        assert first.dedup_group_id != second.dedup_group_id

    def test_exact_duplicate_collapses(self):
        dedup = Deduplicator()
        src = 'contract A {}\n'
        first = dedup.process(src, Path('a.sol'))
        second = dedup.process(src, Path('b.sol'))
        assert not first.is_duplicate
        assert second.is_duplicate
        assert second.duplicate_kind == 'exact'
        assert first.dedup_group_id == second.dedup_group_id

    def test_comment_and_format_variant_gets_normalized_group(self):
        dedup = Deduplicator()
        first = dedup.process('contract A { uint x; }\n', Path('a.sol'))
        second = dedup.process('contract A {\n  uint x; // note\n}\n', Path('b.sol'))
        assert not first.is_duplicate
        assert second.is_duplicate
        assert second.duplicate_kind == 'normalized'
        assert first.dedup_group_id == second.dedup_group_id


class TestR4CompilerFlags:
    def test_solc_049_does_not_receive_allow_paths(self, tmp_path):
        command = build_solc_command(
            Path('/tool/solc-0.4.9'),
            tmp_path / 'C.sol',
            '0.4.9',
            allow_root=tmp_path,
        )
        assert '--allow-paths' not in command

    def test_solc_050_and_newer_can_receive_allow_paths(self, tmp_path):
        command = build_solc_command(
            Path('/tool/solc-0.5.0'),
            tmp_path / 'C.sol',
            '0.5.0',
            allow_root=tmp_path,
        )
        assert command[-2:] == ['--allow-paths', str(tmp_path)]


class TestR4DeterministicPromotion:
    def _record(self, staging: Path, *, record_id: str, artifact_sha: str) -> PreparedRecord:
        path = staging / f'{record_id}.sol'
        path.write_text('pragma solidity ^0.8.0;\ncontract C {}\n')
        return PreparedRecord(
            source_name='fixture',
            original_path=f'repo/{record_id}.sol',
            source_record_id=record_id,
            raw_sha256=record_id * 64,
            flattened_sha256='f' * 64,
            normalized_text_sha256=artifact_sha,
            normalized_code_sha256='n' * 64,
            address_literals=(),
            pragma='^0.8.0',
            solc_version='0.8.20',
            attempted_solc_versions=('0.8.20',),
            command_flags=('--allow-paths', '/fixture'),
            flatten_status='skipped_no_imports',
            version_bucket='modern',
            has_unchecked_block=False,
            contract_names=('C',),
            n_raw_lines=3,
            n_normalized_lines=3,
            normalizer_version='r4-lexical-v2',
            staging_path=str(path),
            ingestion_entry={'path': f'repo/{record_id}.sol'},
        )

    def test_exact_artifact_aggregates_all_source_provenance(self, tmp_path):
        staging = tmp_path / 'staging'
        output = tmp_path / 'output'
        staging.mkdir()
        output.mkdir()
        artifact_sha = 'a' * 64
        records = [
            self._record(staging, record_id='b', artifact_sha=artifact_sha),
            self._record(staging, record_id='a', artifact_sha=artifact_sha),
        ]

        result = _materialize('fixture', records, [], output)
        assert result.artifacts_written == 1
        meta = json.loads((output / f'{artifact_sha}.meta.json').read_text())
        assert meta['source_record_count'] == 2
        assert [r['source_record_id'] for r in meta['source_records']] == ['a', 'b']
        assert meta['meta_schema_version'] == PREPROCESSING_META_SCHEMA_VERSION
        assert meta['preprocessing_artifact_version'] == PREPROCESSING_ARTIFACT_VERSION

    def test_address_candidates_are_evidence_not_auto_merge(self, tmp_path):
        staging = tmp_path / 'staging'
        output = tmp_path / 'output'
        staging.mkdir()
        output.mkdir()
        r1 = self._record(staging, record_id='a', artifact_sha='a' * 64)
        r2 = self._record(staging, record_id='b', artifact_sha='b' * 64)
        address = '0x2222222222222222222222222222222222222222'
        r1 = PreparedRecord(**{**r1.__dict__, 'address_literals': (address,)})
        r2 = PreparedRecord(**{**r2.__dict__, 'address_literals': (address,)})

        result = _materialize('fixture', [r2, r1], [], output)
        assert result.artifacts_written == 2
        manifest = json.loads((output / 'repaired_preprocessing_manifest.json').read_text())
        assert manifest['address_family_candidates_not_auto_merged'][address] == [
            'a' * 64,
            'b' * 64,
        ]
