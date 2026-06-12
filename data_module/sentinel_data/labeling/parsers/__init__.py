"""Source-specific label parsers.

Each parser reads preprocessed ``.meta.json`` sidecars, applies the
corresponding crosswalk YAML, and writes per-contract ``.labels.json``
files.  Parsers are source-specific — adding a new data source means
adding a new parser without touching the merger or taxonomy.
"""
