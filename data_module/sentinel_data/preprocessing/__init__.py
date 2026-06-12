"""Preprocessing submodule — 5-step pipeline for .sol contract files.

Pipeline order: flatten → compile → dedup → normalize → segment+bucket.
Each step writes sidecar meta.json fields for downstream ML stages.
"""
