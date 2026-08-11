#!/usr/bin/env bash
set -euo pipefail

# Reconstruct contracts/lib deterministically from contracts/foundry.lock.
# The historical contracts/.gitmodules file is not a repository-root
# .gitmodules file, so a fresh Git checkout does not initialise these libs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$ROOT/lib"

clone_at() {
  local name="$1"
  local url="$2"
  local rev="$3"
  local dest="$LIB/$name"

  if [[ -d "$dest/.git" ]]; then
    actual="$(git -C "$dest" rev-parse HEAD)"
    if [[ "$actual" == "$rev" ]]; then
      echo "[deps] $name already at $rev"
      return 0
    fi
    echo "[deps] ERROR: $dest exists at $actual, expected $rev" >&2
    echo "[deps] Remove that directory explicitly before bootstrapping another revision." >&2
    return 1
  fi
  if [[ -e "$dest" ]]; then
    echo "[deps] ERROR: $dest exists but is not a Git checkout" >&2
    return 1
  fi

  echo "[deps] cloning $name"
  git clone --no-checkout "$url" "$dest"
  git -C "$dest" checkout --detach "$rev"
  actual="$(git -C "$dest" rev-parse HEAD)"
  [[ "$actual" == "$rev" ]] || {
    echo "[deps] ERROR: $name resolved to $actual, expected $rev" >&2
    return 1
  }
}

mkdir -p "$LIB"

clone_at \
  "forge-std" \
  "https://github.com/foundry-rs/forge-std" \
  "7117c90c8cf6c68e5acce4f09a6b24715cea4de6"

clone_at \
  "openzeppelin-contracts" \
  "https://github.com/OpenZeppelin/openzeppelin-contracts" \
  "f910b26cfeeec28f793ef3cf1938da4b40082d5c"

clone_at \
  "openzeppelin-contracts-upgradeable" \
  "https://github.com/OpenZeppelin/openzeppelin-contracts-upgradeable" \
  "37d55aa59ee263673badb8a7c97dc78c72d4e27e"

echo "[deps] deterministic Foundry dependencies ready"
