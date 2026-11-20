#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${WASM_PROFILE:-wasm-release}"
if [[ "$PROFILE" == "dev" ]]; then
  PROFILE_DIR="debug"
else
  PROFILE_DIR="$PROFILE"
fi

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "error: wasm-bindgen not found in PATH." >&2
  echo "Install it with: cargo install --locked wasm-bindgen-cli" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  CRATES=("$@")
else
  CRATES=()
  for dir in crates/*/; do
    [[ -f "${dir}Cargo.toml" ]] || continue
    CRATES+=("$(basename "$dir")")
  done
fi

for crate in "${CRATES[@]}"; do
  out_name="${crate//-/_}"
  case "$crate" in
    op-core-wasm) out_dir="op-core/workstation/pkg" ;;
    *)            out_dir="playground/${crate}/pkg" ;;
  esac

  echo "==> Building ${crate} (profile: ${PROFILE})"
  cargo build --profile "$PROFILE" --target wasm32-unknown-unknown -p "$crate"

  target_base="target/wasm32-unknown-unknown/${PROFILE_DIR}"
  if [[ -f "${target_base}/${out_name}.wasm" ]]; then
    wasm_path="${target_base}/${out_name}.wasm"
  elif [[ -f "${target_base}/${crate}.wasm" ]]; then
    wasm_path="${target_base}/${crate}.wasm"
  else
    echo "error: could not locate compiled .wasm for ${crate} in ${target_base}" >&2
    exit 1
  fi

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  wb_args=(
    --target web
    --no-typescript
    --out-name "$out_name"
    --out-dir "$out_dir"
  )
  if [[ "$PROFILE" != "dev" ]]; then
    wb_args+=(--remove-name-section --remove-producers-section)
  fi

  wasm-bindgen "${wb_args[@]}" "$wasm_path"

  wasm_bg="${out_dir}/${out_name}_bg.wasm"
  if [[ "$PROFILE" != "dev" ]]; then
    if command -v wasm-opt >/dev/null 2>&1; then
      echo "    Optimizing ${wasm_bg} with wasm-opt -Oz"
      wasm-opt -Oz -o "$wasm_bg" "$wasm_bg"
    else
      echo "    Note: wasm-opt not found; skipping Binaryen -Oz pass (runs automatically in CI)"
    fi
  fi

  size="$(du -h --apparent-size "$wasm_bg" | cut -f1)"
  echo "    Ready: ${out_dir}/${out_name}.js (${size})"
done
