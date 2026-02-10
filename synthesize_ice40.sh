#!/usr/bin/env bash
# ============================================================================
# synthesize_ice40.sh — Synthesize LUTNN SystemVerilog for iCE40 (iCEBreaker)
#
# Usage:
#   ./synthesize_ice40.sh <sv_directory> [options]
#
# Examples:
#   ./synthesize_ice40.sh data/sv/simple_lutnn              # synth + PnR + bitstream
#   ./synthesize_ice40.sh data/sv/simple_lutnn --synth-only # Yosys synthesis only
#   ./synthesize_ice40.sh data/sv/simple_lutnn --program    # also program the FPGA
#
# Requires: yosys, nextpnr-ice40, icepack, openFPGALoader (for --program)
# ============================================================================
set -euo pipefail

# --- Defaults (iCEBreaker v1.0e) ---
DEVICE="up5k"
PACKAGE="sg48"
TOP_MODULE="top"
SYNTH_ONLY=false
PROGRAM=false

# --- Parse args ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sv_directory> [--synth-only] [--program] [--device DEVICE] [--package PKG] [--top MODULE] [--pcf FILE]"
    exit 1
fi

SV_DIR="$1"
shift

PCF_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synth-only)  SYNTH_ONLY=true ;;
        --program)     PROGRAM=true ;;
        --device)      DEVICE="$2"; shift ;;
        --package)     PACKAGE="$2"; shift ;;
        --top)         TOP_MODULE="$2"; shift ;;
        --pcf)         PCF_FILE="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate input directory ---
if [[ ! -d "$SV_DIR" ]]; then
    echo "Error: SV directory '$SV_DIR' not found."
    exit 1
fi

# --- Collect all .sv files (skip Globals.sv — it's `included by other files) ---
SV_FILES=()
for f in "$SV_DIR"/*.sv; do
    [[ "$(basename "$f")" == "Globals.sv" ]] && continue
    SV_FILES+=("$f")
done

if [[ ${#SV_FILES[@]} -eq 0 ]]; then
    echo "Error: No .sv files found in '$SV_DIR'."
    exit 1
fi

# --- Output paths ---
BUILD_DIR="$SV_DIR/build"
mkdir -p "$BUILD_DIR"
JSON_OUT="$BUILD_DIR/${TOP_MODULE}.json"
ASC_OUT="$BUILD_DIR/${TOP_MODULE}.asc"
BIN_OUT="$BUILD_DIR/${TOP_MODULE}.bin"

echo "========================================"
echo " LUTNN iCE40 Synthesis"
echo "========================================"
echo " SV directory : $SV_DIR"
echo " Device       : $DEVICE"
echo " Package      : $PACKAGE"
echo " Top module   : $TOP_MODULE"
echo " Build dir    : $BUILD_DIR"
echo "========================================"

# --- Step 1: Yosys synthesis ---
echo ""
echo "[1/4] Yosys: Reading SystemVerilog and synthesizing for iCE40..."

READ_CMDS=""
for f in "${SV_FILES[@]}"; do
    READ_CMDS+="read_verilog -sv -I$SV_DIR $f; "
done

yosys -p "${READ_CMDS} synth_ice40 -top $TOP_MODULE -json $JSON_OUT" \
    2>&1 | tee "$BUILD_DIR/yosys.log"

echo "[1/4] Yosys: Done. Netlist: $JSON_OUT"

if $SYNTH_ONLY; then
    echo ""
    echo "Synthesis-only mode. Stopping here."
    echo "Resource usage summary is in $BUILD_DIR/yosys.log"
    exit 0
fi

# --- Step 2: nextpnr place & route ---
echo ""
echo "[2/4] nextpnr-ice40: Place and route..."

NEXTPNR_ARGS=(
    --"$DEVICE"
    --package "$PACKAGE"
    --json "$JSON_OUT"
    --asc "$ASC_OUT"
)

# Add PCF constraints if provided
if [[ -n "$PCF_FILE" ]]; then
    NEXTPNR_ARGS+=(--pcf "$PCF_FILE")
fi

nextpnr-ice40 "${NEXTPNR_ARGS[@]}" 2>&1 | tee "$BUILD_DIR/nextpnr.log"

echo "[2/4] nextpnr-ice40: Done. ASC: $ASC_OUT"

# --- Step 3: icepack ---
echo ""
echo "[3/4] icepack: Generating binary bitstream..."

icepack "$ASC_OUT" "$BIN_OUT"

echo "[3/4] icepack: Done. Bitstream: $BIN_OUT"

# --- Step 4: Program (optional) ---
if $PROGRAM; then
    echo ""
    echo "[4/4] openFPGALoader: Programming iCEBreaker..."
    openFPGALoader -b ice40_generic "$BIN_OUT"
    echo "[4/4] Programming complete!"
else
    echo ""
    echo "[4/4] Skipped programming. Run with --program to flash the FPGA, or manually:"
    echo "      openFPGALoader -b ice40_generic $BIN_OUT"
fi

echo ""
echo "========================================"
echo " Build complete!"
echo " Bitstream: $BIN_OUT"
echo "========================================"
