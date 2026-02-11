#!/usr/bin/env bash
# ============================================================================
# synthesize_ice40.sh — Synthesize LUTNN SystemVerilog for iCE40 (iCEBreaker)
#
# Usage:
#   ./synthesize_ice40.sh <sv_directory> [options]
#
# Examples:
#   ./synthesize_ice40.sh data/sv/breast_lutnn --uart                  # UART wrapper
#   ./synthesize_ice40.sh data/sv/breast_lutnn --uart --program        # + flash FPGA
#   ./synthesize_ice40.sh data/sv/breast_lutnn --uart --synth-only     # Yosys only
#   ./synthesize_ice40.sh data/sv/simple_lutnn                         # bare (parallel I/O)
#
# Requires: yosys, nextpnr-ice40, icepack, openFPGALoader (for --program)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Defaults (iCEBreaker v1.0e) ---
DEVICE="up5k"
PACKAGE="sg48"
TOP_MODULE="top"
SYNTH_ONLY=false
PROGRAM=false
USE_UART=false
PCF_FILE=""

# --- Parse args ---
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sv_directory> [--uart] [--synth-only] [--program] [--device DEVICE] [--package PKG] [--top MODULE] [--pcf FILE]"
    exit 1
fi

SV_DIR="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uart)        USE_UART=true ;;
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

# --- Collect LUTNN .sv files (skip Globals.sv — it's `included by other files) ---
SV_FILES=()
for f in "$SV_DIR"/*.sv; do
    [[ "$(basename "$f")" == "Globals.sv" ]] && continue
    SV_FILES+=("$f")
done

if [[ ${#SV_FILES[@]} -eq 0 ]]; then
    echo "Error: No .sv files found in '$SV_DIR'."
    exit 1
fi

# --- UART wrapper mode ---
INCLUDE_DIRS=("-I$SV_DIR")

if $USE_UART; then
    TOP_MODULE="uart_wrapper"

    # Add the UART wrapper
    SV_FILES+=("$SCRIPT_DIR/hdl/uart_wrapper.sv")

    # Add alexforencich's uart_rx.v and uart_tx.v
    UART_RTL="$SCRIPT_DIR/3rdparty/verilog-uart/rtl"
    if [[ ! -d "$UART_RTL" ]]; then
        echo "Error: verilog-uart submodule not found at $UART_RTL"
        echo "       Run: git submodule update --init --recursive"
        exit 1
    fi
    SV_FILES+=("$UART_RTL/uart_rx.v" "$UART_RTL/uart_tx.v")

    # Add hdl/ as include path for uart_wrapper.sv -> Globals.sv
    INCLUDE_DIRS+=("-I$SCRIPT_DIR/hdl")

    # Default PCF for iCEBreaker if not explicitly provided
    if [[ -z "$PCF_FILE" ]]; then
        PCF_FILE="$SCRIPT_DIR/hdl/icebreaker.pcf"
    fi

    echo "UART wrapper mode enabled (top module: uart_wrapper)"
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
if [[ -n "$PCF_FILE" ]]; then
echo " PCF file     : $PCF_FILE"
fi
echo " Build dir    : $BUILD_DIR"
echo "========================================"

# --- Step 1: Yosys synthesis ---
echo ""
echo "[1/4] Yosys: Reading sources and synthesizing for iCE40..."

INCLUDE_FLAGS="${INCLUDE_DIRS[*]}"

READ_CMDS=""
for f in "${SV_FILES[@]}"; do
    if [[ "$f" == *.v ]]; then
        # Plain Verilog (alexforencich's UART)
        READ_CMDS+="read_verilog $f; "
    else
        # SystemVerilog
        READ_CMDS+="read_verilog -sv $INCLUDE_FLAGS $f; "
    fi
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
    --placer heap
)

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
