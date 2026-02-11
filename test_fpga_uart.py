#!/usr/bin/env python3
"""
test_fpga_uart.py — Send Breast Cancer test samples to LUTNN on iCEBreaker via UART.

Usage:
    python test_fpga_uart.py                    # test first sample
    python test_fpga_uart.py --all              # test all samples
    python test_fpga_uart.py --port /dev/tty.usbserial-ibNy1qMf1
"""
import argparse
import struct
import time
import serial
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.uci_datasets import BreastCancerDataset


def find_serial_port():
    """Auto-detect the iCEBreaker UART serial port on macOS.
    The FTDI FT2232H has two channels: A=JTAG (index 0), B=UART (index 1).
    We want channel B — the second/last port.
    """
    import glob
    candidates = sorted(glob.glob("/dev/tty.usbserial-*"))
    if len(candidates) >= 2:
        return candidates[-1]  # Channel B (UART) is the second one
    if candidates:
        return candidates[0]
    # Also try Linux-style
    candidates = sorted(glob.glob("/dev/ttyUSB*"))
    if candidates:
        return candidates[-1]
    return None


def sample_to_bytes(sample, n_inputs=51):
    """Pack a binary input vector into ceil(n_inputs/8) bytes, MSB-first."""
    n_bytes = (n_inputs + 7) // 8
    bits = sample.flatten().int().tolist()[:n_inputs]
    # Pad to n_bytes * 8 bits
    bits = bits + [0] * (n_bytes * 8 - len(bits))

    result = bytearray(n_bytes)
    for i in range(n_bytes):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i * 8 + j]
        result[i] = byte_val
    return bytes(result)


def classify_fpga(ser, sample, n_inputs=51):
    """Send a sample to the FPGA and get the classification result."""
    data = sample_to_bytes(sample, n_inputs)

    # Flush any stale data
    ser.reset_input_buffer()

    # Send input bytes
    ser.write(data)
    ser.flush()

    # Read 1-byte response
    response = ser.read(1)
    if len(response) == 0:
        return None
    return response[0]


def main():
    parser = argparse.ArgumentParser(description="Test LUTNN on iCEBreaker via UART")
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--all", action="store_true", help="Test all samples in test set")
    parser.add_argument("--count", type=int, default=1, help="Number of samples to test")
    args = parser.parse_args()

    # Find serial port
    port = args.port or find_serial_port()
    if port is None:
        print("Error: No serial port found. Specify with --port")
        print("  Available ports: ls /dev/tty.usbserial-*")
        sys.exit(1)
    print(f"Using serial port: {port}")

    # Load test dataset
    test_set = BreastCancerDataset('./data-uci', split='test', download=True, with_val=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    labels = {0: "Benign", 1: "Malignant"}

    # Open serial connection
    ser = serial.Serial(port, args.baud, timeout=2)
    time.sleep(0.1)  # Let the connection stabilize

    count = len(test_set) if args.all else args.count
    correct = 0
    total = 0

    print(f"\nTesting {count} sample(s)...\n")
    print(f"{'#':>4}  {'FPGA':>10}  {'Expected':>10}  {'Match':>5}")
    print("-" * 40)

    for i, (sample, target) in enumerate(test_loader):
        if i >= count:
            break

        result = classify_fpga(ser, sample)

        if result is None:
            print(f"{i:4d}  {'TIMEOUT':>10}  {labels.get(target.item(), '?'):>10}  {'':>5}")
            continue

        expected = target.item()
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        total += 1

        print(f"{i:4d}  {labels.get(result, f'?({result})'):>10}  {labels.get(expected, '?'):>10}  {match:>5}")

    ser.close()

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {100 * correct / total:.1f}%")


if __name__ == "__main__":
    main()
