#!/usr/bin/env python3
"""Print an ONNX model's inputs, outputs, opsets, op histogram and parameter count.

Dependency-free: decodes the protobuf wire format directly (field numbers from
onnx/onnx.proto), so no `onnx` package is needed. Used to write models/README.md
and to size the M5 custom-kernel work.

    scripts/onnx_info.py models/rec.onnx [--markdown]
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def fields(buf: bytes):
    """Yield (field_number, wire_type, value) for one message; value is int or bytes."""
    i = 0
    while i < len(buf):
        key, i = read_varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = read_varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, v


def value_info(buf: bytes) -> tuple[str, list[str]]:
    """ValueInfoProto -> (name, dims) with symbolic dims as their names."""
    name, dims = "", []
    for num, _, v in fields(buf):
        if num == 1:
            name = v.decode()
        elif num == 2:  # TypeProto
            for tn, _, tv in fields(v):
                if tn != 1:  # tensor_type
                    continue
                for sn, _, sv in fields(tv):
                    if sn != 2:  # shape
                        continue
                    for dn, _, dv in fields(sv):
                        if dn != 1:  # dim
                            continue
                        dim = "?"
                        for kn, _, kv in fields(dv):
                            dim = str(kv) if kn == 1 else kv.decode() if kn == 2 else dim
                        dims.append(dim)
    return name, dims


def packed_or_repeated_ints(entries: list[tuple[int, object]]) -> list[int]:
    out = []
    for wt, v in entries:
        if wt == 0:
            out.append(v)
        else:  # packed
            i = 0
            while i < len(v):
                x, i = read_varint(v, i)
                out.append(x)
    return out


def inspect(path: Path) -> dict:
    model = path.read_bytes()
    info = {"ir_version": None, "opsets": [], "inputs": [], "outputs": [],
            "ops": collections.Counter(), "params": 0, "initializers": 0}
    graph = b""
    for num, _, v in fields(model):
        if num == 1:
            info["ir_version"] = v
        elif num == 7:
            graph = v
        elif num == 8:
            dom, ver = "", 0
            for on, _, ov in fields(v):
                dom = ov.decode() if on == 1 else dom
                ver = ov if on == 2 else ver
            info["opsets"].append((dom or "ai.onnx", ver))
    def tensor_elems(t: bytes) -> int:
        dims = [(twt, tv) for tn, twt, tv in fields(t) if tn == 1]
        return math.prod(packed_or_repeated_ints(dims)) if dims else 1

    init_names = set()
    for num, wt, v in fields(graph):
        if num == 1:  # NodeProto; exporters may store weights as Constant nodes
            op, consts = "", 0
            for nn, _, nv in fields(v):
                if nn == 4:
                    op = nv.decode()
                elif nn == 5:  # AttributeProto
                    for an, _, av in fields(nv):
                        if an == 5:  # t: TensorProto
                            consts += tensor_elems(av)
            info["ops"][op] += 1
            if op == "Constant":
                info["params"] += consts
        elif num == 5:  # TensorProto initializer
            name = ""
            for tn, _, tv in fields(v):
                if tn == 8:
                    name = tv.decode()
            init_names.add(name)
            info["initializers"] += 1
            info["params"] += tensor_elems(v)
        elif num == 11:
            name, dims = value_info(v)
            if name not in init_names:
                info["inputs"].append((name, dims))
        elif num == 12:
            info["outputs"].append(value_info(v))
    # Older exporters list initializers as graph inputs too; drop them.
    info["inputs"] = [(n, d) for n, d in info["inputs"] if n not in init_names]
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--markdown", action="store_true", help="op table as markdown")
    args = ap.parse_args()
    info = inspect(args.model)
    shape = lambda d: "[" + ", ".join(d) + "]"
    print(f"ir_version: {info['ir_version']}  opsets: {info['opsets']}")
    for n, d in info["inputs"]:
        print(f"input:  {n} {shape(d)}")
    for n, d in info["outputs"]:
        print(f"output: {n} {shape(d)}")
    print(f"parameters: {info['params']:,} (initializers: {info['initializers']}; "
          f"Constant nodes included)")
    ops = info["ops"].most_common()
    print(f"nodes: {sum(c for _, c in ops)}  distinct ops: {len(ops)}")
    if args.markdown:
        print("\n| op | count |\n|---|---:|")
        for op, c in ops:
            print(f"| {op} | {c} |")
    else:
        print("  " + ", ".join(f"{op}×{c}" for op, c in ops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
