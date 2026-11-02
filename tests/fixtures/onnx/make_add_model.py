#!/usr/bin/env python3
"""Write add.onnx: C = A + B over float32[1, 3] (opset 13, IR 8).

Encodes the ONNX protobuf by hand so no `onnx` Python package is needed; the
field numbers come from onnx/onnx.proto (ModelProto, GraphProto, NodeProto,
ValueInfoProto, TypeProto, TensorShapeProto, OperatorSetIdProto). Used only by
the ONNX Runtime smoke test; deterministic output.
"""

import sys
from pathlib import Path


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def field(num: int, payload) -> bytes:
    if isinstance(payload, int):  # wire type 0: varint
        return varint(num << 3) + varint(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return varint((num << 3) | 2) + varint(len(payload)) + payload  # wire type 2


def tensor_value_info(name: str, dims: list[int]) -> bytes:
    shape = b"".join(field(1, field(1, d)) for d in dims)  # dim { dim_value }
    tensor_type = field(1, 1) + field(2, shape)  # elem_type FLOAT=1, shape
    return field(1, name) + field(2, field(1, tensor_type))  # name, type{tensor_type}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("add.onnx")
    node = field(1, "A") + field(1, "B") + field(2, "C") + field(3, "add") + field(4, "Add")
    graph = (field(1, node) + field(2, "add_graph")
             + field(11, tensor_value_info("A", [1, 3]))
             + field(11, tensor_value_info("B", [1, 3]))
             + field(12, tensor_value_info("C", [1, 3])))
    model = (field(1, 8)                              # ir_version
             + field(2, "cudaocr-test")               # producer_name
             + field(7, graph)
             + field(8, field(1, "") + field(2, 13)))  # opset_import {domain "", version 13}
    out.write_bytes(model)
    print(f"wrote {out} ({len(model)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
