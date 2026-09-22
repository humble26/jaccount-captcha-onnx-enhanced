import os as _os
import sys as _sys
_REPO = _os.environ.get("PROD_WS") or _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_VD = _os.path.join(_REPO, "vm_dump")
_NODE_MODULES = _os.environ.get("ORT_NODE_WORKSPACE") or _os.path.join(_REPO, "node_modules")
_NODE_BIN = _os.environ.get("NODE_BIN") or "node"
_PY_BIN = _os.environ.get("PY_BIN") or _sys.executable

import os

p = _os.path.join(_VD, "nn_model.onnx")
data = open(p, "rb").read()

def read_varint(buf, i):
    r = 0; s = 0
    while True:
        b = buf[i]; i += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80): break
        s += 7
    return r, i

def fields(buf):
    i = 0; n = len(buf)
    while i < n:
        key, i = read_varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(buf, i); yield fno, 0, v
        elif wt == 2:
            ln, i = read_varint(buf, i)
            yield fno, 2, buf[i:i+ln]; i += ln
        elif wt == 5:
            yield fno, 5, buf[i:i+4]; i += 4
        elif wt == 1:
            yield fno, 1, buf[i:i+8]; i += 8
        else:
            raise ValueError("wt %d" % wt)

def parse_type(b):
    """TypeProto -> (elem_type, [dims])"""
    et, dims = None, []
    for fno, wt, v in fields(b):
        if fno == 1 and wt == 2:  # tensor_type
            for f2, w2, v2 in fields(v):
                if f2 == 1 and w2 == 0:
                    et = v2
                elif f2 == 2 and w2 == 2:  # shape
                    for f3, w3, v3 in fields(v2):
                        if f3 == 1 and w3 == 2:  # dim
                            dv = "?"
                            for f4, w4, v4 in fields(v3):
                                if f4 == 1 and w4 == 0:
                                    dv = v4
                            dims.append(dv)
    return et, dims

ET = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 9: "bool", 10: "float16",
      11: "double", 12: "uint32", 13: "uint64"}

graph = None
for fno, wt, v in fields(data):
    if fno == 7 and wt == 2:
        graph = v
        break

print("=== GRAPH INPUTS ===")
out_names = []
node_out = []
node_op = {}
for fno, wt, v in fields(graph):
    if fno == 11 and wt == 2:
        name = "?"
        for f2, w2, v2 in fields(v):
            if f2 == 1 and w2 == 2:
                name = v2.decode("utf-8", "replace")
            elif f2 == 2 and w2 == 2:
                et, dims = parse_type(v2)
        print(f"  {name}  dtype={ET.get(et, et)}  shape={dims}")
    elif fno == 12 and wt == 2:
        name = "?"
        for f2, w2, v2 in fields(v):
            if f2 == 1 and w2 == 2:
                name = v2.decode("utf-8", "replace")
            elif f2 == 2 and w2 == 2:
                et, dims = parse_type(v2)
        print(f"  {name}  dtype={ET.get(et, et)}  shape={dims}   <-- OUTPUT")
        out_names.append((name, ET.get(et, et), dims))
    elif fno == 1 and wt == 2:  # node
        op, outs = None, []
        for f2, w2, v2 in fields(v):
            if f2 == 3 and w2 == 2:
                outs.append(v2.decode("utf-8", "replace"))
            elif f2 == 4 and w2 == 2:
                op = v2.decode("utf-8", "replace")
            elif f2 == 1 and w2 == 2:
                pass
        node_op[op] = node_op.get(op, 0) + 1
        node_out.extend(outs)

print("\n=== node op counts ===")
for k, c in sorted(node_op.items(), key=lambda x: -x[1]):
    print(f"  {k}: {c}")

print("\n=== last 12 node outputs ===")
for o in node_out[-12:]:
    print("  ", o)
