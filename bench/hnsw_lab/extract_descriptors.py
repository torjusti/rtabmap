#!/usr/bin/env python3
"""Extract visual word descriptors from an RTAB-Map database Word table
into a raw float32 binary file (N x dim, row-major) + a .meta text file.

Usage: extract_descriptors.py <db> <out.bin> [maxN]
"""
import sqlite3, sys, struct

db, out = sys.argv[1], sys.argv[2]
maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 0

con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cur = con.execute("SELECT descriptor_size, LENGTH(descriptor) FROM Word LIMIT 1")
dim, bloblen = cur.fetchone()
assert bloblen == dim * 4, f"expected float32 blobs, got dim={dim} bytes={bloblen}"

n = 0
with open(out, "wb") as f:
    q = "SELECT descriptor FROM Word ORDER BY id"
    for (blob,) in con.execute(q):
        f.write(blob)
        n += 1
        if maxn and n >= maxn:
            break
con.close()
with open(out + ".meta", "w") as f:
    f.write(f"count={n}\ndim={dim}\ndtype=float32\nsource={db}\n")
print(f"wrote {n} descriptors dim={dim} -> {out}")
