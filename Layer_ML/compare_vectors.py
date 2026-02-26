"""Quick script to compare two .npy vectors."""
import numpy as np
import sys

p1 = sys.argv[1] if len(sys.argv) > 1 else "vector.npy"
p2 = sys.argv[2] if len(sys.argv) > 2 else "vector_2.npy"

v1 = np.load(p1)
v2 = np.load(p2)
print("vector.npy   shape:", v1.shape, "dtype:", v1.dtype)
print("vector_2.npy shape:", v2.shape, "dtype:", v2.dtype)

if v1.shape != v2.shape:
    print("Different shapes - compare up to min length")
    n = min(len(v1.ravel()), len(v2.ravel()))
    v1 = v1.ravel()[:n].astype(float)
    v2 = v2.ravel()[:n].astype(float)
else:
    v1 = v1.ravel().astype(float)
    v2 = v2.ravel().astype(float)

diff = v1 - v2
print()
print("--- Differences ---")
print("L2 (Euclidean) distance:", np.linalg.norm(diff))
print("Max |v1 - v2|:", np.max(np.abs(diff)))
print("Mean |v1 - v2|:", np.mean(np.abs(diff)))
print("Std of difference:", np.std(diff))
print("Non-zero in v1:", np.count_nonzero(v1), ", in v2:", np.count_nonzero(v2))
n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
if n1 > 1e-10 and n2 > 1e-10:
    cos = np.dot(v1, v2) / (n1 * n2)
    print("Cosine similarity:", cos)
print("All close (rtol=1e-5)?", np.allclose(v1, v2))
