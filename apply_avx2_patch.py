"""Patch engine.cpp to use the AVX2 kernels.

Usage (run inside the llama3.2 folder, next to engine.cpp and avx2_kernels.h):
    python apply_avx2_patch.py

Makes engine.cpp.bak, then:
  * adds  #include "avx2_kernels.h"
  * makes matvec_q4 / matvec_q8 call the AVX2 versions when compiled with -mavx2 -mfma
Safe to run twice. Without -mavx2 -mfma the original scalar code still runs.
"""
import re
import shutil
import sys

PATH = "engine.cpp"

with open(PATH, "r", newline="", encoding="utf-8") as f:
    src = f.read()

if "avx2_kernels.h" in src:
    print("engine.cpp already patched, nothing to do.")
    sys.exit(0)

shutil.copyfile(PATH, PATH + ".bak")


def must_sub(pattern, repl, text):
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        sys.exit(f"Could not find anchor: {pattern}\nengine.cpp left unchanged (backup in {PATH}.bak).")
    return new


src = must_sub(r"static const int GS\b", '#include "avx2_kernels.h"\n\nstatic const int GS', src)

src = must_sub(
    r"(static void matvec_q4\([^)]*\)\s*\{)",
    r"\1\n#ifdef HAVE_AVX2_KERNELS\n    matvec_q4_avx2(y, x, w.q.data(), w.s.data(), out, in);\n    return;\n#endif",
    src,
)
src = must_sub(
    r"(static void matvec_q8\([^)]*\)\s*\{)",
    r"\1\n#ifdef HAVE_AVX2_KERNELS\n    matvec_q8_avx2(y, x, (const int8_t *)w.q.data(), w.s.data(), out, in);\n    return;\n#endif",
    src,
)

with open(PATH, "w", newline="", encoding="utf-8") as f:
    f.write(src)

print("Patched engine.cpp (backup: engine.cpp.bak).")
print("Rebuild with:  g++ -O3 -fopenmp -mavx2 -mfma -static engine.cpp -o engine.exe")
