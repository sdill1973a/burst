"""burst_codec — the Burst codec as an installable package (Apache-2.0).

The codec modules are flat (det_codec, open_codec, mrc, repro) and import each other by name, exactly as in the
source tree; importing this package puts them on the path so `import det_codec` works after `pip install`.
Frame 0x08 = det_codec (deterministic, the default); 0x03 = open_codec (mismatch-resistant, any GGUF).
Both ends need the same model file: export BURST_GGUF=/path/to/Qwen2.5-0.5B.Q8_0.gguf (sha256 d5163baa…).
"""
import os as _os, sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

__version__ = "0.2.0"

# libdetlm.so (the bit-identical engine det_codec loads) is built once, here, with the same determinism flags as
# Makefile.detlm — never -ffast-math, never -march=native, -ffp-contract=off is load-bearing. Needs a C compiler.
_SO = _os.path.join(_HERE, "libdetlm.so")
if not _os.path.exists(_SO):
    import subprocess as _sp
    try:
        _sp.run([_os.environ.get("CC", "cc"), "-O2", "-ffp-contract=off", "-fno-math-errno", "-fno-trapping-math", "-std=c11",
                 "-fPIC", "-shared", "-o", _SO, _os.path.join(_HERE, "detlm.c"), "-lpthread", "-lm"], check=True,
                capture_output=True)
    except (OSError, _sp.CalledProcessError) as _e:
        _sys.stderr.write(f"burst_codec: could not build libdetlm.so ({_e}); frame 0x08 on the real model needs a C compiler\n")
