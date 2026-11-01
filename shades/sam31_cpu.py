"""CPU port of the SAM 3.1 multiplex video predictor.

The upstream SAM 3.1 video path is written for CUDA only (`demo_model.cuda()`,
`torch.device("cuda")`, pinned-memory copies, a module-level
`torch.cuda.get_device_properties(0)` call ...). On a CUDA machine we use the
upstream package untouched. On a machine without CUDA we generate a patched
*copy* of the package in a cache folder and import that instead, so the
upstream clone in ``sam3/`` is never modified.

The rewrite is purely mechanical (device strings and CUDA-only helpers). The
model weights, architecture and inference logic are unchanged, so the masks
are the same SAM 3.1 masks, only slower to compute (bf16 autocast on CPU).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

# (regex, replacement) pairs applied to every .py file of the copied package.
# Each rule is narrow on purpose. The model is written around bf16 autocast
# (some modules cast to bf16 explicitly), so autocast is moved to the CPU
# instead of being dropped; modern x86 CPUs (AVX512-BF16 / AMX) run it fast.
_RULES: list[tuple[str, str]] = [
    (r'autocast\(device_type="cuda"', 'autocast(device_type="cpu"'),
    (r'torch\.device\("cuda"\)', 'torch.device("cpu")'),
    (r'(?<![_\w])device="cuda"', 'device="cpu"'),
    (r'\.get\("device", "cuda"\)', '.get("device", "cpu")'),
    (r'f"cuda:\{self\.gpu_id\}"', '"cpu"'),
    # tensor/module .cuda(...) -> .cpu()  (handles multi-line non_blocking kwargs)
    (r"\.cuda\(\s*(?:non_blocking\s*=\s*True\s*)?\)", ".cpu()"),
    # pinned memory needs an accelerator; plain CPU memory is equivalent here
    (r"\.pin_memory\(\)", ""),
    (r"pin_memory=True", "pin_memory=False"),
    # module-level / device-capability queries
    (r"torch\.cuda\.get_device_properties\(0\)\.major", "0"),
    (r"torch\.cuda\.current_device\(\)", "0"),
    (r"torch\.cuda\.empty_cache\(\)", "None"),
    (r"assert data\.is_cuda", "pass"),
]


# Rules applied to a single file only.
_FILE_RULES: dict[str, list[tuple[str, str]]] = {
    # Frames are stored as fp16. CPU autocast cannot mix fp16 inputs into
    # bf16-autocast ops (torch.stack -> "Unexpected floating ScalarType"), so
    # store them as bf16, the same width as fp16 and the dtype the model runs in.
    "model/io_utils.py": [(r"torch\.float16", "torch.bfloat16")],
}


def _source_root(upstream_pkg_dir: Path) -> Path:
    if not (upstream_pkg_dir / "model_builder.py").exists():
        raise FileNotFoundError(f"Not a sam3 package dir: {upstream_pkg_dir}")
    return upstream_pkg_dir


def _fingerprint(pkg_dir: Path) -> str:
    h = hashlib.sha1()
    h.update(repr((_RULES, _FILE_RULES)).encode())
    for p in sorted(pkg_dir.rglob("*.py")):
        h.update(p.relative_to(pkg_dir).as_posix().encode())
        h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()[:12]


def build_cpu_port(upstream_pkg_dir: Path, cache_dir: Path) -> Path:
    """Copy ``upstream_pkg_dir`` (the inner ``sam3`` package) to
    ``cache_dir/<hash>/sam3`` with the CPU rewrites applied.

    Returns the directory to put on ``sys.path``.
    """
    src = _source_root(upstream_pkg_dir)
    root = cache_dir / f"sam3_cpu_{_fingerprint(src)}"
    dst = root / "sam3"
    if (root / ".done").exists():
        return root
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    n_files = 0
    for py in dst.rglob("*.py"):
        text = py.read_text()
        new = text
        rel = py.relative_to(dst).as_posix()
        for pat, rep in _RULES + _FILE_RULES.get(rel, []):
            new = re.sub(pat, rep, new)
        if new != text:
            py.write_text(new)
            n_files += 1
    _patch_edt(dst)
    (root / ".done").write_text(f"patched {n_files} files\n")
    return root


def _patch_edt(dst: Path) -> None:
    """edt_triton is a Triton (GPU) kernel. Swap in an OpenCV distance
    transform with the same contract (B,H,W) -> float distances."""
    edt = dst / "model" / "edt.py"
    # The original module imports triton at top level, so replace it wholesale.
    edt.write_text(
        '''"""CPU port: OpenCV replacement for the Triton EDT kernel."""


def edt_triton(data):
    import cv2
    import numpy as np
    import torch

    arr = data.detach().cpu().numpy().astype(np.uint8)
    out = np.stack(
        [cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for m in arr]
    )
    return torch.from_numpy(out).to(data.device)
'''
    )


def import_sam3(repo_sam3_dir: Path, cache_dir: Path, force_cpu: bool | None = None):
    """Import and return the ``sam3.model_builder`` module.

    Uses the upstream package on CUDA machines and the generated CPU port
    otherwise. Must run before anything else imports ``sam3``.
    """
    import torch

    if force_cpu is None:
        force_cpu = not torch.cuda.is_available()
    if "sam3" in sys.modules:
        raise RuntimeError("sam3 already imported; call import_sam3() first")
    if force_cpu:
        root = build_cpu_port(repo_sam3_dir / "sam3", cache_dir)
        sys.path.insert(0, str(root))
    else:
        sys.path.insert(0, str(repo_sam3_dir))
    import sam3.model_builder as mb  # noqa: E402

    return mb
