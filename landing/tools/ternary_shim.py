# -*- coding: utf-8 -*-
"""Runtime shim: register the two Prism ternary formats into the Python-side
closed registries (tools.artifact.numeric / tools.artifact.layouts).

WHY THIS EXISTS
---------------
The released `changed-files` bundle ships the C++ side of the ternary port
(storage_layouts.cpp L49-55:  PTQ1_0_G128 = {group 128, base 24 B, high 2 B}
= 28 B/group;  PQ2_0_G128 = {128, 32 B, 0 B} = 34 B/group) but omits the
matching Python-side registration.  On a pristine v1.0.8 baseline tree,
pack.py therefore aborts with `unknown numeric format: 'PQ2_0_G128'` (the
baseline numeric.py is a *closed* registry, and layouts.py row-split geometry
only covers 4/5/6/8-bit codes).  The author's own runs used a development tree
whose Python side carries this registration; it was not part of the release.

This shim mirrors the published C++ semantics exactly and is meant to be
REPLACED by the author's official Python-side files once available.

WHAT IT PATCHES (three points, full validation chain)
-----------------------------------------------------
1. numeric.NUMERIC_FORMATS       -> + PQ2_0_G128 / PTQ1_0_G128 (TernaryFormat
                                    subclasses QuantFormat, so the
                                    `isinstance(..., QuantFormat)` gate in
                                    layouts.encoded_size passes).
2. layouts.ROW_SPLIT_K128_V1     -> formats whitelist extended with both names
                                    (encoded_size L279 checks membership).
3. layouts.row_split_geometry    -> ternary branch mirroring C++
                                    ResolveRowSplitLayout: offsets aligned to
                                    kTensorAlignment=256, K aligned to
                                    kKAlignment=128 (same constants as the
                                    baseline Python: PLANE_ALIGNMENT=256,
                                    K_ALIGNMENT=128).

GEOMETRY SANITY (cross-checked before writing)
----------------------------------------------
payload = rows * groups_per_row * (base + high + 2), groups_per_row = K/128:
  [248320, 5120] PQ2_0 -> 248320*40*34 = 337,715,200 B   (docs/03 §1.1)
  [248320, 5120] PTQ1_0 -> 248320*40*28 = 278,118,400 B  (docs/03 §1.1)

In-plane order is owned by pack.py's own producer (its byte assembly was the
author's verified artifact); this shim only unblocks geometry/length checks.

Usage (forwards everything to pack.py):
    python ternary_shim.py --template <tpl.ninfer> --gguf <PQ2_0.gguf> check
    python ternary_shim.py --template <tpl.ninfer> --gguf <PQ2_0.gguf> build <out.ninfer>
"""
from __future__ import annotations

import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

sys.stdout.reconfigure(encoding="utf-8")

BASELINE_ROOT = r"J:\Bonsai\landing\repos\ninfer-4090-windows"
PACK_PY = r"J:\Bonsai\landing\tools\ninfer-ada-ternary\tools\pack.py"

if BASELINE_ROOT not in sys.path:
    sys.path.insert(0, BASELINE_ROOT)

import tools.artifact.layouts as L  # noqa: E402
import tools.artifact.numeric as N  # noqa: E402


@dataclass(frozen=True, slots=True)
class TernaryFormat(N.QuantFormat):
    """Signed ternary codes with one binary16 multiplier per 128 group.

    Mirrors C++ NumericFormat::PQ2_0_G128 / PTQ1_0_G128
    (storage_layouts.cpp L49-55).
    """

    base_bytes_per_group: int
    high_bytes_per_group: int


PQ2_0_G128 = TernaryFormat(
    name="PQ2_0_G128", bits=2, group_size=128, qmin=-1, qmax=2,
    base_bytes_per_group=32, high_bytes_per_group=0,
)
PTQ1_0_G128 = TernaryFormat(
    name="PTQ1_0_G128", bits=2, group_size=128, qmin=-1, qmax=1,
    base_bytes_per_group=24, high_bytes_per_group=2,
)


def apply() -> None:
    # 1. numeric registry (closed MappingProxyType -> rebuild and rebind)
    N.PQ2_0_G128 = PQ2_0_G128
    N.PTQ1_0_G128 = PTQ1_0_G128
    N.NUMERIC_FORMATS = MappingProxyType({
        **N.NUMERIC_FORMATS,
        PQ2_0_G128.name: PQ2_0_G128,
        PTQ1_0_G128.name: PTQ1_0_G128,
    })

    # 2. row-split layout whitelist (Layout is frozen -> rebuild)
    old = L.ROW_SPLIT_K128_V1
    new_layout = L.Layout(
        old.name, old.alignment,
        old.formats | {PQ2_0_G128.name, PTQ1_0_G128.name},
    )
    L.ROW_SPLIT_K128_V1 = new_layout
    L.LAYOUTS = MappingProxyType({**L.LAYOUTS, new_layout.name: new_layout})

    # 3. geometry: ternary branch first, then delegate to the baseline impl
    original = L.row_split_geometry

    def row_split_geometry(fmt, shape):
        spec = fmt if not isinstance(fmt, str) else N.NUMERIC_FORMATS[fmt]
        if isinstance(spec, TernaryFormat):
            n, k = L._shape(shape, rank=2)
            k_pad = L.align_up(k, L.K_ALIGNMENT)
            if k_pad != k:
                raise ValueError(
                    "ternary row-split requires K already aligned to 128")
            gpr = k_pad // spec.group_size
            base_row = gpr * spec.base_bytes_per_group
            high_row = gpr * spec.high_bytes_per_group
            scale_row = gpr * 2
            base_bytes = n * base_row
            high_bytes = n * high_row
            scale_bytes = n * scale_row
            high_offset = L.align_up(base_bytes, L.PLANE_ALIGNMENT)
            scale_offset = high_offset + L.align_up(high_bytes, L.PLANE_ALIGNMENT)
            return L.RowSplitGeometry(
                n=n, k=k, k_pad=k_pad,
                groups_per_row=gpr,
                base_bytes_per_group=spec.base_bytes_per_group,
                high_bytes_per_group=spec.high_bytes_per_group,
                base_row_bytes=base_row,
                high_row_bytes=high_row,
                scale_row_bytes=scale_row,
                base_offset=0,
                base_bytes=base_bytes,
                high_offset=high_offset,
                high_bytes=high_bytes,
                scale_offset=scale_offset,
                scale_bytes=scale_bytes,
                payload_bytes=scale_offset + scale_bytes,
            )
        return original(fmt, shape)

    L.row_split_geometry = row_split_geometry
    print("[shim] ternary formats registered: PQ2_0_G128 (34 B/128), "
          "PTQ1_0_G128 (28 B/128); row-split whitelist + geometry patched")


def main() -> int:
    apply()
    args = sys.argv[1:]
    if args and args[0] == "--run":
        # generic mode: run any verify script inside the patched environment
        if len(args) < 2:
            print("usage: ternary_shim.py --run <script.py> [args...]")
            return 2
        script = args[1]
        script_dir = str(Path(script).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        code = compile(Path(script).read_text(encoding="utf-8"), script, "exec")
        g = {"__name__": "__main__", "__file__": script}
        sys.argv = [script] + args[2:]
        exec(code, g)
        return 0
    if not args:
        print("usage: ternary_shim.py [--run <script.py>] | "
              "[--template P] [--gguf P] {check|layer3 OUT|build OUT}")
        return 2

    # Load pack.py as a module (not runpy): its `from tools.artifact import
    # row_split_geometry` binds the *pre-patch* function object via the package
    # re-export, so after exec we rebind the module global to the patched one.
    import importlib.util

    spec = importlib.util.spec_from_file_location("ternary_pack", PACK_PY)
    pack_mod = importlib.util.module_from_spec(spec)
    sys.modules["ternary_pack"] = pack_mod
    assert spec.loader is not None
    spec.loader.exec_module(pack_mod)
    pack_mod.row_split_geometry = L.row_split_geometry

    sys.argv = [PACK_PY] + args
    return pack_mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
