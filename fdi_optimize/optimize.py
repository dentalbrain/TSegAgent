from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

# Compatible with both package import and script execution
try:  # pragma: no cover
    from . import geometry
except ImportError:  # pragma: no cover
    import importlib

    geometry = importlib.import_module("geometry")  # type: ignore


def optimize(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    gpt_output: Optional[List[Dict]] = None
) -> np.ndarray:

    # -------- helpers --------
    def _to_int_list(v: Any) -> List[int]:
        if v is None:
            return []
        if isinstance(v, (int, np.integer)):
            return [int(v)]
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return [int(s)]
            s = s.replace("[", " ").replace("]", " ").replace(",", " ")
            out: List[int] = []
            for tok in s.split():
                if tok.lstrip("-").isdigit():
                    out.append(int(tok))
            return out
        if isinstance(v, (list, tuple, set)):
            out: List[int] = []
            for x in v:
                out.extend(_to_int_list(x))
            return out
        return []

    def _to_str_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip().lower()
            return [s] if s else []
        if isinstance(v, (list, tuple, set)):
            out: List[str] = []
            for x in v:
                out.extend(_to_str_list(x))
            return out
        return []

    # -------- input checks --------
    if vertices is None or faces is None or face_ids is None:
        raise ValueError("vertices/faces/face_ids must not be empty")
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    face_ids = np.asarray(face_ids)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices shape should be [N,3], but got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces shape should be [M,3], but got {faces.shape}")
    if face_ids.ndim != 1 or face_ids.shape[0] != faces.shape[0]:
        raise ValueError(f"face_ids shape should be [M] with M=faces.shape[0], but got {face_ids.shape}")

    if gpt_output is None:
        gpt_output = []

    # -------- assign labels back to faces --------
    final_labels = np.zeros_like(face_ids, dtype=np.int32)
    for item in gpt_output:
        tid = item["id"]
        fdis = _to_int_list(item["fdi"])
        if not fdis:
            continue
        fdi = min(fdis)
        final_labels = np.where(face_ids == int(tid), int(fdi), final_labels)
    final_labels[final_labels < 0] = 0
    return final_labels
