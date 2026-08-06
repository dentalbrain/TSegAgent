"""Deterministic false-positive cleanup for FDI-labelled mesh faces."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _components(face_indices: np.ndarray, faces: np.ndarray) -> tuple[int, np.ndarray]:
    edge_owner: dict[tuple[int, int], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    for local_index, face_index in enumerate(face_indices):
        triangle = faces[face_index]
        for left, right in ((0, 1), (1, 2), (2, 0)):
            first = int(triangle[left])
            second = int(triangle[right])
            edge = (first, second) if first < second else (second, first)
            owner = edge_owner.get(edge)
            if owner is None:
                edge_owner[edge] = local_index
            else:
                rows.extend((local_index, owner))
                columns.extend((owner, local_index))
    count = len(face_indices)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(count, count),
    )
    return connected_components(graph, directed=False)


def clean(
    vertices,
    faces,
    face_labels,
    *,
    gap_fraction: float = 0.08,
    maximum_secondary_share: float = 0.45,
    minimum_faces: int = 500,
):
    """Return labels with remote or tiny secondary components set to gingiva.

    The largest connected component is retained for each non-zero FDI label.
    A secondary component is removed when it is either smaller than
    ``minimum_faces`` or both below ``maximum_secondary_share`` and farther
    than ``gap_fraction`` of the complete mesh bounding-box diagonal.
    """

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    face_labels = np.asarray(face_labels)
    output = np.array(face_labels, copy=True)
    triangle_centroids = vertices[faces].mean(axis=1)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not np.isfinite(scale) or scale <= 0.0:
        return output, []

    report = []
    for label in np.unique(face_labels):
        if int(label) == 0:
            continue
        indices = np.flatnonzero(face_labels == label)
        component_count, component_labels = _components(indices, faces)
        if component_count < 2:
            continue
        sizes = np.bincount(component_labels)
        main = int(np.argmax(sizes))
        main_centroid = triangle_centroids[indices[component_labels == main]].mean(axis=0)
        for component in range(component_count):
            if component == main:
                continue
            component_indices = indices[component_labels == component]
            share = float(sizes[component] / sizes.sum())
            gap = float(
                np.linalg.norm(triangle_centroids[component_indices].mean(axis=0) - main_centroid)
                / scale
            )
            too_far = share < maximum_secondary_share and gap > gap_fraction
            too_small = int(sizes[component]) < minimum_faces
            if not (too_far or too_small):
                continue
            output[component_indices] = 0
            report.append(
                {
                    "label": int(label),
                    "faces": int(sizes[component]),
                    "share": round(share, 3),
                    "gap_pct": round(gap * 100.0, 1),
                    "reason": "far" if too_far else "tiny",
                }
            )
    return output, report
