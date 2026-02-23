import numpy as np
import trimesh
from sklearn.neighbors import KDTree

def vertex_labels_to_face_labels(vertex_labels: np.ndarray, mesh: trimesh.Trimesh):
    """
    vertex_labels: np.ndarray, shape (N,)
    mesh: trimesh.Trimesh
    """
    face_labels = vertex_labels[mesh.faces]
    # Take the mode for each face
    face_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=face_labels)
    return face_labels.astype(int)


def face_labels_to_vertex_labels(face_labels: np.ndarray,
                                 mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Convert face labels to vertex labels using majority voting (NumPy only).

    Args:
        face_labels (np.ndarray): shape (F,), label per face
        mesh (trimesh.Trimesh): mesh with faces (F, 3)

    Returns:
        np.ndarray: shape (V,), label per vertex
    """
    num_vertices = mesh.vertices.shape[0]
    vertex_faces = [[] for _ in range(num_vertices)]

    # collect face labels for each vertex
    for f_idx, face in enumerate(mesh.faces):
        lbl = face_labels[f_idx]
        vertex_faces[face[0]].append(lbl)
        vertex_faces[face[1]].append(lbl)
        vertex_faces[face[2]].append(lbl)

    vertex_labels = np.empty(num_vertices, dtype=face_labels.dtype)

    for v_idx, labels in enumerate(vertex_faces):
        if len(labels) == 0:
            # isolated vertex (rare)
            vertex_labels[v_idx] = -1
        else:
            labels_np = np.asarray(labels)
            values, counts = np.unique(labels_np, return_counts=True)
            vertex_labels[v_idx] = values[np.argmax(counts)]

    return vertex_labels


def model_curvature(verts):
    tree = KDTree(verts[:, 0:3])
    neighbours = tree.query(verts[:, 0:3], 20, return_distance=False)
    norms = verts[neighbours][:, :, 3:]
    norms /= (np.sqrt(np.sum(norms ** 2, axis=-1, keepdims=True)) + 1e-6)
    mean_norm = np.mean(norms, axis=1, keepdims=True)
    curv = np.einsum('ijk,ikn->ijn', norms, mean_norm.transpose([0, 2, 1])).squeeze()
    curv = np.mean(np.arccos(np.clip(curv, -1, 1)), axis=-1)
    return curv
