import numpy as np
import trimesh
from .anatomy_colors import AnatomyColors

ac = AnatomyColors()

def visualize_face_labels(vertices: np.ndarray, faces: np.ndarray, face_labels: np.ndarray, is_fdi: bool = False):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if is_fdi:
        mesh.visual.face_colors = ac.get_arr_tooth_color(face_labels)
    else:
        mesh.visual.face_colors = ac.get_color(face_labels)
    return mesh


def visualize_vertex_labels_bounds(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    is_fdi: bool = False
):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 1) First assign original label colors to all faces
    if is_fdi:
        face_colors = ac.get_arr_tooth_color(face_labels)
    else:
        face_colors = ac.get_color(face_labels)

    face_colors = np.asarray(face_colors)

    # 2) Find face adjacency: each row is a pair of adjacent faces (f0, f1) sharing an edge
    #    shape: (K, 2)
    adj = mesh.face_adjacency
    if adj is None or len(adj) == 0:
        # No adjacency, return directly (e.g., only one face)
        mesh.visual.face_colors = face_colors
        return mesh

    f0 = adj[:, 0]
    f1 = adj[:, 1]

    # 3) Boundary faces: any adjacent faces with different labels
    diff = face_labels[f0] != face_labels[f1]
    boundary_faces = np.unique(np.concatenate([f0[diff], f1[diff]], axis=0))

    boundary_rgba = np.array([0x33, 0x33, 0x33, 0xFF], dtype=np.uint8)

    # Compatible with face_colors being either RGB or RGBA
    if face_colors.ndim != 2 or face_colors.shape[0] != faces.shape[0]:
        raise ValueError("face_colors must be (num_faces, 3) or (num_faces, 4)")

    if face_colors.shape[1] == 3:
        # Add alpha channel
        alpha = np.full((face_colors.shape[0], 1), 0xFF, dtype=face_colors.dtype)
        face_colors = np.concatenate([face_colors, alpha], axis=1)

    face_colors[boundary_faces] = boundary_rgba

    mesh.visual.face_colors = face_colors
    return mesh, boundary_faces

def visualize_vertex_labels(vertices: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray, is_fdi: bool = False):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if is_fdi:
        mesh.visual.vertex_colors = ac.get_arr_tooth_color(vertex_labels)
    else:
        mesh.visual.vertex_colors = ac.get_color(vertex_labels)
    return mesh
