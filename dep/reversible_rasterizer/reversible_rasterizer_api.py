import librevras as rr
import numpy as np
from sklearn.cluster import DBSCAN


class ReversibleRasterizer:
    """
    Reversible Rasterizer for rendering meshes using orthographic projection.
    Given a mesh under certain camera intrinsic and extrinsic, this class produces a pixel-to-face index map.
    """
    def __init__(self):
        self.instance = rr.init()
        self.camera_config = rr.CameraConfig()
        self.render_config = rr.RenderConfig()
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.id_map = None

    def set_camera(self, width: int, height: int, R: list = None, T: list = None, zoom: float = 1.0, near: float = 1e-4,
                   far: float = 1e4) -> 'ReversibleRasterizer':
        """
        Set orthographic camera parameters

        :param width: render width
        :param height: render height
        :param R: euler rotation in degrees
        :param T: translation in mm
        :param zoom: zoom factor
        :param near: near clip plane
        :param far: far clip plane
        :return:
        """
        if T is None:
            T = [0, 0, 0]
        if R is None:
            R = [1, 0, 0]
        self.camera_config.width = width
        self.camera_config.height = height
        self.camera_config.R = R
        self.camera_config.T = T
        self.camera_config.zoom = zoom
        self.camera_config.near = near
        self.camera_config.far = far
        return self

    def set_mesh(self, vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> 'ReversibleRasterizer':
        """
        Set the mesh data for the object.

        :param vertices: A numpy array of 3D vertices with shape (N, 3).
        :param normals: A numpy array of corresponding vertex normals with shape (N, 3).
        :param faces: A numpy array of faces, where each face is a list of vertex indices with shape (M, 3).
        """
        self.vertices = vertices
        self.faces = faces
        self.mesh = rr.Mesh(vertices, normals, faces)
        return self

    def render_id_map(self) -> 'ReversibleRasterizer':
        """
        Render the mesh using the current camera configuration.
        """
        self.id_map = np.array(rr.render_id_map(self.instance, self.mesh, self.camera_config))
        return self

    def render(self) -> np.ndarray:
        return np.asarray(rr.render(self.instance, self.mesh, self.camera_config, self.render_config)).astype(np.uint8)

    def render_normal(self) -> np.ndarray:
        return np.asarray(rr.render_normal(self.instance, self.mesh, self.camera_config, self.render_config)).astype(np.uint8)

    def render_depth(self) -> np.ndarray:
        return np.asarray(rr.render_depth(self.instance, self.mesh, self.camera_config, self.render_config)).astype(np.uint8)

    def clear_outliers(self, **kwargs) -> 'ReversibleRasterizer':
        id_map_flat = self.id_map.reshape(-1) - 1
        valid_mask = id_map_flat > -1
        id_map_u = id_map_flat[valid_mask]
        id_map_u, rev_index = np.unique(id_map_u, return_inverse=True)
        face_u = self.faces[id_map_u]
        points = np.mean(self.vertices[face_u], axis=1)

        model_args = {
            'eps': 0.25,
            'min_samples': 5
        }
        model_args.update(kwargs)
        clustering = DBSCAN(**model_args).fit_predict(points)
        id_map_u[clustering < 0] = -1
        id_map_u = id_map_u[rev_index]
        id_map_flat[valid_mask] = id_map_u
        self.id_map = (id_map_flat + 1).reshape(self.id_map.shape)
        return self

    def result(self) -> np.ndarray:
        return self.id_map

    def end(self):
        rr.end(self.instance)
