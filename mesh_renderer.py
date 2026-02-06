import numpy as np
import trimesh
import librevras as rr
from PIL import Image


class MeshRenderer:
    def __init__(self, mesh_path='', normalize=True):
        self.bbox_min = None
        self.bbox_max = None
        self.bbox_size = None
        self.centroid = None
        self.scale = 1.0
        self.rrmesh = None
        self._mesh_vertices = None
        self._mesh_vertex_normals = None

        self.normalize = normalize
        if mesh_path is not None and mesh_path != '':
            self.mesh = trimesh.load(mesh_path, process=False)
            self.set_mesh(self.mesh)
        else:
            self.mesh = None

        self.camera_config = rr.CameraConfig()
        self.camera_config.width = 1280
        self.camera_config.height = 800
        self.camera_config.R = [0, 0, 0]
        self.camera_config.T = [0, 0, 1]
        self.camera_config.zoom = 600
        self.camera_config.near = 0.0001
        self.camera_config.far = 1000
        self.render_config = rr.RenderConfig()
        self.render_config.specularStrength = 0.1
        self.instance = rr.init()

        self.R = [0, 0, 0]

    @staticmethod
    def _rotation_matrix_from_euler(euler_deg):
        rx, ry, rz = np.deg2rad(euler_deg)

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0],
                          [0, cx, -sx],
                          [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy],
                          [0, 1, 0],
                          [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0],
                          [sz, cz, 0],
                          [0, 0, 1]])

        # 先绕 x，再 y，最后 z，匹配 camera_config 的欧拉角约定
        return rot_x @ rot_y @ rot_z

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """
        R: 3x3 旋转矩阵 (numpy array)
        返回: [roll, pitch, yaw]，单位：弧度
        约定: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        """

        # sy 用来判断是否接近万向节锁（pitch ~ ±90°）
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            # roll (X)
            roll = np.arctan2(R[2, 1], R[2, 2])
            # pitch (Y)
            pitch = np.arctan2(-R[2, 0], sy)
            # yaw (Z)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # 万向节锁时的处理：yaw 和 roll 纠缠在一起，只能固定其中一个
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return np.degrees([roll, pitch, yaw])

    def _compute_orthographic_zoom(self):
        rotation_matrix = self._rotation_matrix_from_euler(self.R)
        rotated_vertices = self._mesh_vertices @ rotation_matrix.T
        rotated_vertex_normals = self._mesh_vertex_normals @ rotation_matrix.T

        projected_xy = rotated_vertices[:, :2]
        bbox_min = projected_xy.min(axis=0)
        bbox_max = projected_xy.max(axis=0)
        bbox_size = bbox_max - bbox_min

        eps = 1e-6
        width_scale = self.camera_config.width / max(bbox_size[0], eps)
        height_scale = self.camera_config.height / max(bbox_size[1], eps)
        return rotated_vertices, rotated_vertex_normals, min(width_scale, height_scale) * 0.95, (bbox_max + bbox_min) / 2

    def set_mesh(self, mesh):
        self.mesh = mesh

        # use bbox for normalization
        self.bbox_min = np.min(self.mesh.vertices, axis=0)
        self.bbox_max = np.max(self.mesh.vertices, axis=0)
        self.bbox_size = self.bbox_max - self.bbox_min

        self.centroid = (self.bbox_min + self.bbox_max) / 2
        self.scale = np.sqrt(np.max(np.sum((self.mesh.vertices - self.centroid) ** 2, axis=1)))
        if self.normalize:
            self.mesh.apply_translation(-self.centroid)
            self.mesh.apply_scale(1.0 / self.scale)

        if hasattr(self.mesh, 'visual') and hasattr(self.mesh.visual, 'vertex_colors'):
            vc = self.mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            self.rrmesh = rr.Mesh(np.asarray(self.mesh.vertices), np.asarray(self.mesh.vertex_normals), np.asarray(self.mesh.faces), vc)
        else:
            self.rrmesh = rr.Mesh(np.asarray(self.mesh.vertices), np.asarray(self.mesh.vertex_normals), np.asarray(self.mesh.faces))
        self._mesh_vertices = np.asarray(self.mesh.vertices)
        self._mesh_vertex_normals = np.asarray(self.mesh.vertex_normals)

        rr.setup_mesh(self.rrmesh, self.camera_config)

    def set_rotation(self, R):
        self.R = R
        self.render_config.R = R

    def render_stack(self, alpha: float = 0.8):
        rotated_vertices, rotated_vertex_normals, self.camera_config.zoom, T = self._compute_orthographic_zoom()
        self.camera_config.T = [T[0], T[1], 1]

        id_map = np.array(rr.render_id_map(self.instance, self.rrmesh, self.camera_config, self.render_config))

        rendered = np.array(rr.render_stacked(self.instance, self.rrmesh, self.camera_config, self.render_config, 3, [alpha, 1 - alpha]))

        id_map[id_map > len(self.mesh.faces)] = -1
        return Image.fromarray(rendered.astype(np.uint8)), id_map

    def render(self):
        rotated_vertices, rotated_vertex_normals, self.camera_config.zoom, T = self._compute_orthographic_zoom()
        self.camera_config.T = [T[0], T[1], 1]

        id_map = np.array(rr.render_id_map(self.instance, self.rrmesh, self.camera_config, self.render_config))
        rendered = np.array(rr.render(self.instance, self.rrmesh, self.camera_config, self.render_config))

        id_map[id_map > len(self.mesh.faces)] = -1
        return Image.fromarray(rendered.astype(np.uint8)), id_map

    def __del__(self):
        rr.end_with_mesh(self.instance, self.rrmesh)

if __name__ == '__main__':
    renderer = MeshRenderer('./data/ios.ply')

    i = 0

    face_coverage_set = np.zeros(len(renderer.mesh.faces) + 1)
    selected_frames = [
        [0, 0, 180],
        [-40, 0, 180],
        [10, 40, 180],
        [10, -40, 180],
        [-90, 0, 0]
    ]
    for R in selected_frames:
        renderer.set_rotation(R)
        rendered, id_map = renderer.render()
        rendered.save(f'./input/{i:05d}.jpg')
        face_coverage_set[np.unique(id_map[id_map > -1])] = 1
        i += 1
    print('Face coverage:', np.sum(face_coverage_set > 0) / len(renderer.mesh.faces))
