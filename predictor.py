import json
import time

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from loguru import logger
import os

import torch

import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import postprocess
from fdi_classifier import generate_fdi_image, predict_fdi_from_images, MODELS
from mesh_renderer import MeshRenderer
from utils import model_curvature
from visualize.visualize import visualize_face_labels

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
sam3_root = os.path.join(os.path.dirname(sam3.__file__))
torch.inference_mode().__enter__()


# The 19 fixed views SAM3 segments from. They were tuned for a jaw whose occlusal surface faces
# +Z, i.e. a lower arch as scanners export it.
SELECTED_FRAMES = [
    [40, 0, 0],
    [30, 0, 0],
    [20, 0, 0],
    [10, 0, 0],
    [0, 0, 0],
    [-10, 0, 0],
    [-20, 0, 0],
    [-30, 0, 0],
    [-40, 0, 0],
    [30, 60, 180],
    [30, -60, 180],
    [30, 30, 180],
    [30, -30, 180],
    [-90, 0, 0],
    [-80, 0, 0],
    [-70, 0, 0],
    [100, 180, 0],
    [120, 180, 30],
    [120, 180, -30]
]

# Rigid poses tried before segmentation. Scanners export the upper arch occlusal-side down, and
# a jaw rendered from its gingival side leaves most faces unseen by every view: rid 466 upper
# reached only 32% face coverage as delivered and 80% after a half turn about X, the same as
# its lower (81%). Coverage is measured from the id maps alone (no SAM3), so trying all three
# costs seconds. Order matters: on a tie the earlier entry wins, and `identity` is kept whenever
# it is within ORIENTATION_TOLERANCE of the best so a good mesh is never flipped needlessly.
ORIENTATION_CANDIDATES = [
    ("identity", np.eye(3)),
    ("flip_x", np.diag([1.0, -1.0, -1.0])),
    ("flip_y", np.diag([-1.0, 1.0, -1.0])),
]
ORIENTATION_TOLERANCE = 0.02

class Predictor:
    def __init__(self, cache_path='./download', use_gpt=True, gpt_model=MODELS['chatgpt-5'], confidence_threshold=0.5):
        self.cache_path = cache_path
        self.use_gpt = use_gpt
        self.gpt_model = gpt_model

        os.environ["TRANSFORMERS_CACHE"] = self.cache_path
        os.environ["HF_HUB_CACHE"] = self.cache_path

        # Load the model
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.sam_model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(self.sam_model, confidence_threshold=confidence_threshold)

        self.output = None
        self.debug_output = {}
        os.makedirs('tmp', exist_ok=True)

        self.current_model_path = ''

        logger.info(f"SAM3 model loaded from {self.cache_path}")

    def predict(self, model_path, text_prompt='tooth', extra_frames=None, orientation='auto'):
        self.current_model_path = model_path
        # init
        self.output = None
        self.debug_output = {}

        mesh = trimesh.load(model_path, process=False)
        chosen, coverage_by_pose, orientation_seconds = self.select_orientation(mesh, orientation)
        self.debug_output["orientation"] = {
            "chosen": chosen,
            "coverage": coverage_by_pose,
            "seconds": round(orientation_seconds, 3),
        }
        if chosen != "identity":
            rotation = dict(ORIENTATION_CANDIDATES)[chosen]
            transform = np.eye(4)
            transform[:3, :3] = rotation
            mesh.apply_transform(transform)
        sam3_started = time.time()
        vn = np.concatenate([mesh.vertices, mesh.vertex_normals], axis=1)
        curv = model_curvature(vn)
        curv = (curv - curv.min()) / (curv.max() - curv.min())

        grey = np.array([0.8, 0.8, 0.8]) * 255
        red = np.array([1.0, 0, 0]) * 255
        mesh.visual.vertex_colors = (curv[:, None] * red + (1 - curv[:, None]) * grey).astype(np.uint8)

        renderer = MeshRenderer()
        renderer.set_mesh(mesh)
        renderer.render_config.vertexColors = True
        # set_mesh normalises the mesh to a unit radius; the output vertices are in that frame.
        # Callers that need millimetres (crown areas for the quality gate) scale back with this.
        self.debug_output["mesh_scale_mm"] = float(renderer.scale)
        face_coverage_set = np.zeros(len(renderer.mesh.faces) + 1)

        i = 0

        output_dict = {
            "vertices": renderer._mesh_vertices,
            "vertex_normals": renderer._mesh_vertex_normals,
            "faces": np.asarray(renderer.mesh.faces),
            "renders": [],
            "id_maps": [],
            "predicts": []
        }

        selected_frames = list(SELECTED_FRAMES)
        if extra_frames:
            selected_frames = selected_frames + list(extra_frames)
        for R in selected_frames:
            renderer.set_rotation(R)

            rendered, id_map = renderer.render_stack()
            output_dict["renders"].append(np.asarray(rendered))
            output_dict["id_maps"].append(id_map)

            face_coverage_set[np.unique(id_map[id_map > 0])] = 1

            inference_state = self.processor.set_image(rendered)
            self.processor.reset_all_prompts(inference_state)
            # Prompt the model with text
            output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            # Get the masks, bounding boxes, and scores
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

            output_dict["predicts"].append({
                "masks": masks.data.cpu().numpy(),
                "boxes": boxes.data.cpu().numpy(),
                "scores": scores.data.cpu().numpy(),
                "R": R
            })

            i += 1

        vertices, faces, face_labels = postprocess.postprocess(output_dict)
        self.output = {
            "vertices": vertices,
            "faces": faces,
            "face_labels": face_labels
        }

        self.debug_output["face_coverage"] = float(np.sum(face_coverage_set[1:] > 0) / len(renderer.mesh.faces))
        self.debug_output["sam3_seconds"] = round(time.time() - sam3_started, 3)

        if self.use_gpt:
            vlm_started = time.time()
            self._generate_fdi_predict_images(renderer)
            self.debug_output["vlm_seconds"] = round(time.time() - vlm_started, 3)

    def select_orientation(self, mesh, orientation='auto'):
        """Pick the rigid pose under which the fixed views see the most faces.

        Returns (name, {name: coverage}, seconds). `orientation` may be 'auto', a candidate
        name to force, or None to skip (identity, nothing measured).
        """
        if orientation is None:
            return "identity", {}, 0.0
        names = [name for name, _ in ORIENTATION_CANDIDATES]
        if orientation != 'auto':
            if orientation not in names:
                raise ValueError(f"unknown orientation {orientation!r}; expected one of {names}")
            return orientation, {}, 0.0
        started = time.time()
        renderer = MeshRenderer()
        coverage_by_pose = {}
        for name, rotation in ORIENTATION_CANDIDATES:
            candidate = mesh.copy()
            transform = np.eye(4)
            transform[:3, :3] = rotation
            candidate.apply_transform(transform)
            renderer.set_mesh(candidate)
            seen = np.zeros(len(candidate.faces) + 1, dtype=bool)
            for R in SELECTED_FRAMES:
                renderer.set_rotation(R)
                id_map = renderer.render_id_map()
                seen[np.unique(id_map[id_map > 0])] = True
            coverage_by_pose[name] = round(float(seen[1:].mean()), 4)
        best = max(names, key=lambda name: coverage_by_pose[name])
        chosen = "identity" if coverage_by_pose["identity"] >= coverage_by_pose[best] - ORIENTATION_TOLERANCE else best
        logger.info(f"orientation {chosen} coverage={coverage_by_pose}")
        return chosen, coverage_by_pose, time.time() - started

    def _generate_fdi_predict_images(self, renderer):
        vertices, faces, face_labels = self.output["vertices"], self.output["faces"], self.output["face_labels"]
        self.output["face_id"] = face_labels
        # Render the two VLM images here and keep them on the output, so a caller can cache
        # the SAM3 stage (instances + images) and re-run only the VLM stage for another model.
        images = list(generate_fdi_image(vertices, faces, face_labels, renderer))
        self.output["fdi_images"] = images
        face_labels_fdi, gpt_output = predict_fdi_from_images(
            vertices, faces, face_labels, renderer, gpt_model=self.gpt_model, images=images,
        )
        self.output["face_labels"] = face_labels_fdi
        self.output["gpt_output"] = gpt_output

    def visualize(self, path):
        if self.output is None:
            return

        vertices, faces, face_labels = self.output["vertices"], self.output["faces"], self.output["face_labels"]
        # if gpt is not used, the face labels are the original id, not FDI
        visualize_face_labels(vertices, faces, face_labels, is_fdi=self.use_gpt).export(path)

    def export_json(self, path):
        if self.output is None:
            return

        vertices, faces, face_labels = self.output["vertices"], self.output["faces"], self.output["face_labels"]
        face_id = self.output["face_id"] if self.use_gpt else np.array([])
        face_labels = np.asarray(face_labels, dtype=np.int32)
        num_vertices = len(vertices)
        vertex_labels = np.zeros(num_vertices, dtype=np.int32)

        # Collect adjacent face labels for each vertex (ignoring unlabeled 0)
        incident_labels = [[] for _ in range(num_vertices)]
        for face, label in zip(faces, face_labels):
            if label <= 0:
                continue
            for vid in face:
                incident_labels[int(vid)].append(int(label))

        # Determine vertex label by majority voting; keep 0 if no valid labels
        for vid, labels in enumerate(incident_labels):
            if not labels:
                continue
            uniq, counts = np.unique(labels, return_counts=True)
            vertex_labels[vid] = int(uniq[np.argmax(counts)])

        with open(path, 'w', encoding='utf-8') as fp:
            json.dump({"labels": vertex_labels.tolist(), "debug_info": self.debug_output, "face_id": face_id.tolist(), "gpt_output": self.output["gpt_output"] if self.use_gpt else None}, fp, ensure_ascii=False)

if __name__ == '__main__':
    predictor = Predictor()
    for case in ['GNR6QR3P_lower']:
        case_name, jaw = case.split('_')
        predictor.predict(f"/mnt/d/work/Teeth3DS/obj/{case_name}/{case}.obj")
        predictor.visualize(f'/mnt/d/work/Teeth3DS/predictions/{case}.ply')
        predictor.export_json(f'/mnt/d/work/Teeth3DS/predictions/{case}.json')
