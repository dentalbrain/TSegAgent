import json

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
from fdi_classifier import predict_fdi_from_images, MODELS
from mesh_renderer import MeshRenderer
from utils import model_curvature
from visualize.visualize import visualize_face_labels

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
sam3_root = os.path.join(os.path.dirname(sam3.__file__))
torch.inference_mode().__enter__()

class Predictor:
    def __init__(self, cache_path='./download', use_gpt=True, gpt_model=MODELS['chatgpt-5']):
        self.cache_path = cache_path
        self.use_gpt = use_gpt
        self.gpt_model = gpt_model

        os.environ["TRANSFORMERS_CACHE"] = self.cache_path
        os.environ["HF_HUB_CACHE"] = self.cache_path

        # Load the model
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.sam_model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(self.sam_model, confidence_threshold=0.5)

        self.output = None
        self.debug_output = {}
        os.makedirs('tmp', exist_ok=True)

        self.current_model_path = ''

        logger.info(f"SAM3 model loaded from {self.cache_path}")

    def predict(self, model_path, text_prompt='tooth'):
        self.current_model_path = model_path
        # init
        self.output = None
        self.debug_output = {}

        mesh = trimesh.load(model_path, process=False)
        vn = np.concatenate([mesh.vertices, mesh.vertex_normals], axis=1)
        curv = model_curvature(vn)
        curv = (curv - curv.min()) / (curv.max() - curv.min())

        grey = np.array([0.8, 0.8, 0.8]) * 255
        red = np.array([1.0, 0, 0]) * 255
        mesh.visual.vertex_colors = (curv[:, None] * red + (1 - curv[:, None]) * grey).astype(np.uint8)

        renderer = MeshRenderer()
        renderer.set_mesh(mesh)
        renderer.render_config.vertexColors = True
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

        selected_frames = [
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
        for R in selected_frames:
            renderer.set_rotation(R)

            rendered, id_map = renderer.render_stack()
            output_dict["renders"].append(np.asarray(rendered))
            output_dict["id_maps"].append(id_map)

            face_coverage_set[np.unique(id_map[id_map > -1])] = 1

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

        self.debug_output["face_coverage"] = np.sum(face_coverage_set > 0) / len(renderer.mesh.faces)
        # logger.info(f'Face coverage: {np.sum(face_coverage_set > 0) / len(renderer.mesh.faces)}')

        if self.use_gpt:
            self._generate_fdi_predict_images(renderer)

    def _generate_fdi_predict_images(self, renderer):
        vertices, faces, face_labels = self.output["vertices"], self.output["faces"], self.output["face_labels"]
        self.output["face_id"] = face_labels
        face_labels_fdi, gpt_output = predict_fdi_from_images(vertices, faces, face_labels, renderer, gpt_model=self.gpt_model)
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
