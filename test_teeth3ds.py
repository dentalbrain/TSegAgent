import os.path

import numpy as np
import json
import glob
import argparse
import pandas as pd

import tqdm
import trimesh

from fdi_classifier import MODELS
from metrics import miou_dice
from predictor import Predictor
from teeth3ds_evaluation import calculate_metrics


def compare(vertices, pred_json, gt_json):
    with open(pred_json, "r") as f:
        pred = json.load(f)
    with open(gt_json, "r") as f:
        gt = json.load(f)
    miou, dice = miou_dice(np.array(pred['labels']), np.array(gt['labels']), 48)

    pred['instances'] = pred['labels']
    gt['instances'] = gt['labels']
    gt['mesh_vertices'] = vertices
    try:
        jaw_TLA, jaw_TSA, jaw_TIR = calculate_metrics(gt, pred)
        jaw_TLA = np.exp(-jaw_TLA)
    except Exception as e:
        print(e)
        jaw_TLA, jaw_TSA, jaw_TIR = 0, 0, 0
    return miou, dice, jaw_TLA, jaw_TSA, jaw_TIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument("--force", "-f", action='store_true', required=False)
    parser.add_argument("--disable-gpt", action='store_true', required=False)
    parser.add_argument("--gpt-model", type=str, default=list(MODELS.keys())[0], choices=MODELS.keys())
    parser.add_argument("--split", "-s", choices=["Teeth3DS", "3DTeethSeg22_challenge", "3DTeethLand_challenge"],
                        type=str, default="Teeth3DS")
    args = parser.parse_args()

    # get testing sets
    testing_set_files = glob.glob(os.path.join(args.data_path, args.split + "_train_test_split", "*testing*"),
                                  recursive=True)
    testing_id = []
    for testing_set_file in testing_set_files:
        with open(testing_set_file, "r") as f:
            testing_id += [x.strip() for x in f.readlines() if x.strip() != '']
    print(f'Number of testing samples: {len(testing_id)}')

    performances = []

    predictor = Predictor(gpt_model=MODELS[args.gpt_model], use_gpt=not args.disable_gpt)
    os.makedirs(args.output_dir, exist_ok=True)
    with tqdm.tqdm(total=len(testing_id)) as pbar:
        for testing_case in testing_id:
            pbar.set_description(testing_case)
            name, jaw = testing_case.split('_')
            obj_path = os.path.join(args.data_path, "obj", name, f"{testing_case}.obj")
            gt_path = os.path.join(args.data_path, "obj", name, f"{testing_case}.json")
            pred_path = os.path.join(args.output_dir, f"{testing_case}.json")
            vis_path = os.path.join(args.output_dir, f"{testing_case}.ply")

            if args.force or not os.path.exists(pred_path):
                predictor.predict(obj_path)
                predictor.visualize(vis_path)
                predictor.export_json(pred_path)

            mesh = trimesh.load(obj_path, process=False)
            vertices = np.asarray(mesh.vertices)
            miou, dice, jaw_TLA, jaw_TSA, jaw_TIR = compare(vertices, pred_path, gt_path)
            pbar.set_postfix(miou=miou, dice=dice, TLA=jaw_TLA, TSA=jaw_TSA, TIR=jaw_TIR)

            performances.append([testing_case, miou, dice, jaw_TLA, jaw_TSA, jaw_TIR])
            pd.DataFrame(performances,
                         columns=['testing_case', 'miou', 'dice', 'TLA', 'TSA', 'TIR']).to_csv(
                os.path.join(args.output_dir, 'performances.csv'), index=False)

            pbar.update(1)
