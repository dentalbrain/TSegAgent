import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from openai import OpenAI
from loguru import logger

from fdi_optimize.optimize import optimize
from fdi_optimize import geometry
from visualize.anatomy_colors import AnatomyColors
from visualize.visualize import visualize_face_labels
from dotenv import load_dotenv

load_dotenv()

ac = AnatomyColors()

PROMPTS = [

    '''
    The images provided by the user show the complete dentition of the same patient from different viewing angles, rendered from five visible regions: oblique occlusal view (top-left), occlusal view (top-right), frontal view (middle-left), and two lateral views (middle-right, bottom-left). The first image has %d-%d ID regions annotated with black bounding boxes, arranged in dental arch order. The second image shows the same segmentation without text labels. The OBB volume of each tooth on the point cloud is: %s.
    In the following tasks, tooth volume is for reference only, as it may differ significantly from the actual size due to reasons such as gingival impaction.
    Identify significant non-tooth regions in the images, such as gingival papillae, oral cavity inner walls, and soft tissues. Return their IDs in JSON format: {"results": [<id>, ...]}. If none exist, return {"results": []}. Note: do not identify partial tooth masks as non-tooth regions. Gingival papillae and oral cavity inner walls are morphologically similar to teeth rather than large areas; please carefully examine the images to find them.
    ''',

    '''
    Task: Based on the non-tooth regions you identified, find the central incisors in the images and return their ID array in JSON format: {"results": [<id>, ...]}. If none exist, return {"results": []}.

    Strict determination rules (execute by priority):

    1. Midline anchor principle (highest priority)
    - The bilateral symmetric structure of the dental arch must be determined first
    - The model in the images may be rotated, so judgment should be based on the actual dental arch structure
    - Central incisors are the tooth or pair of teeth at the midline boundary between the left and right dentition

    2. Midline conditions (all must be satisfied):
    - If there are two teeth, they must be directly adjacent
    - No third tooth exists between them
    - Their boundary position corresponds to the symmetry axis of the dental arch's left-right structure
    - They should have similar dental morphological features (size, shape, etc.)
    ''',

    '''
    Using the central incisors and non-tooth region information you provided as reference, perform the following task.
    You need to analyze the morphology of each ID tooth one by one, their positions relative to other teeth on the dental arch, and their likely corresponding tooth types.
    Due to segmentation algorithm limitations, a single tooth may be split into multiple ID regions (IDs belonging to the same tooth need to be reported to me). Therefore, when analyzing tooth shape, prioritize what you see on the model rather than the ID labels, to avoid assigning multiple IDs of the same tooth to different teeth.
    Similarly, multiple teeth may be segmented as a single ID region; you should mark such regions as bad_id and set their FDI to none.
    For central incisors, they are usually located at the very front of the dental arch, with a distinct incisal edge and no prominent cusp. For lateral incisors, they are usually located on both sides of the central incisors, with a distinct incisal edge and no prominent cusp. For canines, there is typically one prominent cusp with a few minor ridges, and the crown surface bulges toward the labial side; they often resemble incisors, so you should classify teeth that look like canines as canines whenever possible. For premolars, there are generally two prominent cusps, and their overall size is similar to anterior teeth. For molars, there are generally 3 or more cusps, a large and rounded fossa, and several minor ridges. Molars are usually larger than premolars. For wisdom teeth, they typically do not appear in intraoral scan models, or are mostly impacted within the gingiva; they have a similar shape to molars, and should only be classified as wisdom teeth when there are 3 posterior molars on one side, with the 3rd being the wisdom tooth.
    For teeth with heavy wear, there are usually no prominent cusps, but the overall size and appearance remain relatively consistent.
    Additionally, you need to determine whether there are non-tooth regions. Typically, these regions are morphologically similar to teeth but may represent gingival papillae misidentified as teeth.
    Gingival papillae are wedge-shaped soft tissue projections of the gingiva in the interdental spaces. Their morphological features include: being located on the gingival soft tissue, closely attached to actual teeth, appearing triangular; typically positioned in the gaps between two teeth, not forming an independent "crown" outline, and having no occlusal surface.
    Finally, assign each ID an FDI label based on tooth type; for maxillary models, the left side of the image corresponds to the patient's left side (2*), and the right side corresponds to the patient's right side (1*); for mandibular models, the left side of the image corresponds to the patient's right side (4*), and the right side corresponds to the patient's left side (3*).
    <hint>
    When assigning the same FDI to different IDs, check whether these IDs are distributed on the **lingual and labial/buccal sides**; if these IDs are actually distributed along the dental arch direction, they are generally not the same tooth. Please prioritize adjusting the central incisor classification to make room on both sides of the arch.
    When assigning FDI labels, do not consider whether the count exceeds the 16-tooth limit. If two IDs potentially belong to two different teeth, assign them two different FDI labels.
    </hint>
    <formatting>
    Summarize your answer in a JSON format: {"results": [{"id": <ID>, "fdi": <FDI>, "type": "<incisor/canine/premolar/molar/wisdom/none>", "bad_id": <BOOL>, ...], "jaw": "<upper/lower>"}
    For teeth, each must have an FDI label. IDs belonging to the same tooth must share the same FDI label. IDs belonging to different teeth must not have duplicate FDI labels. For non-teeth (e.g., gingival papillae), the FDI label must be -1 and type must be none.
    FDI labels must correspond to tooth types: [incisor, *1/*2], [canine, *3], [premolar, *4/*5], [molar, *6/*7], [wisdom, *8].
    </formatting>
    ''',

    '''
    Your predicted central incisors %s most likely: %s. Please re-examine whether the central incisor assignment is reasonable, and return the final classification result in the required format.
    '''
]

PROMPT_NO_CONVERSATION = '''
    The images provided by the user show the complete dentition of the same patient from different viewing angles, rendered from five visible regions: oblique occlusal view (top-left), occlusal view (top-right), frontal view (middle-left), and two lateral views (middle-right, bottom-left). The first image has several ID regions annotated with black bounding boxes, arranged in dental arch order. The second image shows the same segmentation without text labels. You need to analyze the morphology of each ID tooth one by one, their positions relative to other teeth on the dental arch, and their likely corresponding tooth types.
    Due to segmentation algorithm limitations, a single tooth may be split into multiple ID regions (IDs belonging to the same tooth need to be reported to me). Therefore, when analyzing tooth shape, prioritize what you see on the model rather than the ID labels, to avoid assigning multiple IDs of the same tooth to different teeth.
    Similarly, multiple teeth may be segmented as a single ID region; you should mark such regions as bad_id and set their FDI to none.
    For central incisors, they are usually located at the very front of the dental arch, with a distinct incisal edge and no prominent cusp. For lateral incisors, they are usually located on both sides of the central incisors, with a distinct incisal edge and no prominent cusp. For canines, there is typically one prominent cusp with a few minor ridges, and the crown surface bulges toward the labial side; they often resemble incisors, so you should classify teeth that look like canines as canines whenever possible. For premolars, there are generally two prominent cusps, and their overall size is similar to anterior teeth. For molars, there are generally 3 or more cusps, a large and rounded fossa, and several minor ridges. Molars are usually larger than premolars. For wisdom teeth, they typically do not appear in intraoral scan models, or are mostly impacted within the gingiva; they have a similar shape to molars, and should only be classified as wisdom teeth when there are 3 posterior molars on one side, with the 3rd being the wisdom tooth.
    For teeth with heavy wear, there are usually no prominent cusps, but the overall size and appearance remain relatively consistent.
    Additionally, you need to determine whether there are non-tooth regions. Typically, these regions are morphologically similar to teeth but may represent gingival papillae misidentified as teeth.
    Gingival papillae are wedge-shaped soft tissue projections of the gingiva in the interdental spaces. Their morphological features include: being located on the gingival soft tissue, closely attached to actual teeth, appearing triangular; typically positioned in the gaps between two teeth, not forming an independent "crown" outline, and having no occlusal surface.
    Finally, assign each ID an FDI label based on tooth type; for maxillary models, the left side of the image corresponds to the patient's left side (2*), and the right side corresponds to the patient's right side (1*); for mandibular models, the left side of the image corresponds to the patient's right side (4*), and the right side corresponds to the patient's left side (3*).
    <hint>
    When assigning the same FDI to different IDs, check whether these IDs are distributed on the **lingual and labial/buccal sides**; if these IDs are actually distributed along the dental arch direction, they are generally not the same tooth. Please prioritize adjusting the central incisor classification to make room on both sides of the arch.
    When assigning FDI labels, do not consider whether the count exceeds the 16-tooth limit. If two IDs potentially belong to two different teeth, assign them two different FDI labels.
    </hint>
    <formatting>
    Summarize your answer in a JSON format: {"results": [{"id": <ID>, "fdi": <FDI>, "type": "<incisor/canine/premolar/molar/wisdom/none>", "bad_id": <BOOL>, ...], "jaw": "<upper/lower>"}
    For teeth, each must have an FDI label. IDs belonging to the same tooth must share the same FDI label. IDs belonging to different teeth must not have duplicate FDI labels. For non-teeth (e.g., gingival papillae), the FDI label must be -1 and type must be none.
    FDI labels must correspond to tooth types: [incisor, *1/*2], [canine, *3], [premolar, *4/*5], [molar, *6/*7], [wisdom, *8].
    </formatting>
'''


NON_TOOTH_PROMPT_WITHOUT_BBOX = '''
    The images provided by the user show the complete dentition of the same patient from different viewing angles, rendered from five visible regions: oblique occlusal view (top-left), occlusal view (top-right), frontal view (middle-left), and two lateral views (middle-right, bottom-left). The first image has %d-%d ID regions annotated with black bounding boxes, arranged in dental arch order. The second image shows the same segmentation without text labels.
    In the following tasks, tooth volume is for reference only, as it may differ significantly from the actual size due to reasons such as gingival impaction.
    Identify significant non-tooth regions in the images, such as gingival papillae, oral cavity inner walls, and soft tissues. Return their IDs in JSON format: {"results": [<id>, ...]}. If none exist, return {"results": []}. Note: do not identify partial tooth masks as non-tooth regions. Gingival papillae and oral cavity inner walls are morphologically similar to teeth rather than large areas; please carefully examine the images to find them.
'''

NON_TOOTH_PROMPT = PROMPTS[0]
INCISOR_PROMPT = PROMPTS[1]
CLS_PROMPT = PROMPTS[2]
CHECK_PROMPT = PROMPTS[3]

current_inst = None
client = None

MODELS = {
    "doubao-seed": "doubao-seed-1-8-251228",
    "chatgpt-5": "gpt-5.2"
}


def init_llm_client(gpt_model=MODELS["doubao-seed"]):
    global client, current_inst
    if current_inst is None or current_inst != gpt_model:
        current_inst = gpt_model
        if gpt_model == MODELS["doubao-seed"]:
            client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.environ.get("ARK_API_KEY"),
                timeout=1800,
            )
        elif gpt_model == MODELS["chatgpt-5"]:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        logger.info(f"LLM client for {gpt_model} inited")


def gpt_json_result_postprocess(vertices, faces, result_json, face_labels, is_lower):
    gpt_output = []
    if result_json is not None and 'results' in result_json:
        gpt_output = result_json['results']
    if result_json is not None and 'items' in result_json:
        gpt_output = result_json['items']
    output_dict = {
        "vertices": vertices,
        "faces": faces,
        "face_labels": face_labels,
        "gpt_output": gpt_output
    }
    np.savez_compressed('tmp/fdi.npz', **output_dict)
    return optimize(vertices, faces, face_labels, gpt_output, is_lower=is_lower)


def generate_fdi_image(vertices, faces, face_labels, renderer, frames_count=5):
    global CLS_PROMPT
    CLS_PROMPT = PROMPTS[1] if frames_count == 3 else PROMPTS[2]

    colored_mesh = visualize_face_labels(vertices, faces, face_labels)
    renderer.set_mesh(colored_mesh)
    renderer.render_config.vertexColors = True
    i = 0

    selected_frames = [
        [-20, 0, 0],
        [10, 40, 0],
        [-75, 0, 0]
    ]

    font = ImageFont.load_default(32)
    file_ids = []

    rendered_big = np.ones((800 * 2 if frames_count == 3 else 800 * 3, 1280 * 2, 3), dtype=np.uint8) * 255
    rendered_big_text = np.ones((800 * 2 if frames_count == 3 else 800 * 3, 1280 * 2, 3), dtype=np.uint8) * 255

    for R in selected_frames:
        renderer.set_rotation(R)

        rendered, id_map = renderer.render()
        face_label_map = np.zeros_like(id_map)

        x = i % 2
        y = i % (2 if frames_count == 3 else 3)
        rendered_big[y * 800:(y + 1) * 800, x * 1280:(x + 1) * 1280, :] = rendered

        draw = ImageDraw.Draw(rendered)

        # add id tag on image
        face_label_map[id_map > 0] = face_labels[id_map[id_map > 0] - 1]
        for tid in np.unique(face_label_map[face_label_map > 0]):
            text_center = np.mean(np.argwhere(face_label_map == tid), axis=0)
            draw.text((text_center[1], text_center[0]), str(tid), font=font, fill=(255, 255, 255), align='center',
                      stroke_fill=(0, 0, 0), stroke_width=2, anchor='mm')
            bb_l, bb_t, bb_r, bb_b = draw.textbbox((text_center[1], text_center[0]), str(tid), font=font,
                                                   align='center', stroke_width=2, anchor='mm')
            draw.rectangle([(bb_l - 4, bb_t - 4), (bb_r + 4, bb_b + 4)], outline=(0, 0, 0), width=2)

        rendered_big_text[y * 800:(y + 1) * 800, x * 1280:(x + 1) * 1280, :] = rendered
        i += 1

    return Image.fromarray(rendered_big), Image.fromarray(rendered_big_text)


def predict_fdi_from_images(vertices, faces, face_labels, renderer, gpt_model=MODELS["chatgpt-5"], is_lower=False,
                            images=None, use_conversation=True, use_bbox=True, ordered_arch=True):
    init_llm_client(gpt_model)
    file_ids = []

    file_key = 'file_id'
    if images is None or len(images) == 0:
        img1, img2 = generate_fdi_image(vertices, faces, face_labels, renderer)
        images = [img1, img2]
    for img in images:
        if isinstance(img, str) and not img.startswith('file:'):
            file_ids.append(img)
            file_key = 'image_url'
        else:
            path = 'tmp/fdi.png'
            if isinstance(img, str):
                path = img[len('file://'):]
            else:
                img.save(path)

            with open(path, 'rb') as fp:
                file = client.files.create(
                    file=fp,
                    purpose="user_data"
                )

                # Wait for the file to finish processing
                while (file.status == "processing"):
                    time.sleep(2)
                    file = client.files.retrieve(file.id)

                file_ids.append(file.id)
                print('uploaded ', file.id)

    face_labels_min = np.min(face_labels[face_labels > 0])
    face_labels_max = np.max(face_labels)

    tooth_vertices = [np.mean(vertices[faces[face_labels == i]], axis=1) for i in np.unique(face_labels[face_labels > 0])]
    tooth_obb_sizes = {
        i + 1: '{:.1f}'.format(np.prod(geometry.compute_oriented_bounding_box_size(verts))) for i, verts in enumerate(tooth_vertices)
    }
    
    gpt_output = {
        "output_text": '',
        "output": []
    }

    def request_once(request_id, file_ids, prompt):
        # feed the images to GPT for classification
        request_image_contents = [
            {
                "type": "input_image",
                file_key: file_id,
                "detail": "high"
            } for file_id in file_ids
        ]

        if gpt_model == MODELS["doubao-seed"]:
            if request_id is not None and request_id != '':
                time.sleep(0.5)
            print('requesting with request_id: ', request_id)
            response = client.responses.create(
                model=gpt_model,
                previous_response_id=request_id,
                store=True,
                input=[
                    {
                        "role": "user",
                        "content": request_image_contents + [
                            {"type": "input_text", "text": masked_prompt},
                        ],
                    }
                ],
                extra_body={
                    "thinking": {
                        "type": "enabled"
                    },
                    "reasoning": {
                        "effort": "medium"
                    }
                }
            )
        elif gpt_model == MODELS["chatgpt-5"]:
            service_tier = 'flex'
            try:
                response = client.responses.create(
                    model=gpt_model,
                    previous_response_id=request_id,
                    input=[
                        {
                            "role": "user",
                            "content": request_image_contents + [
                                {"type": "input_text", "text": prompt},
                            ],
                        }
                    ],
                    reasoning={"effort": "medium", "summary": "auto"},
                    service_tier=service_tier
                )
            except:
                service_tier = 'auto'
                response = client.responses.create(
                    model=gpt_model,
                    previous_response_id=request_id,
                    input=[
                        {
                            "role": "user",
                            "content": request_image_contents + [
                                {"type": "input_text", "text": prompt},
                            ],
                        }
                    ],
                    reasoning={"effort": "medium", "summary": "auto"},
                    service_tier=service_tier
                )
        else:
            raise ValueError(f"Unknown gpt model: {gpt_model}")

        gpt_output['output_text'] = response.output_text

        for out in response.output:
            if out.type == 'reasoning':
                for ct in out.summary:
                    gpt_output['output'].append({
                        "type": ct.type,
                        "text": ct.text
                    })
            elif out.type == 'message':
                for ct in out.content:
                    gpt_output['output'].append({
                        "type": ct.type,
                        "text": ct.text
                    })
        return response.id, response.output_text

    def parse_json(result):
        if result.__contains__('```json'):
            result = result.split('```json')[1].split('```')[0]
        try:
            result_json = json.loads(result)
        except:
            try:
                if result.__contains__('{"results":'):
                    start = result.rindex('{"results":')
                    result = result[start:]
                    result_json = json.loads(result.split('\n')[0])
                else:
                    raise
            except:
                logger.error("Failed to parse json return.\nOrigin:\n{}\ntoParse:\n{}\n".format(gpt_output, result))
                result_json = None
        return result_json


    if not use_conversation:
        prompt = PROMPT_NO_CONVERSATION
        _, response_text = request_once(None, file_ids, prompt)
    else:
        if not use_bbox or not ordered_arch:
            masked_prompt = NON_TOOTH_PROMPT_WITHOUT_BBOX % (face_labels_min, face_labels_max)
        else:
            masked_prompt = NON_TOOTH_PROMPT % (face_labels_min, face_labels_max, json.dumps(tooth_obb_sizes))

        exist_id = np.arange(face_labels_min, face_labels_max + 1)
        resp_id, response_text = request_once(None, file_ids, masked_prompt)
        non_tooth = parse_json(response_text)
        if non_tooth is not None:
            exist_id = np.setdiff1d(exist_id, non_tooth['results'])

        resp_id, response_text = request_once(resp_id, file_ids, INCISOR_PROMPT)

        maybe_incisors = parse_json(response_text)
        incisor_error_flag = False
        incisor_error_reason = ''
        try:
            if maybe_incisors is not None and len(maybe_incisors['results']) > 0:
                # First check size
                incisor_vertices = [np.mean(vertices[faces[face_labels == i]], axis=1) for i in maybe_incisors['results']]
                incisor_sizes = [np.prod(geometry.compute_oriented_bounding_box_size(verts)) for verts in incisor_vertices]
                if np.min(incisor_sizes) / np.max(incisor_sizes) < 0.5:
                    incisor_error_flag = True
                    incisor_error_reason += 'Central incisor size difference is too large; '

                incisor_idx = [int(np.argwhere(exist_id == i).squeeze()) for i in maybe_incisors['results']]
                if len(incisor_idx) == 2:
                    if abs(incisor_idx[1] - incisor_idx[0]) != 1:
                        incisor_error_flag = True
                pleft = np.min(incisor_idx)
                pright = len(exist_id) - np.max(incisor_idx) - 1
                if abs(pright - pleft) > 1:
                    incisor_error_flag = True
                    incisor_error_reason += 'Central incisors are not at the midline of the dental arch; '
        except:
            pass

        resp_id, response_text = request_once(resp_id, file_ids, CLS_PROMPT)

        if incisor_error_flag:
            resp_id, response_text = request_once(resp_id, file_ids, CHECK_PROMPT % (json.dumps(maybe_incisors['results']), incisor_error_reason))

    for file_id in file_ids:
        try:
            response = client.files.delete(
                file_id=file_id
            )
        except:
            continue

    # decide fdi
    result = response_text
    if result.__contains__('```json'):
        result = result.split('```json')[1].split('```')[0]
    try:
        result_json = json.loads(result)
    except:
        try:
            if result.__contains__('{"results":'):
                start = result.rindex('{"results":')
                result = result[start:]
                result_json = json.loads(result.split('\n')[0])
            else:
                raise
        except:
            logger.error("Failed to parse json return.\nOrigin:\n{}\ntoParse:\n{}\n".format(gpt_output, result))
            result_json = {
                "results": [
                    {"id": i, "fdi": i}
                    for i in np.unique(face_labels)
                ]
            }
    ## sample: {"results": [{"id": 1, "fdi": 22, "type": "incisor"}, {"id": 2, "fdi": 28, "type": "wisdom"}, {"id": 3, "fdi": 11, "type": "incisor"}, {"id": 4, "fdi": 18, "type": "wisdom"}, {"id": 5, "fdi": 15, "type": "premolar"}, {"id": 6, "fdi": 13, "type": "canine"}, {"id": 7, "fdi": 17, "type": "molar"}, {"id": 8, "fdi": 26, "type": "molar"}, {"id": 9, "fdi": 16, "type": "molar"}, {"id": 10, "fdi": 12, "type": "incisor"}, {"id": 11, "fdi": 23, "type": "canine"}, {"id": 12, "fdi": 27, "type": "molar"}, {"id": 13, "fdi": 21, "type": "incisor"}, {"id": 14, "fdi": 24, "type": "premolar"}, {"id": 15, "fdi": -1, "type": "none"}], "jaw": "upper"}
    face_labels_fdi = gpt_json_result_postprocess(vertices, faces, result_json, face_labels, is_lower=is_lower)
    face_labels_fdi[face_labels_fdi < 0] = 0
    return face_labels_fdi, gpt_output


