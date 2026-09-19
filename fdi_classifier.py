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
    The images provided by the user show a complete dentition of the same patient from different viewpoints. The visible regions are drawn from five views: a slightly top-down view (upper-left), occlusal view (upper-right), frontal view (middle-left), both lateral views (middle-right and lower-left). In the first image, several ID regions %d-%d have been marked with black boxes and arranged in dental-arch order; the second image shows the same segmentation without any text labels. The OBB volume of each tooth in the point cloud is: %s.
    In the following tasks, tooth volume is only for reference and may differ significantly from the true size due to factors such as being partially covered by gingiva.
    Identify conspicuous non-tooth regions in the image, such as gingival papillae, inner oral wall, soft tissues, etc., and return their IDs in JSON format: {"results": [<id>, ...]}. If none, return {"results": []}. Note: do not misidentify a region that could be a tooth (even partially) as a non-tooth region. Gingival papillae and the inner oral wall may look tooth-like rather than being large continuous areas; please carefully search by combining all views.

    ## Common mistakes:
    - Do not rely solely on volume. OBB volume is only a reference and not decisive; you must also judge tooth morphology from the images.
    - The model may include noise. If one tooth has an abnormally large OBB volume, but from multiple views it contains only a single tooth, then it is a normal tooth.
    ''',

    '''
    Task: Based on the non-tooth regions you identified, find the central incisors in the image and return their IDs as an array in JSON format: {"results": [<id>, ...]}. If none, return {"results": []}.

    Strict decision rules (apply in priority order):

    1. Midline anchor principle (highest priority)
    - You must first determine the left–right symmetric structure of the dental arch.
    - The model in the images may be rotated, so you must judge according to the actual arch structure.
    - Central incisors are the tooth or pair of teeth that lie on the midline dividing the left and right dentition; if there are more than two incisors, there must be a pair of central incisors.

    2. Midline conditions (must all be satisfied):
    - If there are two teeth, they must be directly adjacent.
    - There is no third tooth between them.
    - Their boundary corresponds to the symmetry axis of the left–right arch structure.
    - They should have similar morphological characteristics (size, shape, etc.).
    ''',

    '''
    Use the central incisors and non-tooth region information you provided as references to perform the following task.
    You now need to analyze, one by one, the morphology of each ID region, its position on the dental arch relative to other teeth, and the tooth type it likely corresponds to.

    ## Requirements:
    - Due to limitations of the segmentation algorithm, a single tooth may be split into multiple ID regions (you must tell me which IDs belong to the same tooth). Therefore, when analyzing tooth shape, prioritize the tooth you see on the model rather than relying purely on the ID labels, and avoid assigning different IDs of the same tooth to different teeth.
      Similarly, multiple teeth may be merged into a single ID region; you must treat such a region as a bad_id and set its FDI to one of the teeth it contains—do not set it to none.
    - For central incisors: they are usually at the very front of the arch, with a clear incisal edge and no prominent cusp tips.
      For lateral incisors: usually adjacent to the central incisors, with a clear incisal edge and no prominent cusp tips.
      For canines: typically have one relatively high cusp tip (plus a few minor, low prominences) and a labial bulge on the crown surface; they can resemble incisors, so you should classify teeth that look like canines as canines whenever possible.
      For premolars: generally have two relatively high cusps, and overall size similar to anterior teeth.
      For molars: generally have three or more cusps, larger and rounder fossae, and multiple low prominences; molars are typically larger than premolars.
      For wisdom teeth: usually do not appear in intraoral scan models, or are largely impacted under the gingiva; they resemble molars. Only when the number of posterior molars on a single side is 3 should the third posterior molar be classified as a wisdom tooth.
    - For heavily worn teeth: they may lack obvious cusp tips, but their overall size and appearance are relatively consistent.
    - In addition, you must determine whether there are any non-tooth regions. These regions often look tooth-like in shape, but may be gingival papillae incorrectly segmented as teeth.
    - Gingival papillae are wedge-shaped soft-tissue protrusions of gingiva between adjacent teeth. Morphologically, they sit on gingival soft tissue and closely adhere to real teeth, presenting a triangular shape; they are typically located in the gap between two teeth, do not form an independent “crown” outline, and have no occlusal surface.
      Some small-tooth regions may be very small and could be mistaken for gingival papillae. If there is truly a 3D protrusion with a cusp tip visible from multiple views, treat it as a tooth.
    - Finally, assign an FDI label to each ID based on tooth type. For maxillary (upper) models, the left side of the image corresponds to the patient’s left side (2*), and the right side corresponds to the patient’s right side (1*). For mandibular (lower) models, the left side of the image corresponds to the patient’s right side (4*), and the right side corresponds to the patient’s left side (3*).

    ## Typical mistakes:
    - Premolars and molars often differ greatly in morphology; use this to determine molar type. Be careful not to classify a second molar (*7) or first molar (*6) as a second premolar (*5)—this is a typical mistake.
    - Because canines (*3) are adjacent to premolars (*4)/*5 and lateral incisors (*2), misclassification is common. Judge whether it is a canine by whether the cutting edge is sharp and whether there are multiple cusp points.

    ## Tips:
    When assigning the same FDI to different IDs, check whether those IDs are distributed on the **lingual side and the labial/buccal side**; if the IDs are actually distributed along the arch direction, they are generally not the same tooth. In that case, prioritize adjusting the central incisor classification to make space for both sides of the arch.
    When assigning FDI labels, do not consider whether the total exceeds the limit of 16 teeth. If two IDs could plausibly belong to two different teeth, assign them two different FDIs.

    ## Output format:
    Summarize your answer as JSON in the format:
    {"results": [{"id": <ID>, "fdi": <FDI>, "type": "<incisor/canine/premolar/molar/wisdom/none>", "bad_id": <BOOL>, ...], "jaw": "<upper/lower>"}
    For teeth, an FDI label is required. IDs belonging to the same tooth must share the same FDI label. IDs belonging to different teeth must not reuse the same FDI label. For non-tooth regions (e.g., gingival papillae), the FDI label must be -1 and type must be none.
    The FDI label must match the tooth type, i.e.:
    [incisor, *1/*2], [canine, *3], [premolar, *4/*5], [molar, *6/*7], [wisdom, *8].
    ''',

    '''
    Your predicted central incisors %s are very likely: %s. Please re-check whether the central incisor assignment is reasonable, and return the final classification result according to the required format.
    '''
]

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

# Flex processing may legitimately be refused (HTTP 429 "Resource Unavailable",
# not charged) or time out. Retry the flex request on this schedule and then
# raise FlexUnavailableError; never silently switch service_tier.
FLEX_RETRY_BACKOFF_SECONDS = (20, 60, 120)
FLEX_UNAVAILABLE_MARKERS = ("resource_unavailable", "resource unavailable")


class FlexUnavailableError(RuntimeError):
    """Flex service tier could not serve the request after the retry schedule."""

    def __init__(self, message, last_error=None):
        super().__init__(message)
        self.last_error = last_error


def is_flex_unavailable_error(error):
    """True for a flex capacity refusal (429 resource unavailable) or a timeout.

    Matched defensively by class name, status_code and message so the module
    does not depend on a specific openai SDK version.
    """
    class_names = {cls.__name__ for cls in type(error).__mro__}
    if "APITimeoutError" in class_names or "TimeoutError" in class_names:
        return True
    status_code = getattr(error, "status_code", None)
    if status_code != 429 and "RateLimitError" not in class_names:
        return False
    parts = [str(getattr(error, "message", "") or ""), str(error), str(getattr(error, "code", "") or "")]
    body = getattr(error, "body", None)
    if body is not None:
        try:
            parts.append(json.dumps(body))
        except (TypeError, ValueError):
            parts.append(str(body))
    text = " ".join(parts).lower()
    return any(marker in text for marker in FLEX_UNAVAILABLE_MARKERS)


def create_flex_response(**create_kwargs):
    """Call client.responses.create with service_tier='flex', retrying only flex-unavailable errors."""
    last_error = None
    for attempt in range(len(FLEX_RETRY_BACKOFF_SECONDS) + 1):
        try:
            return client.responses.create(service_tier='flex', **create_kwargs)
        except Exception as error:
            if not is_flex_unavailable_error(error):
                raise
            last_error = error
            if attempt >= len(FLEX_RETRY_BACKOFF_SECONDS):
                break
            delay = FLEX_RETRY_BACKOFF_SECONDS[attempt]
            logger.warning(
                "flex tier unavailable (attempt {}/{}): {}; retrying in {}s",
                attempt + 1, len(FLEX_RETRY_BACKOFF_SECONDS) + 1, error, delay,
            )
            time.sleep(delay)
    raise FlexUnavailableError(
        "flex service tier unavailable after {} attempts: {}".format(
            len(FLEX_RETRY_BACKOFF_SECONDS) + 1, last_error),
        last_error,
    ) from last_error


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


def gpt_json_result_postprocess(vertices, faces, result_json, face_labels):
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
    return optimize(vertices, faces, face_labels, gpt_output)


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


def predict_fdi_from_images(vertices, faces, face_labels, renderer, gpt_model=MODELS["chatgpt-5"], images=None):
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
        "output": [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "requests": []
        }
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
            response = create_flex_response(
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
            )
        else:
            raise ValueError(f"Unknown gpt model: {gpt_model}")

        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            input_details = getattr(usage, "input_tokens_details", None)
            output_details = getattr(usage, "output_tokens_details", None)
            cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
            reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
            totals = gpt_output["usage"]
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["total_tokens"] += total_tokens
            totals["cached_input_tokens"] += cached_tokens
            totals["reasoning_tokens"] += reasoning_tokens
            totals["requests"].append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cached_input_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
                "service_tier": getattr(response, "service_tier", None),
            })

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
    face_labels_fdi = gpt_json_result_postprocess(vertices, faces, result_json, face_labels)
    face_labels_fdi[face_labels_fdi < 0] = 0
    return face_labels_fdi, gpt_output
