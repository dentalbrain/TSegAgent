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
    用户给出的图像是同一患者不同视角的完整的牙列图像，分别从偏顶面（左上）、咬合面（右上）、正面（左中）、两侧（右中、左下）五处绘制了可见区域，第一张已用黑色方框标注 %d-%d 若干个ID区域，按照牙弓顺序排列，第二张是不带文本标记的相同分割。每颗牙齿在点云上的OBB体积为：%s。
    接下来的任务中，牙齿体积仅用作参考，会因为被牙龈埋伏等原因导致与真实大小差异较大。
    找出图中牙龈乳头、口腔内壁、软组织等显著的非牙齿区域，以json格式返回他们的ID:{"results": [<id>, ...]}，如果没有，则返回{"results": []}。注意，不要将可能是牙齿的局部mask识别为非牙齿区域。牙龈乳头、口腔内壁等在形态上类似牙齿，而不是大片的区域，请结合图像仔细寻找。
    ''',

    '''
    任务：结合你判断的非牙齿区域，找出图中的中切牙，以json格式返回他们的ID数组: {"results": [<id>, ...]}。如果不存在，则返回{"results": []}。

    严格判定规则（按优先级执行）：

    1. 中线锚点原则（最高优先级）
    - 必须首先确定牙弓的左右对称结构
    - 图片中的模型可能存在旋转，因此需要根据实际牙弓结构进行判断
    - 中切牙是左右牙列在中线处分界的那一颗或一对牙

    2. 中线条件（必须同时满足）：
    - 若是两颗牙，则两颗牙直接相邻
    - 其间不存在第三颗牙
    - 它们的分界位置对应牙弓左右结构的对称轴
    - 它们应具有相似的牙齿形态特征（大小、形状等）
    ''',

    '''
    根据你给出的中切牙和非牙齿区域信息为参考，执行以下任务。
    现在你需要逐一分析每个ID牙齿的形态，他们在牙弓上相对于其他牙齿的位置，以及他们可能对应的牙齿类型。
    由于分割算法限制，可能会将一颗牙齿分出多个ID区域标记（属于同一颗牙齿的ID需要告诉我），因此在分析牙齿形状时，优先根据你看到的模型上的牙齿、而不是以ID标记为依据，避免将属于一颗牙齿的几个ID分到不同的牙齿上。
    同理，也可能会将多颗牙齿分割为一个ID区域，你需要将这一区域认定为bad_id，并将其FDI设为none。
    对于中切牙，他们通常位于牙弓最前端，有较明显的切缘，没有明显的牙尖突起。对于侧切牙，他们通常位于中切牙两侧，有较明显的切缘，没有明显的牙尖突起。对于尖牙，通常只有一处较高的尖牙突起，以及少许不高的突起，且会向唇侧有鼓起的牙冠面；它们通常会与切牙形似，所以你需要尽可能地将像尖牙的牙齿分类为尖牙。对于前磨牙，一般有两处较高突起，且整体大小与前牙相近。对于后磨牙，一般有3处以上的牙尖，较大较圆的牙窝，以及若干不高的突起。后磨牙通常会比前磨牙更大。对于智齿，通常在口扫模型中不会出现，或大部分埋伏在牙龈内；他们与后磨牙具有相似的外形，且判定时，只有单侧后磨牙的数量为3时，第3后磨牙才判定为智齿。
    对于有较重磨损的牙齿，通常不会有明显的牙尖突起，但整体的大小及外观相对固定。
    此外，你需要判定是否有非牙齿的区域。通常，这些区域在外形上近似牙齿，但可能将牙龈乳头识别为牙齿。
    牙龈乳头是牙龈在牙齿相邻间隙的楔形软组织突起，其形态特点是位于牙龈软组织上，与真实的牙齿紧密贴合，呈现三角状；位置通常在两颗牙齿之间的间隙中，不形成独立的“牙冠”轮廓，无咬合面。
    最后，根据牙齿类型赋予每个ID一个FDI标签；对于上颌模型，画面左侧对应患者的左侧（2*），画面右侧对应患者的右侧（1*）；对于下颌模型，画面左侧对应患者右侧（4*），画面右侧对应患者左侧（3*）。
    <hint>
    在为不同ID赋予相同FDI时，请检查这些ID是否分布于**舌侧和唇/颊侧**；若这些ID实际上沿着牙弓方向分布，则他们一般不是同一颗牙齿，请优先调整中切牙的分类，以腾出两侧牙弓的空间。
    在赋予FDI标签时，请不要考虑是否超出了16颗牙齿的数量限制，如果两个ID有可能属于两颗牙齿，则为他们赋予两个不同的FDI。
    </hint>
    <formatting>
    将你的回答总结成一个json返回给我，格式为：{"results": [{"id": <ID>, "fdi": <FDI>, "type": "<incisor/canine/premolar/molar/wisdom/none>", "bad_id": <BOOL>, ...], "jaw": "<upper/lower>"}
    对于牙齿，它必须拥有一个FDI标签。对于属于相同牙齿的ID，他们的FDI标签必须相同。对于属于不同牙齿的ID，他们的FDI标签不能重复。对于非牙齿（如牙龈乳头），他们的FDI标签必须为-1，且type为none。
    FDI标签必须与牙齿类型对应，即：[incisor, *1/*2], [canine, *3], [premolar, *4/*5], [molar, *6/*7], [wisdom, *8].
    </formatting>
    ''',

    '''
    你预测的中切牙 %s 很可能：%s。请重新检查中切牙分配是否合理，并按照格式要求返回最终的分类结果。
    '''
]

PROMPT_NO_CONVERSATION = '''
    用户给出的图像是同一患者不同视角的完整的牙列图像，分别从偏顶面（左上）、咬合面（右上）、正面（左中）、两侧（右中、左下）五处绘制了可见区域，第一张已用黑色方框标注若干个ID区域，按照牙弓顺序排列，第二张是不带文本标记的相同分割。现在你需要逐一分析每个ID牙齿的形态，他们在牙弓上相对于其他牙齿的位置，以及他们可能对应的牙齿类型。
    由于分割算法限制，可能会将一颗牙齿分出多个ID区域标记（属于同一颗牙齿的ID需要告诉我），因此在分析牙齿形状时，优先根据你看到的模型上的牙齿、而不是以ID标记为依据，避免将属于一颗牙齿的几个ID分到不同的牙齿上。
    同理，也可能会将多颗牙齿分割为一个ID区域，你需要将这一区域认定为bad_id，并将其FDI设为none。
    对于中切牙，他们通常位于牙弓最前端，有较明显的切缘，没有明显的牙尖突起。对于侧切牙，他们通常位于中切牙两侧，有较明显的切缘，没有明显的牙尖突起。对于尖牙，通常只有一处较高的尖牙突起，以及少许不高的突起，且会向唇侧有鼓起的牙冠面；它们通常会与切牙形似，所以你需要尽可能地将像尖牙的牙齿分类为尖牙。对于前磨牙，一般有两处较高突起，且整体大小与前牙相近。对于后磨牙，一般有3处以上的牙尖，较大较圆的牙窝，以及若干不高的突起。后磨牙通常会比前磨牙更大。对于智齿，通常在口扫模型中不会出现，或大部分埋伏在牙龈内；他们与后磨牙具有相似的外形，且判定时，只有单侧后磨牙的数量为3时，第3后磨牙才判定为智齿。
    对于有较重磨损的牙齿，通常不会有明显的牙尖突起，但整体的大小及外观相对固定。
    此外，你需要判定是否有非牙齿的区域。通常，这些区域在外形上近似牙齿，但可能将牙龈乳头识别为牙齿。
    牙龈乳头是牙龈在牙齿相邻间隙的楔形软组织突起，其形态特点是位于牙龈软组织上，与真实的牙齿紧密贴合，呈现三角状；位置通常在两颗牙齿之间的间隙中，不形成独立的“牙冠”轮廓，无咬合面。
    最后，根据牙齿类型赋予每个ID一个FDI标签；对于上颌模型，画面左侧对应患者的左侧（2*），画面右侧对应患者的右侧（1*）；对于下颌模型，画面左侧对应患者右侧（4*），画面右侧对应患者左侧（3*）。
    <hint>
    在为不同ID赋予相同FDI时，请检查这些ID是否分布于**舌侧和唇/颊侧**；若这些ID实际上沿着牙弓方向分布，则他们一般不是同一颗牙齿，请优先调整中切牙的分类，以腾出两侧牙弓的空间。
    在赋予FDI标签时，请不要考虑是否超出了16颗牙齿的数量限制，如果两个ID有可能属于两颗牙齿，则为他们赋予两个不同的FDI。
    </hint>
    <formatting>
    将你的回答总结成一个json返回给我，格式为：{"results": [{"id": <ID>, "fdi": <FDI>, "type": "<incisor/canine/premolar/molar/wisdom/none>", "bad_id": <BOOL>, ...], "jaw": "<upper/lower>"}
    对于牙齿，它必须拥有一个FDI标签。对于属于相同牙齿的ID，他们的FDI标签必须相同。对于属于不同牙齿的ID，他们的FDI标签不能重复。对于非牙齿（如牙龈乳头），他们的FDI标签必须为-1，且type为none。
    FDI标签必须与牙齿类型对应，即：[incisor, *1/*2], [canine, *3], [premolar, *4/*5], [molar, *6/*7], [wisdom, *8].
    </formatting>
'''


NON_TOOTH_PROMPT_WITHOUT_BBOX = '''
    用户给出的图像是同一患者不同视角的完整的牙列图像，分别从偏顶面（左上）、咬合面（右上）、正面（左中）、两侧（右中、左下）五处绘制了可见区域，第一张已用黑色方框标注 %d-%d 若干个ID区域，按照牙弓顺序排列，第二张是不带文本标记的相同分割。
    接下来的任务中，牙齿体积仅用作参考，会因为被牙龈埋伏等原因导致与真实大小差异较大。
    找出图中牙龈乳头、口腔内壁、软组织等显著的非牙齿区域，以json格式返回他们的ID:{"results": [<id>, ...]}，如果没有，则返回{"results": []}。注意，不要将可能是牙齿的局部mask识别为非牙齿区域。牙龈乳头、口腔内壁等在形态上类似牙齿，而不是大片的区域，请结合图像仔细寻找。
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
                # 先检查大小
                incisor_vertices = [np.mean(vertices[faces[face_labels == i]], axis=1) for i in maybe_incisors['results']]
                incisor_sizes = [np.prod(geometry.compute_oriented_bounding_box_size(verts)) for verts in incisor_vertices]
                if np.min(incisor_sizes) / np.max(incisor_sizes) < 0.5:
                    incisor_error_flag = True
                    incisor_error_reason += '中切牙尺寸差异过大；'

                incisor_idx = [int(np.argwhere(exist_id == i).squeeze()) for i in maybe_incisors['results']]
                if len(incisor_idx) == 2:
                    if abs(incisor_idx[1] - incisor_idx[0]) != 1:
                        incisor_error_flag = True
                pleft = np.min(incisor_idx)
                pright = len(exist_id) - np.max(incisor_idx) - 1
                if abs(pright - pleft) > 1:
                    incisor_error_flag = True
                    incisor_error_reason += '中切牙不在牙弓中线位置；'
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


