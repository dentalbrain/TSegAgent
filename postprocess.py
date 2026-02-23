import numpy as np
import trimesh

from fdi_optimize import geometry
from visualize.visualize import visualize_face_labels


def _remove_large_predictions_v1(masks: np.ndarray, boxes: np.ndarray, id_map: np.ndarray, threshold: float = 0.2):
    face_count = np.sum(id_map > 0)
    filtered_id = []
    for part in range(len(masks)):
        positive = np.sum(masks[part])
        if positive / face_count < threshold:
            filtered_id.append(part)
    if len(filtered_id) == 0:
        return boxes[0:0], masks[0:0]
    filtered_id = np.array(filtered_id)
    return boxes[filtered_id], masks[filtered_id]


def _remove_large_predictions(masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray, id_map: np.ndarray, threshold: float = 0.2):
    face_count = np.sum(id_map > 0)
    filtered_id = []
    for part in range(len(masks)):
        positive = np.sum(masks[part])
        if positive / face_count < threshold:
            filtered_id.append(part)
    if len(filtered_id) == 0:
        return boxes[0:0], masks[0:0], scores[0:0]
    filtered_id = np.array(filtered_id)
    return boxes[filtered_id], masks[filtered_id], scores[filtered_id]


def _voxelize_prediction(pred, origin, vox_size=0.5):
    """
    pred: (N,3) point cloud
    origin: (3,) voxel grid origin (must be consistent between A/B)
    vox_size: float

    return: (K,) 1D "voxel tokens" (structured dtype), each token represents a unique voxel
            Can be used directly with np.intersect1d / np.union1d (assume_unique=True is faster)
    """
    pred = np.asarray(pred, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    if pred.size == 0:
        return np.empty((0,), dtype=[("x", np.int64), ("y", np.int64), ("z", np.int64)])
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("pred must have shape (N,3)")
    if vox_size <= 0:
        raise ValueError("vox_size must be > 0")

    v = np.floor((pred - origin) / vox_size).astype(np.int64)      # (N,3)
    v = np.unique(v, axis=0)                                       # (K,3) deduplicated occupied voxels

    # Convert to structured array, one token per row
    tokens = np.ascontiguousarray(v).view(
        np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    ).reshape(-1)

    # Already unique and sorted (result of np.unique), so intersect1d can use assume_unique=True
    return tokens


def _calc_mutual_iou(pred_a, pred_b, origin=None, vox_size=0.005, origin_mode="global_min"):
    """
    pred_a/pred_b: point clouds (Na,3)/(Nb,3)

    Returns:
      ioa = |Va ∩ Vb| / |Va|   (Va = occupied voxel set of a)
      iob = |Va ∩ Vb| / |Vb|
    """
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)

    if pred_a.size == 0 or pred_b.size == 0:
        return 0.0, 0.0

    if origin is None:
        if origin_mode == "global_min":
            origin = np.minimum(np.min(pred_a, axis=0), np.min(pred_b, axis=0))
        elif origin_mode == "zero":
            origin = np.zeros(3, dtype=np.float64)
        elif origin_mode == "a_min":
            origin = np.min(pred_a, axis=0)
        elif origin_mode == "b_min":
            origin = np.min(pred_b, axis=0)
        else:
            raise ValueError(f"Unknown origin_mode: {origin_mode}")
    else:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)

    va = _voxelize_prediction(pred_a, origin, vox_size)
    vb = _voxelize_prediction(pred_b, origin, vox_size)

    if va.size == 0 or vb.size == 0:
        return 0.0, 0.0

    inter = np.intersect1d(va, vb, assume_unique=True).size
    ua = va.size
    ub = vb.size

    ioa = inter / ua if ua > 0 else 0.0
    iob = inter / ub if ub > 0 else 0.0
    return float(ioa), float(iob)

def _merge_predictions(vertices: np.ndarray, faces: np.ndarray, id_maps: np.ndarray, predicts: dict, merge_threshold: float = 0.7, face_visit_threshold: float = 0.33):
    face_labels = np.zeros(len(faces), dtype=np.int32)
    face_positives = np.zeros(len(faces), dtype=np.int32)
    face_visits = np.zeros(len(faces), dtype=np.float32)

    merged_masks = []
    merged_mask_prob_list = []

    face_centers = vertices[faces].mean(axis=1)

    for i in range(len(predicts)):  # for each predicted image view

        if i in [9, 10, 14]:
            continue
        id_map = id_maps[i] # [H, W] np.int64, 1-based face id
        boxes = predicts[i]['boxes']    # [num_parts, 4] np.float32
        masks = predicts[i]['masks']    # [num_parts, 1, H, W] np.bool
        scores = predicts[i]['scores']  # [num_parts] np.float32
        if len(boxes) == 0:
            continue
        boxes, masks, scores = _remove_large_predictions(masks, boxes, scores, id_map)

        rendered_face_ids = np.unique(id_map - 1)
        rendered_face_ids = rendered_face_ids[rendered_face_ids >= 0]
        face_visits[rendered_face_ids] += 1
        face_positives_part = np.zeros(len(faces), dtype=np.int8)

        ######### Merge masks with overlap greater than merge_threshold #########
        for mask_i in range(len(masks)):
            mask = masks[mask_i][0]
            positive_face_ids = np.unique(id_map[mask] - 1)
            positive_face_ids = positive_face_ids[positive_face_ids >= 0]

            face_positives_part[positive_face_ids] = 1

            positive_face_centers = face_centers[positive_face_ids]

            is_merged = False
            for merged_i in range(len(merged_masks)):
                ioa, iob = _calc_mutual_iou(positive_face_centers, face_centers[merged_masks[merged_i]])
                # print(f'IOU[{mask_i}, {merged_i}] ioa={ioa:.4f}, iob={iob:.4f}')
                if ioa > merge_threshold and iob > merge_threshold:
                    # If mask overlap with merged_i exceeds merge_threshold, merge into merged_mask
                    merged_masks[merged_i] = np.union1d(merged_masks[merged_i], positive_face_ids)
                    merged_mask_prob_list[merged_i].append(scores[mask_i])
                    is_merged = True
                    # print(f'Merged i={i} mask_i={mask_i} , merged_i={merged_i}')
                    break
                if i in [9, 10, 11, 12, 13, 14, 15]:
                    if ioa > merge_threshold or iob > merge_threshold:
                        # If mask overlap with merged_i exceeds merge_threshold, merge into merged_mask
                        merged_masks[merged_i] = np.union1d(merged_masks[merged_i], positive_face_ids)
                        merged_mask_prob_list[merged_i].append(scores[mask_i])
                        is_merged = True
                        # print(f'Merged i={i} mask_i={mask_i} , merged_i={merged_i}')
                        break

            if not is_merged:
                # This is a new instance mask
                merged_masks.append(positive_face_ids)
                merged_mask_prob_list.append([scores[mask_i]])
                # print(f'New instance i={i} mask_i={mask_i}, merged_i={len(merged_masks) - 1}')
        face_positives += face_positives_part

    ######### Merge probability values #########
    for merged_i in range(len(merged_masks)):
        merged_mask_prob_list[merged_i] = np.mean(merged_mask_prob_list[merged_i])  # Could try sum to see the effect
        np.savetxt(f'tmp/fmask/{merged_i}.xyz', face_centers[merged_masks[merged_i]])

    ######### Find merged_masks with containment relations, determine if they belong to the same instance based on score #########
    reduced_count = np.zeros(len(merged_masks), dtype=np.int32)
    for merged_i in range(len(merged_masks) - 1):
        for merged_j in range(merged_i + 1, len(merged_masks)):
            if len(merged_masks[merged_i]) == 0 or len(merged_masks[merged_j]) == 0:
                continue
            ioa, iob = _calc_mutual_iou(face_centers[merged_masks[merged_i]], face_centers[merged_masks[merged_j]])
            # If there is significant overlap, determine whether the instance belongs to the smaller or larger region
            if ioa > merge_threshold + 0.15:
                # merged_i is the smaller region
                small_i, large_i = merged_i, merged_j
            elif iob > merge_threshold + 0.15:
                # merged_j is the smaller region
                small_i, large_i = merged_j, merged_i
            else:
                continue
            sc_small, sc_large = merged_mask_prob_list[small_i], merged_mask_prob_list[large_i]
            if sc_small > sc_large:
                # Prefer the smaller region, subtract from the larger region
                reduced = np.setdiff1d(merged_masks[large_i], merged_masks[small_i])
                reduced_count[large_i] += (len(merged_masks[large_i]) - len(reduced))
                merged_masks[large_i] = reduced
            else:
                # Prefer the larger region, merge into the larger region and clear the smaller mask
                merged_masks[large_i] = np.union1d(merged_masks[large_i], merged_masks[small_i])
                merged_masks[small_i] = np.zeros(0)

    ######### Write back labels #########
    print(reduced_count.tolist())
    print([len(x) for x in merged_masks])
    face_id = 1
    for merged_i in range(len(merged_masks)):
        if len(merged_masks[merged_i]) > 0:
            print(reduced_count[merged_i] / (reduced_count[merged_i] + len(merged_masks[merged_i])))
            if reduced_count[merged_i] / (reduced_count[merged_i] + len(merged_masks[merged_i])) > 0.5:
                continue
            face_labels[merged_masks[merged_i]] = face_id
            face_id += 1

    face_labels[np.argwhere(face_positives / (face_visits + 1e-3) < face_visit_threshold)] = 0

    return face_labels


def _merge_predictions_v1(faces: np.ndarray, id_maps: np.ndarray, predicts: dict, merge_threshold: float = 0.4, face_visit_threshold: float = 0.33):
    face_labels = np.zeros(len(faces), dtype=np.int32)
    face_positives = np.zeros(len(faces), dtype=np.int32)
    face_visits = np.zeros(len(faces), dtype=np.float32)
    id_cnt = 1

    for i in range(len(predicts)):
        id_map = id_maps[i] # [H, W] np.int64, 1-based face id
        boxes = predicts[i]['boxes']    # [num_parts, 4] np.float32
        masks = predicts[i]['masks']    # [num_parts, 1, H, W] np.bool
        boxes, masks = _remove_large_predictions_v1(masks, boxes, id_map)

        rendered_face_ids = np.unique(id_map - 1)
        rendered_face_ids = rendered_face_ids[rendered_face_ids >= 0]
        face_visits[rendered_face_ids] += 1
        face_positives_part = np.zeros(len(faces), dtype=np.int8)

        for part in range(len(masks)):
            mask = masks[part][0]
            # positive_face_ids = np.unique(id_map[mask] - 1)
            positive_face_ids = np.unique(id_map[mask] - 1)
            positive_face_ids = positive_face_ids[positive_face_ids >= 0]

            face_positives_part[positive_face_ids] = 1

            current_masked = face_labels[positive_face_ids]
            current_masked_gt_0 = current_masked[current_masked > 0]
            current_masked_unique, current_masked_unique_cnt = np.unique(current_masked_gt_0, return_counts=True)
            if len(current_masked_unique) == 0:
                face_labels[positive_face_ids] = id_cnt
                id_cnt += 1
            else:
                max_cnt_idx = np.argmax(current_masked_unique_cnt)
                alternative_id = current_masked_unique[max_cnt_idx]
                alt_over_positive = current_masked_unique_cnt[max_cnt_idx] / len(positive_face_ids)
                if alt_over_positive > merge_threshold:
                    face_labels[positive_face_ids] = alternative_id
                else:
                    face_labels[positive_face_ids] = id_cnt
                    id_cnt += 1

        face_positives += face_positives_part

    face_labels[np.argwhere(face_positives / (face_visits + 1e-3) < face_visit_threshold)] = 0

    return face_labels


def _remove_disconnected_parts(faces: np.ndarray, face_labels: np.ndarray, vertices: np.ndarray, cc_size_threshold: int = 50, cc_ratio_threshold: float = 0.1):
    """
    Remove disconnected parts from the mesh by connected components.
    Args:
        faces: np.ndarray, [num_faces, 3]
        face_labels: np.ndarray, [num_faces]
        vertices: np.ndarray, [num_vertices, 3]
        cc_size_threshold: int, the minimum face count of the connected component
        cc_ratio_threshold: float, the minimum face count ratio of the connected component
    Returns:
        face_labels: np.ndarray, [num_faces]
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_adjacency = mesh.face_adjacency
    cleaned_labels = face_labels.copy()

    for label in np.unique(cleaned_labels):
        if label <= 0:
            continue

        label_mask = cleaned_labels == label
        label_faces = np.nonzero(label_mask)[0]
        if label_faces.size == 0:
            continue

        # Keep only adjacency edges whose two faces share the same label.
        if face_adjacency.size == 0:
            label_edges = np.empty((0, 2), dtype=np.int64)
        else:
            mask = label_mask[face_adjacency[:, 0]] & label_mask[face_adjacency[:, 1]]
            label_edges = face_adjacency[mask]

        components = trimesh.graph.connected_components(label_edges, nodes=label_faces, min_len=1)
        max_component_faces = np.max([len(component) for component in components])
        for component in components:
            if len(component) < cc_ratio_threshold * max_component_faces or len(component) < cc_size_threshold:
                cleaned_labels[component] = 0

    return cleaned_labels


def _fill_small_gaps(
    faces: np.ndarray,
    face_labels: np.ndarray,
    vertices: np.ndarray,
    gap_size_threshold: int = 30,
    majority_ratio: float = 0.5,
):
    """
    Fill small unlabeled gaps that are surrounded by labeled faces.
    Args:
        faces: np.ndarray, [num_faces, 3]
        face_labels: np.ndarray, [num_faces]
        vertices: np.ndarray, [num_vertices, 3]
        gap_size_threshold: int, maximum size of gap components to be filled
        majority_ratio: float, required ratio for the dominant neighboring label
    Returns:
        face_labels: np.ndarray, [num_faces]
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_adjacency = mesh.face_adjacency
    cleaned_labels = face_labels.copy()
    num_faces = len(faces)

    neighbors = [[] for _ in range(num_faces)]
    if face_adjacency.size > 0:
        for a, b in face_adjacency:
            neighbors[a].append(b)
            neighbors[b].append(a)

    unlabeled_mask = cleaned_labels <= 0
    unlabeled_faces = np.nonzero(unlabeled_mask)[0]
    if unlabeled_faces.size == 0:
        return cleaned_labels

    if face_adjacency.size == 0:
        unlabeled_edges = np.empty((0, 2), dtype=np.int64)
    else:
        mask = unlabeled_mask[face_adjacency[:, 0]] & unlabeled_mask[face_adjacency[:, 1]]
        unlabeled_edges = face_adjacency[mask]

    components = trimesh.graph.connected_components(unlabeled_edges, nodes=unlabeled_faces, min_len=1)
    for component in components:
        if len(component) > gap_size_threshold:
            continue

        boundary_labels = []
        for face_idx in component:
            for neighbor_idx in neighbors[face_idx]:
                neighbor_label = cleaned_labels[neighbor_idx]
                if neighbor_label > 0:
                    boundary_labels.append(neighbor_label)

        if len(boundary_labels) == 0:
            continue

        labels, counts = np.unique(boundary_labels, return_counts=True)
        dominant_idx = np.argmax(counts)
        dominant_label = labels[dominant_idx]
        if counts[dominant_idx] / np.sum(counts) >= majority_ratio:
            cleaned_labels[component] = dominant_label

    return cleaned_labels


def postprocess(predictions: dict, reorder_face: bool = True):
    id_maps = predictions["id_maps"]
    predicts = predictions["predicts"]
    vertices = predictions["vertices"]
    faces = predictions["faces"]

    face_labels = _merge_predictions_v1(faces, id_maps, predicts)
    face_labels = _remove_disconnected_parts(faces, face_labels, vertices, cc_ratio_threshold=0.1)
    face_labels = _fill_small_gaps(
        faces,
        face_labels,
        vertices,
        gap_size_threshold=max(10, int(len(faces) * 0.002)),
        majority_ratio=0.6,
    )

    if reorder_face:
        # Reorder face_labels
        _c = np.mean(vertices, 0)
        _m = np.sqrt(np.max(np.sum((vertices - _c) ** 2, -1)))
        vertices_norm = (vertices - _c) / _m

        # reorder face labels
        face_labels_uniq = np.unique(face_labels)
        face_labels_uniq = face_labels_uniq[face_labels_uniq > 0]

        centroids = []
        for tid in face_labels_uniq:
            centroids.append(np.mean(np.mean(vertices_norm[faces[face_labels == tid]], axis=0), axis=0))
        centroids = np.array(centroids)

        curve_param = geometry.parameterize_points_on_xy_plane(centroids)[0]
        curve_param = np.asarray(curve_param, dtype=float)

        order_idx = np.argsort(curve_param)
        face_labels_reorder = np.zeros_like(face_labels)
        for i, order in enumerate(order_idx):
            face_labels_reorder[face_labels == face_labels_uniq[order]] = i + 1
        face_labels = face_labels_reorder
    return vertices, faces, face_labels


def _merge_predictions_coarse(faces: np.ndarray, id_maps: np.ndarray, predicts: dict):
    face_labels = np.zeros(len(faces), dtype=np.int32)
    id_cnt = 1

    for i in range(len(predicts)):
        id_map = id_maps[i] # [H, W] np.int64, 1-based face id
        boxes = predicts[i]['boxes']    # [num_parts, 4] np.float32
        masks = predicts[i]['masks']    # [num_parts, 1, H, W] np.bool
        scores = predicts[i]['scores']  # [num_parts] np.float32
        boxes, masks, scores = _remove_large_predictions(masks, boxes, scores, id_map)

        for part in range(len(masks)):
            mask = masks[part][0]
            # positive_face_ids = np.unique(id_map[mask] - 1)
            positive_face_ids = np.unique(id_map[mask] - 1)
            positive_face_ids = positive_face_ids[positive_face_ids >= 0]

            current_masked = face_labels[positive_face_ids]
            current_masked_gt_0 = current_masked[current_masked > 0]
            current_masked_unique, current_masked_unique_cnt = np.unique(current_masked_gt_0, return_counts=True)
            zero_face_ids = positive_face_ids[current_masked == 0]

            face_labels[zero_face_ids] = id_cnt
            id_cnt += 1

            for masked in current_masked_unique:
                face_labels[current_masked[current_masked == masked]] = id_cnt
                id_cnt += 1

    return face_labels


def postprocess_coarse(predictions: dict, reorder_face: bool = True):
    id_maps = predictions["id_maps"]
    predicts = predictions["predicts"]
    vertices = predictions["vertices"]
    faces = predictions["faces"]

    face_labels = _merge_predictions_coarse(faces, id_maps, predicts)
    face_labels = _remove_disconnected_parts(faces, face_labels, vertices, cc_ratio_threshold=0.1)
    face_labels = _fill_small_gaps(
        faces,
        face_labels,
        vertices,
        gap_size_threshold=max(10, int(len(faces) * 0.002)),
        majority_ratio=0.6,
    )

    if reorder_face:
        _c = np.mean(vertices, 0)
        _m = np.sqrt(np.max(np.sum((vertices - _c) ** 2, -1)))
        vertices_norm = (vertices - _c) / _m

        # reorder face labels
        face_labels_uniq = np.unique(face_labels)
        face_labels_uniq = face_labels_uniq[face_labels_uniq > 0]

        centroids = []
        for tid in face_labels_uniq:
            centroids.append(np.mean(np.mean(vertices_norm[faces[face_labels == tid]], axis=0), axis=0))
        centroids = np.array(centroids)

        curve_param = geometry.parameterize_points_on_xy_plane(centroids)[0]
        curve_param = np.asarray(curve_param, dtype=float)

        order_idx = np.argsort(curve_param)
        face_labels_reorder = np.zeros_like(face_labels)
        for i, order in enumerate(order_idx):
            face_labels_reorder[face_labels == face_labels_uniq[order]] = i + 1
        face_labels = face_labels_reorder

    return vertices, faces, face_labels


def find_label_edge_vertices(vertices, faces, face_labels):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vedges = np.zeros(len(vertices), dtype=np.int32)

    adj = mesh.face_adjacency
    if adj is None or len(adj) == 0:
        return mesh, vedges

    f0 = adj[:, 0]
    f1 = adj[:, 1]

    diff = face_labels[f0] != face_labels[f1]
    # boundary_vertices = adj
    boundary_vertices = np.intersect1d(faces[f0[diff]], faces[f1[diff]])
    vedges[boundary_vertices] = 1

    boundary_rgb = np.array([0x33, 0x33, 0x33], dtype=np.uint8)

    vertex_colors = np.ones((len(vertices), 3), np.uint8) * 205
    vertex_colors[boundary_vertices] = boundary_rgb
    mesh.visual.vertex_colors = vertex_colors
    return mesh, vedges
