import numpy as np
import torch


def miou_dice(pred, target, n_classes=1):
    miou, dice = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred = torch.from_numpy(pred).int().to(device)
    target = torch.from_numpy(target).int().to(device)
    for cls in range(1, n_classes + 1):
        intersections = torch.sum((pred == cls) * (target == cls))
        unions = torch.sum(pred == cls) + torch.sum(target == cls)
        if unions == 0 or intersections == 0:
            continue
        miou.append((intersections / (unions - intersections)).data.cpu().numpy())
        dice.append((2 * intersections / unions).data.cpu().numpy())
    if len(miou) == 0:
        miou = [0.0]
    if len(dice) == 0:
        dice = [0.0]
    return np.mean(miou), np.mean(dice)


def mean_intersection_over_union_np(pred, target, n_classes=1):
    """
    mIoU evaluation metric

    :param n_classes: Number of classes
    :type n_classes: int
    :param pred: [B, N]
    :type pred: numpy.ndarray
    :param target: [B, N]
    :type target: numpy.ndarray
    :return: mIoU value
    :rtype: float
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    values = []
    for cls in range(1, n_classes + 1):
        intersections = np.sum((pred == cls) * (target == cls))
        unions = np.sum(pred == cls) + np.sum(target == cls) - intersections
        if unions == 0:
            continue
        values.append(intersections / unions)
    if len(values) == 0:
        return 1.0
    return np.mean(values)


def mean_intersection_over_union(pred, target, n_classes=1):
    """
    mIoU evaluation metric

    :param n_classes: Number of classes
    :type n_classes: int
    :param pred: [B, N]
    :type pred: torch.Tensor
    :param target: [B, N]
    :type target: torch.Tensor
    :return: mIoU value
    :rtype: float
    """
    pred = pred.view(-1)
    target = target.view(-1)
    values = []
    for cls in range(1, n_classes + 1):
        intersections = torch.sum((pred == cls) * (target == cls))
        unions = torch.sum(pred == cls) + torch.sum(target == cls) - intersections
        if unions.item() == 0:
            continue
        values.append(intersections.item() / unions.item())
    if len(values) == 0:
        return 1.0
    return np.mean(values)


def dice_similarity_np(pred, target, n_classes=1):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    values = []
    for cls in range(1, n_classes + 1):
        target_sum = np.sum(target == cls)
        if target_sum < 0:
            continue
        intersections = np.sum((pred == cls) * (target == cls))
        unions = np.sum(pred == cls) + target_sum
        if unions == 0:
            continue
        values.append(2 * intersections / unions)
    if len(values) == 0:
        return 1.0
    return np.mean(values)


def dice_similarity(pred, target, n_classes=1):
    pred = pred.view(-1)
    target = target.view(-1)
    values = []
    for cls in range(1, n_classes + 1):
        target_sum = (target == cls).sum()
        if target_sum < 0:
            continue
        intersections = torch.sum((pred == cls) * (target == cls))
        unions = torch.sum(pred == cls) + torch.sum(target == cls)
        if unions.item() == 0:
            continue
        values.append(2 * intersections.item() / unions.item())
    if len(values) == 0:
        return 1.0
    return torch.mean(torch.stack(values))


if __name__ == '__main__':
    a = np.random.randint(0, 2, (40, 40, 40))
    b = np.random.randint(0, 2, (40, 40, 40))
    b[..., :3] = 0
    s = np.array([0.3, 0.3, 0.3])
    hausdorff_distance(a, b, s)
