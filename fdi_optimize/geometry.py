import numpy as np
try:
    from scipy.optimize import leastsq  # type: ignore
except Exception:  # pragma: no cover
    # fallback using numpy lstsq to fit y = a x^2 + b x + c
    def leastsq(func, x0, args=()):
        x, y = args
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.vstack([x * x, x, np.ones_like(x)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coeffs, None


def parameterize_points_on_xy_plane(points, params=None):
    """
    散点投影到平面参数化，拟合抛物线参数

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [N, 2]，平面投影点集; [N] 距离
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    projected_points_2d = points[:, 0:2]

    # 最小二乘法拟合抛物线
    def func(params, x):
        a, b, c = params
        return a * x * x + b * x + c

    def error(params, x, y):
        return func(params, x) - y

    init_params = np.array([-1, 0, 0])
    if params is not None:
        a, b, c = params
    else:
        a, b, c = leastsq(error, init_params, args=(projected_points_2d[:, 0], projected_points_2d[:, 1]))[0]

    def point_on_curve(x, y, l=-1, r=1):
        # 求斜率和切线需要求解三次方程，不利于计算
        # 因此采用微分方法
        splits = 1000
        result_x = l

        min_dist = 10000
        current = l
        while current < r:
            current_y = func((a, b, c), current)
            if min_dist > ((current_y - y) ** 2 + (current - x) ** 2):
                min_dist = (current_y - y) ** 2 + (current - x) ** 2
                result_x = current
            current += (r - l) / splits

        return result_x, min_dist

    results = []
    dists = []
    for point_2d in projected_points_2d:
        point_on_curve_x, min_dist = point_on_curve(point_2d[0], point_2d[1])
        results.append(point_on_curve_x)
        dists.append(min_dist)
    return results, np.array(dists) + points[:, 2] ** 2


def compute_oriented_bounding_box_size(points):
    """
    计算点集的定向包围盒尺寸

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [3]，xyz方向尺寸
    :rtype: np.ndarray
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    pca.fit(points)
    rotated_points = pca.transform(points)

    min_xyz = np.min(rotated_points, axis=0)
    max_xyz = np.max(rotated_points, axis=0)

    size = max_xyz - min_xyz
    return size