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
    Project scattered points onto a plane and parameterize them by fitting a parabola.

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [N, 2], projected point set on the plane; [N] distances
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    projected_points_2d = points[:, 0:2]

    # Least squares fit for parabola
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
        # Finding slope and tangent requires solving a cubic equation, which is not computationally efficient
        # Therefore, use a numerical differentiation approach
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
    Compute the oriented bounding box size of a point set.

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [3], size in xyz directions
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