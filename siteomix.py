import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.optimize import differential_evolution
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull


def rigid_transform_3D(A, B):
    """Вычисляет оптимальную жесткую трансформацию между двумя наборами точек."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot((A - centroid_A).T, (B - centroid_B))
    U, S, Vt = np.linalg.svd(H)

    if np.linalg.det(Vt.T) * np.linalg.det(U.T) < 0:
        Vt[2, :] *= -1

    R = Vt.T @ U.T
    t = centroid_B - R.dot(centroid_A)
    return R, t


def icp_align(source_pts, target_pts, max_iter=100, tol=1e-6):
    """
    Выравнивает source_pts к target_pts с помощью алгоритма ICP.
    Возвращает выровненные точки и среднюю ошибку (mean_error).
    """
    source_pts = source_pts.copy()
    prev_error = 0

    for i in range(max_iter):
        kdtree = cKDTree(target_pts)
        distances, indices = kdtree.query(source_pts)
        correspondences = target_pts[indices]

        R, t = rigid_transform_3D(source_pts, correspondences)
        source_pts = (R @ source_pts.T).T + t

        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tol:
            break
        prev_error = mean_error

    return source_pts, mean_error


def create_surface_points(points: np.ndarray, density_factor: float = 2.0) -> np.ndarray:
    """Извлекает точки, описывающие форму поверхности облака."""
    if len(points) < 10:
        return points

    eps = np.median(NearestNeighbors(n_neighbors=5).fit(points).kneighbors()[0][:, 4])
    dbscan = DBSCAN(eps=eps, min_samples=3).fit(points)

    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    surface_points = points[core_samples_mask]

    if len(surface_points) < 10:
        try:
            hull = ConvexHull(points)
            surface_points = points[hull.vertices]
        except:
            surface_points = points

    return surface_points


def calculate_volume_overlap(points1, points2, resolution=1.0):
    """Оценка перекрытия объёмов двух облаков точек."""
    if len(points1) == 0 or len(points2) == 0:
        return 0.0

    min_coords = np.minimum(points1.min(axis=0), points2.min(axis=0))
    max_coords = np.maximum(points1.max(axis=0), points2.max(axis=0))

    ranges = [np.arange(min_coords[i], max_coords[i] + resolution, resolution)
              for i in range(3)]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1)
    grid_points = grid.reshape(-1, 3)

    radius = 3.0 * resolution
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    occupied1 = np.array([len(idx) > 0 for idx in tree1.query_ball_point(grid_points, r=radius)])
    occupied2 = np.array([len(idx) > 0 for idx in tree2.query_ball_point(grid_points, r=radius)])

    overlap_voxels = np.sum(occupied1 & occupied2)
    union_voxels = np.sum(occupied1 | occupied2)

    if union_voxels == 0:
        return 0.0
    return overlap_voxels / union_voxels


def apply_transform_3d(points, angles, translation):
    """Применяет поворот (углы Эйлера в градусах) и сдвиг к точкам."""
    rx, ry, rz = np.radians(angles)
    dx, dy, dz = translation

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rotated = np.dot(centered, R.T)
    return rotated + centroid + np.array([dx, dy, dz])


def optimize_alignment_3d(target_points, source_points):
    """
    Тонкая оптимизация позы через дифференциальную эволюцию.
    Работает с полными облаками точек.
    """
    def loss(params):
        rx, ry, rz, dx, dy, dz = params
        transformed = apply_transform_3d(source_points, [rx, ry, rz], [dx, dy, dz])
        hausdorff_dist = directed_hausdorff(target_points, transformed)[0]
        overlap = calculate_volume_overlap(target_points, transformed, resolution=2.0)
        return hausdorff_dist - 0.1 * overlap

    size_estimate = np.ptp(target_points, axis=0).max()
    bounds = [
        (-180, 180), (-180, 180), (-180, 180),
        (-size_estimate/2, size_estimate/2),
        (-size_estimate/2, size_estimate/2),
        (-size_estimate/2, size_estimate/2)
    ]

    result = differential_evolution(
        loss, bounds, strategy='best1bin', popsize=15,
        maxiter=100, tol=0.01, seed=42
    )
    return apply_transform_3d(source_points, result.x[:3], result.x[3:])


def full_alignment(ref_cloud: np.ndarray, target_cloud: np.ndarray) -> dict:

    # 1. Грубое выравнивание ICP (с mean_error)
    aligned_icp, icp_error = icp_align(target_cloud, ref_cloud)

    # 2. Тонкая DE-оптимизация на полных облаках
    aligned_fine = optimize_alignment_3d(ref_cloud, aligned_icp)

    # 3. Объёмное перекрытие по полным облакам
    overlap = calculate_volume_overlap(ref_cloud, aligned_fine)

    # 4. Финальный RMSD
    kdtree_ref = cKDTree(ref_cloud)
    distances, _ = kdtree_ref.query(aligned_fine)
    fine_rmsd = np.sqrt(np.mean(np.square(distances)))

    return {
        "aligned_cloud": aligned_fine,
        "overlap": overlap,
        "icp_error": icp_error,      
        "fine_rmsd": fine_rmsd
    }