"""
Maximum-likelihood estimation of the foot-ground contact point.

Standard multiview triangulation (DLT/SVD) minimizes an algebraic residual and
ignores per-view keypoint confidence. This module instead:

  1. Models each camera's 2D keypoint as the true reprojection plus Gaussian
     pixel noise with variance inversely proportional to detector confidence.
     Under that model, the MLE of the 3D point is the confidence-weighted
     reprojection-error minimizer (weighted nonlinear least squares) —
     `triangulate_mle`.

  2. During stance, the foot is known to lie on the (pre-calibrated) floor
     plane. Constraining the estimate to the plane removes one degree of
     freedom (3D point -> 2D point-on-plane), which is where most
     triangulation noise concentrates in short-baseline multicamera rigs.
     Every 2D observation from every camera AND every frame in the stance
     window can be pooled as independent evidence for the same static
     ground point — `estimate_stance_point_mle`.
"""

import numpy as np
from scipy.optimize import least_squares


def ground_plane_from_alignment(R_fix, floor_offset):
    """
    Recover the floor plane equation n . X = d in the *raw* (pre-alignment)
    triangulation frame, from the rotation/offset produced by
    utils_floor_align.calibrate_floor_pca or pose_estimation.compute_floor_rotation.

    Both compute aligned = raw @ R_fix.T; aligned[:, 1] -= floor_offset, so
    the floor (aligned Y = 0) satisfies R_fix[1, :] . raw = floor_offset.
    """
    n = np.asarray(R_fix, dtype=float)[1, :]
    n = n / np.linalg.norm(n)
    return n, float(floor_offset)


def plane_basis(n, d):
    """Orthonormal in-plane basis (x0, e1, e2) for the plane n . X = d."""
    n = n / np.linalg.norm(n)
    x0 = d * n
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return x0, e1, e2


def project(P, X):
    """Pinhole projection of 3D point X through camera matrix P (3x4)."""
    Xh = np.append(X, 1.0)
    xh = P @ Xh
    return xh[:2] / xh[2]


def _dlt_triangulate(observations):
    """Algebraic DLT initializer: observations = [(P, (u, v)), ...]."""
    A = []
    for P, (u, v) in observations:
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vh = np.linalg.svd(np.array(A))
    X = vh[-1]
    return (X / X[3])[:3]


def triangulate_mle(observations, conf_to_weight=lambda s: s):
    """
    Unconstrained weighted-reprojection MLE triangulation of one 3D point.

    observations : list of (P (3x4), (u, v), confidence)
    conf_to_weight: maps confidence -> residual weight (default: identity,
        i.e. assumed pixel variance sigma^2 = sigma0^2 / confidence).
    """
    if len(observations) < 2:
        return np.array([np.nan, np.nan, np.nan])

    weighted = [(P, uv, conf_to_weight(s)) for P, uv, s in observations]
    X_init = _dlt_triangulate([(P, uv) for P, uv, _ in weighted])

    def residuals(X):
        res = []
        for P, (u, v), w in weighted:
            pu, pv = project(P, X)
            sw = np.sqrt(w)
            res.append(sw * (pu - u))
            res.append(sw * (pv - v))
        return np.array(res)

    result = least_squares(residuals, x0=X_init, method="lm")
    return result.x


def estimate_ground_point_mle(observations, n, d, conf_to_weight=lambda s: s):
    """
    MLE of a point constrained to the floor plane n . X = d.

    observations : list of (P (3x4), (u, v), confidence); may span multiple
        cameras and multiple frames if the point is assumed static over that
        span (e.g. a heel during a single stance interval).

    Returns dict: X (3,), a, b (in-plane coords), cov_3d (3x3, rank <= 2),
        sigma_pos (1-sigma positional uncertainty, meters), n_obs,
        residual_rms (pixels).
    """
    if len(observations) < 2:
        raise ValueError("need at least 2 weighted observations to solve for 2 DOF")

    x0, e1, e2 = plane_basis(n, d)
    weighted = [(P, uv, conf_to_weight(s)) for P, uv, s in observations]

    X_init = _dlt_triangulate([(P, uv) for P, uv, _ in weighted])
    a0 = np.dot(X_init - x0, e1)
    b0 = np.dot(X_init - x0, e2)

    def residuals(params):
        a, b = params
        X = x0 + a * e1 + b * e2
        res = []
        for P, (u, v), w in weighted:
            pu, pv = project(P, X)
            sw = np.sqrt(w)
            res.append(sw * (pu - u))
            res.append(sw * (pv - v))
        return np.array(res)

    result = least_squares(residuals, x0=[a0, b0], method="lm")
    a, b = result.x
    X = x0 + a * e1 + b * e2

    # Laplace approximation: Cov(params) ~= (J^T J)^-1 * residual_variance
    J = result.jac
    dof = max(1, len(result.fun) - 2)
    resid_var = float(np.sum(result.fun ** 2) / dof)
    try:
        cov_ab = resid_var * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov_ab = np.full((2, 2), np.nan)

    basis = np.column_stack([e1, e2])  # 3x2
    cov_3d = basis @ cov_ab @ basis.T
    sigma_pos = float(np.sqrt(np.trace(cov_ab))) if np.all(np.isfinite(cov_ab)) else np.nan

    return {
        "X": X,
        "a": float(a),
        "b": float(b),
        "cov_ab": cov_ab,
        "cov_3d": cov_3d,
        "sigma_pos": sigma_pos,
        "n_obs": len(observations),
        "residual_rms": float(np.sqrt(resid_var)),
    }


def estimate_stance_point_mle(frame_observations, n, d, conf_to_weight=lambda s: s):
    """
    Pool every (camera, frame) 2D observation across a stance window into one
    ground-plane MLE solve, assuming the foot doesn't slide while planted.

    frame_observations : list over frames; each element is a list of
        (P, (u, v), confidence) for the cameras that saw the foot in that frame.
    """
    pooled = [obs for frame_obs in frame_observations for obs in frame_obs]
    return estimate_ground_point_mle(pooled, n, d, conf_to_weight=conf_to_weight)


if __name__ == "__main__":
    # Synthetic sanity check: 3 cameras looking at a room, a static ground
    # point observed over 5 "stance" frames with pixel noise, confirms the
    # plane-constrained MLE recovers the true point more accurately than
    # single-frame unconstrained DLT.
    rng = np.random.default_rng(0)

    def make_camera(eye, target, f=800, cx=320, cy=240):
        eye, target = np.array(eye, dtype=float), np.array(target, dtype=float)
        z = target - eye
        z /= np.linalg.norm(z)
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        T = (-R @ eye).reshape(3, 1)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
        return K @ np.hstack([R, T])

    cams = [
        make_camera([3, 2, -3], [0, 0, 0]),
        make_camera([-3, 2, -3], [0, 0, 0]),
        make_camera([0, 2.5, 4], [0, 0, 0]),
    ]

    n_true, d_true = np.array([0.0, 1.0, 0.0]), 0.0  # floor y=0
    X_true = np.array([0.05, 0.0, -0.10])

    frame_obs = []
    for _ in range(5):
        obs = []
        for P in cams:
            u, v = project(P, X_true)
            noise = rng.normal(0, 0.8, size=2)  # ~0.8 px noise
            score = rng.uniform(0.6, 0.95)
            obs.append((P, (u + noise[0], v + noise[1]), score))
        frame_obs.append(obs)

    single_frame_dlt = _dlt_triangulate([(P, uv) for P, uv, _ in frame_obs[0]])
    pooled_result = estimate_stance_point_mle(frame_obs, n_true, d_true)

    print("true point           :", X_true)
    print("single-frame DLT     :", single_frame_dlt,
          " err =", np.linalg.norm(single_frame_dlt - X_true))
    print("pooled plane MLE     :", pooled_result["X"],
          " err =", np.linalg.norm(pooled_result["X"] - X_true))
    print("sigma_pos (1-sigma)  :", pooled_result["sigma_pos"])
    print("residual_rms (px)    :", pooled_result["residual_rms"])
