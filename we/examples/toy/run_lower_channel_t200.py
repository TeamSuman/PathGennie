#!/usr/bin/env python3
"""Short channel-specific WE flux test for the 2D three-hole potential."""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from toy import OpenMMRunner
from wepath import WESS
from run_path1 import PathCV


LEFT_BASIN = np.array([-1.13366645, -0.03864255, 0.0])
RIGHT_BASIN = np.array([1.13366645, -0.03864255, 0.0])


def warp_criteria(positions, kwargs):
    target = kwargs.get("target", RIGHT_BASIN)
    radius = kwargs.get("radius", 0.5)
    return np.linalg.norm(np.asarray(positions).reshape(-1, 3)[0] - target) < radius


def project(positions, kwargs):
    path_cv = kwargs["path_cv"]
    coords = np.asarray(positions).reshape(1, 3)
    s, z = path_cv.compute(coords)
    return np.array([s, z])


def lower_channel_initial_guess():
    """Initial string from the left basin to the right basin through y < 0."""
    return np.array(
        [
            LEFT_BASIN,
            [-0.85, -0.30, 0.00],
            [-0.55, -0.58, 0.00],
            [-0.20, -0.75, 0.00],
            [0.20, -0.75, 0.00],
            [0.55, -0.58, 0.00],
            [0.85, -0.30, 0.00],
            RIGHT_BASIN,
        ]
    )


def upper_channel_initial_guess():
    """Initial string from the left basin to the right basin through y > 0."""
    return np.array(
        [
            LEFT_BASIN,
            [-0.96, 0.53, 0.00],
            [-0.71, 0.98, 0.00],
            [-0.51, 1.26, 0.00],
            [0.00, 1.54, 0.00],
            [0.51, 1.26, 0.00],
            [0.71, 0.98, 0.00],
            [0.96, 0.53, 0.00],
            RIGHT_BASIN,
        ]
    )


def three_hole_gradient_xy(points):
    """Analytical gradient of the 2D three-hole potential."""
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]

    exp_x = np.exp(-(x * x))
    exp_y = np.exp(-(y * y))
    exp_y_upper = np.exp(-((y - 5.0 / 3.0) ** 2))
    exp_y_mid = np.exp(-((y - 1.0 / 3.0) ** 2))
    exp_x_right = np.exp(-((x - 1.0) ** 2))
    exp_x_left = np.exp(-((x + 1.0) ** 2))

    grad_x = (
        6.0 * x * exp_x * (exp_y_upper - exp_y_mid)
        + 10.0 * exp_y * ((x - 1.0) * exp_x_right + (x + 1.0) * exp_x_left)
    )
    grad_y = (
        6.0
        * exp_x
        * ((y - 5.0 / 3.0) * exp_y_upper - (y - 1.0 / 3.0) * exp_y_mid)
        + 10.0 * y * exp_y * (exp_x_right + exp_x_left)
    )
    return np.column_stack([grad_x, grad_y])


def resample_equal_arclength(points):
    """Redistribute string images uniformly along arc length."""
    points = np.asarray(points, dtype=float)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total == 0.0:
        return points.copy()

    targets = np.linspace(0.0, total, len(points))
    resampled = np.empty_like(points)
    for dim in range(points.shape[1]):
        resampled[:, dim] = np.interp(targets, cumulative, points[:, dim])
    return resampled


def zts_mfep_points(
    channel,
    n_images=21,
    n_steps=2000,
    step_size=0.01,
    tolerance=1.0e-8,
):
    """
    Compute an MFEP-like string with the zero-temperature string method.

    The string evolves by the perpendicular component of -grad(U), then is
    reparameterized to equal arc length after each update. Endpoints are fixed.
    """
    if channel == "lower":
        initial = lower_channel_initial_guess()
    elif channel == "upper":
        initial = upper_channel_initial_guess()
    else:
        raise ValueError("channel must be 'lower' or 'upper'")

    alphas = np.linspace(0.0, 1.0, n_images)
    string = np.array([interpolate_path(initial, alpha) for alpha in alphas])
    xy = string[:, :2]
    endpoints = xy[[0, -1]].copy()

    for _ in range(n_steps):
        old_xy = xy.copy()
        grad = three_hole_gradient_xy(xy)

        tangent = np.empty_like(xy)
        tangent[1:-1] = xy[2:] - xy[:-2]
        tangent[0] = xy[1] - xy[0]
        tangent[-1] = xy[-1] - xy[-2]
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(tangent_norm, 1.0e-12)
        grad_perp = grad - np.sum(grad * tangent, axis=1, keepdims=True) * tangent

        xy[1:-1] -= step_size * grad_perp[1:-1]
        if channel == "lower":
            xy[1:-1, 1] = np.minimum(xy[1:-1, 1], -1.0e-6)
        else:
            xy[1:-1, 1] = np.maximum(xy[1:-1, 1], 1.0e-6)

        xy[0] = endpoints[0]
        xy[-1] = endpoints[1]
        xy = resample_equal_arclength(xy)
        xy[0] = endpoints[0]
        xy[-1] = endpoints[1]

        if np.max(np.linalg.norm(xy - old_xy, axis=1)) < tolerance:
            break

    z = np.zeros((len(xy), 1))
    return np.column_stack([xy, z])


def interpolate_path(points, alpha):
    """Return a point at normalized arc-length position alpha along path nodes."""
    if alpha <= 0.0:
        return points[0].copy()
    if alpha >= 1.0:
        return points[-1].copy()

    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    target = alpha * total
    segment = np.searchsorted(cumulative, target, side="right") - 1
    segment = min(segment, len(segment_lengths) - 1)
    local = (target - cumulative[segment]) / segment_lengths[segment]
    return points[segment] + local * deltas[segment]


def preseed_positions_for_bins(path_cv, points, s_edges, samples_per_bin=200):
    """
    Build one starting state per s-bin by selecting path points whose computed
    PathCV coordinate lands closest to each bin center.
    """
    s_edges = np.asarray(s_edges)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    alphas = np.linspace(0.0, 1.0, samples_per_bin * len(s_centers))
    candidates = np.array([interpolate_path(points, alpha) for alpha in alphas])
    candidate_s = np.array(
        [path_cv.compute(candidate.reshape(1, 3))[0] for candidate in candidates]
    )
    return [
        candidates[np.argmin(np.abs(candidate_s - center))].copy()
        for center in s_centers
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a short WE flux test for one toy-potential channel."
    )
    parser.add_argument(
        "--channel",
        choices=("lower", "upper"),
        default=os.environ.get("WEPATH_CHANNEL", "lower"),
        help="Path channel to test.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("WEPATH_TEMPERATURE", "200")),
        help="Simulation temperature in kelvin.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("WEPATH_TEST_ITERATIONS", "25")),
        help="Number of WE iterations.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("WEPATH_TEST_STEPS", "50")),
        help="OpenMM integration steps per WE iteration.",
    )
    parser.add_argument(
        "--single-source",
        action="store_true",
        help="Start only from the left basin instead of pre-seeding every s-bin.",
    )
    parser.add_argument(
        "--zts-images",
        type=int,
        default=int(os.environ.get("WEPATH_ZTS_IMAGES", "21")),
        help="Number of images in the ZTS MFEP string.",
    )
    parser.add_argument(
        "--zts-steps",
        type=int,
        default=int(os.environ.get("WEPATH_ZTS_STEPS", "2000")),
        help="Maximum ZTS relaxation steps.",
    )
    parser.add_argument(
        "--zts-step-size",
        type=float,
        default=float(os.environ.get("WEPATH_ZTS_STEP_SIZE", "0.01")),
        help="ZTS gradient descent step size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    temp_label = f"t{int(args.temperature)}"
    output_dir = Path(f"outputs/toy_{args.channel}_{temp_label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = zts_mfep_points(
        args.channel,
        n_images=args.zts_images,
        n_steps=args.zts_steps,
        step_size=args.zts_step_size,
    )
    np.savetxt(output_dir / f"zts_mfep_{args.channel}_{temp_label}.txt", points)
    path_cv = PathCV(
        points.reshape(-1, 1, 3),
        equidistance_tol=0.5,
        normalize_output=True,
    )

    s_edges = np.linspace(0.0, 1.0, 45)
    initial_positions = (
        [LEFT_BASIN.copy()]
        if args.single_source
        else preseed_positions_for_bins(path_cv, points, s_edges)
    )

    config = {
        "platform": os.environ.get("WEPATH_PLATFORM", "cpu"),
        "n_workers": int(os.environ.get("WEPATH_TEST_WORKERS", "1")),
        "threads_per_worker": int(os.environ.get("WEPATH_TEST_THREADS", "1")),
        "runner_class": OpenMMRunner,
        "runner_kwargs": {"temperature": args.temperature, "platform": "CPU"},
        "temperature": args.temperature,
        "skip_initial_warping": not args.single_source,
        "bin_edges": [s_edges, [0.0, 1.0]],
        "source_bin_indices": np.array([0, 1, 2, 3]),
        "n_walkers_per_bin": 4,
        "dt": 0.002,
        "n_steps_per_tau": args.steps,
        "n_iterations": args.iterations,
        "enable_cleaning": True,
        "clean_threshold": 0.25,
        "survive_empty": False,
        "warp_function": warp_criteria,
        "warp_kwargs": {"target": RIGHT_BASIN.copy(), "radius": 0.5},
        "flux_file": str(output_dir / f"flux_{args.channel}_{temp_label}.txt"),
        "bin_file": str(output_dir / f"bin_{args.channel}_{temp_label}.txt"),
        "walkers_file": str(output_dir / f"walkers_{args.channel}_{temp_label}.txt"),
    }

    simulation = WESS(
        config=config,
        initial_positions=initial_positions,
        projection_fn=project,
        kwargs={"path_cv": path_cv},
    )
    simulation.run()
