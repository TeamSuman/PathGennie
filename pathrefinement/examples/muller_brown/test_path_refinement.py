"""
test_path_refinement.py
─────────────────────────
Comprehensive test of the Muller-Brown path refinement methodology.

Tests:
1. Check existing results exist and are loadable
2. Verify refinement convergence (decreasing MSD between iterations)
3. Test PathCV (s, z) computation is correct
4. Test PrincipalCurve smoothing
5. Test that iterative refinement approaches the physically correct path
6. Verify the path stays within the physically valid region

Run: conda run -n pathgennie python3 test_path_refinement.py
"""

import os
import sys
import glob
import numpy as np
import json

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from pathrefinement.pathcv import PathCV
from pathrefinement.principal_curve import PrincipalCurve

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
INIT_DIR = os.path.join(BASE_DIR, "results", "initial_path")
REFINE_DIR = os.path.join(BASE_DIR, "results", "refinement")
INIT_PATH_FILE = os.path.join(INIT_DIR, "initial_path.npy")

# Known Muller-Brown minima from paper
MINIMA = {
    "A": np.array([-0.558224, 1.441726]),   # deeper minimum (source)
    "B": np.array([0.623499, 0.028038]),     # shallow minimum (target)
}


def load_all_paths():
    """Load initial path + all refinement iterations."""
    paths = [np.load(INIT_PATH_FILE)]
    labels = ["Initial"]

    iter_files = sorted(
        glob.glob(os.path.join(REFINE_DIR, "path_iter_*.npy")),
        key=lambda f: int(os.path.basename(f).replace("path_iter_", "").replace(".npy", "")),
    )
    for f in iter_files:
        it = int(os.path.basename(f).replace("path_iter_", "").replace(".npy", ""))
        paths.append(np.load(f))
        labels.append(f"Iter {it}")

    # Also load final path if exists
    final_file = os.path.join(REFINE_DIR, "final_path.npy")
    if os.path.exists(final_file):
        final_path = np.load(final_file)
        # Check if final_path duplicates the last iteration
        if not np.allclose(final_path, paths[-1]):
            paths.append(final_path)
            labels.append("Final")

    return paths, labels


def test_existing_results():
    """Test 1: Verify all expected result files exist."""
    print("=" * 60)
    print("TEST 1: Verify existing result files")
    print("=" * 60)

    checks = [
        ("Initial path", os.path.exists(INIT_PATH_FILE)),
        ("Refinement directory", os.path.isdir(REFINE_DIR)),
        ("Final path", os.path.exists(os.path.join(REFINE_DIR, "final_path.npy"))),
        ("Path history", os.path.exists(os.path.join(REFINE_DIR, "path_history.npz"))),
    ]

    all_ok = True
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        all_ok = all_ok and ok

    # Check for iteration files
    iter_files = sorted(
        glob.glob(os.path.join(REFINE_DIR, "path_iter_*.npy")),
        key=lambda f: int(os.path.basename(f).replace("path_iter_", "").replace(".npy", "")),
    )
    print(f"  {'✓' if len(iter_files) > 0 else '✗'} Iteration files: {len(iter_files)} found")
    for f in iter_files:
        it = int(os.path.basename(f).replace("path_iter_", "").replace(".npy", ""))
        data = np.load(f)
        print(f"       Iter {it}: shape {data.shape}")
    all_ok = all_ok and len(iter_files) > 0

    print()
    return all_ok


def test_pathcv_construction():
    """Test 2: Verify PathCV can be constructed from paths and computes (s, z) correctly."""
    print("=" * 60)
    print("TEST 2: PathCV construction and (s, z) computation")
    print("=" * 60)

    paths, labels = load_all_paths()

    for label, path in zip(labels, paths):
        print(f"\n  Testing PathCV for {label}...")
        # PathCV expects (N_nodes, N_atoms, D) shape
        path_3d = path[:, np.newaxis, :]

        pathcv = PathCV(
            list(path_3d),
            enforce_equidistance=False,
            normalize_output=True,
        )
        print(f"    PathCV: {pathcv}")

        # Test computation on path nodes themselves
        s_vals = []
        z_vals = []
        for pt in path_3d:
            s, z = pathcv.compute(pt)
            s_vals.append(s)
            z_vals.append(z)

        s_vals = np.array(s_vals)
        z_vals = np.array(z_vals)

        # s should monotonically increase from 0 to 1
        s_monotonic = np.all(np.diff(s_vals) >= -0.05)  # Allow small numerical noise
        s_range = (0.0 <= s_vals.min()) and (s_vals.max() <= 1.0)
        z_at_nodes = np.all(z_vals < 0.5)  # z should be small at path nodes

        print(f"    s monotonic: {s_monotonic}  (min={s_vals.min():.3f}, max={s_vals.max():.3f})")
        print(f"    s range [0,1]: {s_range}")
        print(f"    z small at nodes: {z_at_nodes}  (mean={np.mean(z_vals):.4f})")

        if not (s_monotonic and s_range and z_at_nodes):
            print(f"    ⚠ WARNING: PathCV behavior unexpected for {label}")
            return False

    print("\n  All PathCV tests passed.")
    print()
    return True


def test_refinement_convergence():
    """Test 3: Verify the refinement converged (decreasing displacement)."""
    print("=" * 60)
    print("TEST 3: Refinement convergence analysis")
    print("=" * 60)

    paths, labels = load_all_paths()

    if len(paths) < 2:
        print("  Need at least 2 paths for convergence test. Found 1.")
        return False

    print(f"\n  Loaded {len(paths)} paths from '{labels[0]}' to '{labels[-1]}'")
    print(f"  Path shape: {paths[0].shape}")

    # Compute pairwise differences
    msd_vals = []
    max_d_vals = []
    frechet_vals = []

    for i in range(1, len(paths)):
        diff = paths[i] - paths[i - 1]
        msd = np.mean(np.sum(diff ** 2, axis=1))
        max_d = np.max(np.linalg.norm(diff, axis=1))

        # Quick Frechet distance (discrete)
        # For two curves p and q of same length, the discrete Frechet distance
        d1 = np.max(np.min(np.linalg.norm(paths[i][:, None, :] - paths[i-1][None, :, :], axis=2), axis=0))
        d2 = np.max(np.min(np.linalg.norm(paths[i][:, None, :] - paths[i-1][None, :, :], axis=2), axis=1))
        frechet = max(d1, d2)

        msd_vals.append(msd)
        max_d_vals.append(max_d)
        frechet_vals.append(frechet)

        print(f"\n  Iter {labels[i]} vs {labels[i-1]}:")
        print(f"    MSD = {msd:.6f}")
        print(f"    Max displacement = {max_d:.6f}")
        print(f"    Discrete Frechet ≈ {frechet:.6f}")

    # Check convergence trend: last MSD should be smaller than first MSD
    convergence_achieved = msd_vals[-1] < msd_vals[0]
    print(f"\n  Convergence trend: {'✓' if convergence_achieved else '✗'}")
    print(f"    MSD: {msd_vals[0]:.6f} → {msd_vals[-1]:.6f}  (ratio = {msd_vals[-1]/msd_vals[0]:.3f})")
    print(f"    Frechet: {frechet_vals[0]:.6f} → {frechet_vals[-1]:.6f}")

    if convergence_achieved:
        print("\n  ✓ Refinement converged successfully!")
    else:
        print("\n  ⚠ Refinement not strictly convergent but may still be acceptable.")
        # Check if the trend is generally decreasing (not necessarily monotonic)
        print(f"     MSD values: {[f'{v:.6f}' for v in msd_vals]}")

    print()
    return convergence_achieved


def test_endpoint_pinning():
    """Test 4: Verify path endpoints remain pinned to the known minima."""
    print("=" * 60)
    print("TEST 4: Endpoint pinning verification")
    print("=" * 60)

    paths, labels = load_all_paths()

    all_ok = True
    for label, path in zip(labels, paths):
        start_ok = np.allclose(path[0], MINIMA["A"], atol=0.02)
        end_ok = np.allclose(path[-1], MINIMA["B"], atol=0.02)

        if not (start_ok and end_ok):
            print(f"  ✗ {label}: endpoints mismatch")
            print(f"    Start: {path[0]} vs {MINIMA['A']}")
            print(f"    End:   {path[-1]} vs {MINIMA['B']}")
            all_ok = False

    if all_ok:
        print("  ✓ All path endpoints correctly pinned to global minima A and B")
    print()
    return all_ok


def test_physically_valid_path():
    """Test 5: Verify paths remain in physically valid region (no excessive excursions)."""
    print("=" * 60)
    print("TEST 5: Physical validity check")
    print("=" * 60)

    paths, labels = load_all_paths()

    # Known MB potential landscape bounds
    X_RANGE = (-1.8, 1.2)
    Y_RANGE = (-0.5, 2.2)

    all_ok = True
    for label, path in zip(labels, paths):
        x_ok = (path[:, 0] >= X_RANGE[0]).all() and (path[:, 0] <= X_RANGE[1]).all()
        y_ok = (path[:, 1] >= Y_RANGE[0]).all() and (path[:, 1] <= Y_RANGE[1]).all()

        # Path should not be too long (valid path between minima)
        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_len = np.sum(seg_lens)
        path_len_ok = total_len < 8.0  # Reasonable upper bound

        if not (x_ok and y_ok and path_len_ok):
            print(f"  ✗ {label}: out of bounds")
            print(f"    x range: [{path[:, 0].min():.3f}, {path[:, 0].max():.3f}] (allowed: {X_RANGE})")
            print(f"    y range: [{path[:, 1].min():.3f}, {path[:, 1].max():.3f}] (allowed: {Y_RANGE})")
            print(f"    Total length: {total_len:.3f}")
            all_ok = False
        else:
            print(f"  ✓ {label}: in bounds, length = {total_len:.3f}")

    print()
    return all_ok


def test_principal_curve_standalone():
    """Test 6: Test PrincipalCurve on synthetic noisy path."""
    print("=" * 60)
    print("TEST 6: PrincipalCurve standalone test")
    print("=" * 60)

    # Generate a simple arc trajectory from A to B with noise
    n_points = 500
    t = np.linspace(0, 1, n_points)
    # Generate noisy path starting at A, ending at B with sinusoidal wiggle + noise
    x = MINIMA["A"][0] + (MINIMA["B"][0] - MINIMA["A"][0]) * t
    y = MINIMA["A"][1] + (MINIMA["B"][1] - MINIMA["A"][1]) * t
    # Add sinusoidal wiggles
    x += 0.1 * np.sin(2 * np.pi * t)
    y += 0.15 * np.cos(3 * np.pi * t)
    # Add noise (ensure endpoints stay at minima by scaling noise to zero at ends)
    noise_scale = np.sin(np.pi * t)  # zero at ends, max in middle
    x += np.random.RandomState(42).normal(0, 0.02, n_points) * noise_scale
    y += np.random.RandomState(42).normal(0, 0.02, n_points) * noise_scale
    # Pin endpoints explicitly to the known minima
    x[0], y[0] = MINIMA["A"]
    x[-1], y[-1] = MINIMA["B"]

    noisy_path = np.column_stack([x, y])

    # Fit principal curve
    pc = PrincipalCurve(n_images=100, lam=0.1, n_iter=100, tol=1e-5, verbose=False)
    smooth_path = pc.fit(noisy_path)

    # Check shape
    assert smooth_path.shape == (100, 2), f"Expected (100, 2), got {smooth_path.shape}"

    # Check endpoints pinned — PrincipalCurve pins to first/last points of input,
    # which we set to MINIMA values
    start_ok = np.allclose(smooth_path[0], MINIMA["A"], atol=0.02)
    end_ok = np.allclose(smooth_path[-1], MINIMA["B"], atol=0.02)

    # Check path is smoother (reduced curvature variance)
    noisy_curv = np.std(np.diff(noisy_path, n=2, axis=0))
    smooth_curv = np.std(np.diff(smooth_path, n=2, axis=0))
    smoother = smooth_curv < noisy_curv * 0.8

    print(f"  Shape: {smooth_path.shape}")
    print(f"  Endpoints pinned: {start_ok and end_ok}")
    print(f"  Curvature std: noisy={noisy_curv:.4f}, smooth={smooth_curv:.4f}")
    print(f"  Smoother: {'✓' if smoother else '✗'}")

    all_ok = (smooth_path.shape == (100, 2)) and start_ok and end_ok and smoother
    print()
    return all_ok


def test_path_history_consistency():
    """Test 7: Verify path_history.npz contains sequential iterations."""
    print("=" * 60)
    print("TEST 7: Path history consistency")
    print("=" * 60)

    history_file = os.path.join(REFINE_DIR, "path_history.npz")

    if not os.path.exists(history_file):
        print("  ✗ path_history.npz not found")
        return False
    data = np.load(history_file)
    keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
    print(f"  Found {len(keys)} entries in history: {keys}")

    # Check they are sequential
    expected_keys = [f"iter_{i}" for i in range(len(keys))]
    keys_match = keys == expected_keys
    print(f"  Sequential keys: {'✓' if keys_match else '✗'}")
    print(f"    Found: {keys}")
    print(f"    Expected: {expected_keys}")

    # Check shapes match
    shapes_match = True
    for k in keys:
        s = data[k].shape
        if s[1] != 2:
            shapes_match = False
            print(f"    ✗ {k}: unexpected shape {s}")
    if shapes_match:
        print(f"  All entry shapes consistent: ✓")

    print()
    return keys_match and shapes_match


def main():
    print("\n" + "█" * 60)
    print("█  MULLER-BROWN PATH REFINEMENT — COMPREHENSIVE TEST")
    print("█" * 60 + "\n")

    tests = [
        ("Existing Results", test_existing_results),
        ("PathCV Construction", test_pathcv_construction),
        ("Refinement Convergence", test_refinement_convergence),
        ("Endpoint Pinning", test_endpoint_pinning),
        ("Physical Validity", test_physically_valid_path),
        ("PrincipalCurve Standalone", test_principal_curve_standalone),
        ("Path History Consistency", test_path_history_consistency),
    ]

    results = {}
    all_passed = True

    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
        except Exception as e:
            results[name] = f"ERROR: {e}"
            all_passed = False
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        if status == "PASS":
            print(f"  ✓ {name}: {status}")
        else:
            print(f"  ✗ {name}: {status}")

    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print()

    # Output machine-readable results
    result_data = {
        "status": "PASS" if all_passed else "FAIL",
        "tests": results,
    }
    result_file = os.path.join(BASE_DIR, "results", "test_results.json")
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to: {result_file}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
