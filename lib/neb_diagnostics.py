"""Diagnose NEB convergence failures from neb_result.json."""
import numpy as np


# Thresholds for failure-mode classification
_COLLAPSE_RMSD   = 0.05   # Å  — min inter-image RMSD below this → collapse
_KINK_D2         = 1.0    # eV — max |d²E/di²| above this → kinking
_BUNCHING_CV     = 0.5    # coefficient of variation above this → bunching
_ENDPOINT_RATIO  = 0.8    # endpoint fmax / max fmax above this → endpoint issue


def diagnose(neb_result: dict) -> dict:
    """Compute diagnostics from the latest NEB result.

    Parameters
    ----------
    neb_result : dict
        Contents of neb_result.json written by neb_runner.py.

    Returns
    -------
    dict
        Flat diagnostic payload consumed by retry.py and diagnostics.py.
    """
    latest = neb_result["latest"]
    energies  = latest["energies"]            # all images incl. endpoints
    img_fmax  = latest["forces_per_image"]    # internal images only
    rmsds     = latest["inter_image_rmsd"]    # n_images-1 values

    n_images  = neb_result["n_images"]
    method    = neb_result["method"]
    k         = neb_result["spring_constant"]
    phase     = latest["phase"]
    fmax_fin  = latest["fmax_final"]
    fmax_tgt  = latest["fmax_target"]
    steps     = latest["steps_taken"]
    max_steps = latest["max_steps"]

    # ── energy smoothness: second differences of the energy profile ──────────
    e_arr = np.array(energies, dtype=float)
    d2    = np.diff(e_arr, n=2)
    energy_smoothness = {
        "values":      d2.tolist(),
        "max_abs_d2":  float(np.max(np.abs(d2))) if len(d2) else 0.0,
    }

    # ── inter-image spacing ───────────────────────────────────────────────────
    r_arr    = np.array(rmsds, dtype=float)
    mean_r   = float(np.mean(r_arr)) if len(r_arr) else 0.0
    std_r    = float(np.std(r_arr))  if len(r_arr) else 0.0
    cv       = std_r / mean_r if mean_r > 1e-8 else 0.0
    image_spacing = {
        "values":    rmsds,
        "min_rmsd":  float(np.min(r_arr)) if len(r_arr) else 0.0,
        "max_rmsd":  float(np.max(r_arr)) if len(r_arr) else 0.0,
        "mean_rmsd": mean_r,
        "cv":        cv,
    }

    # ── failure-mode classification ───────────────────────────────────────────
    kink_score  = energy_smoothness["max_abs_d2"]
    min_rmsd    = image_spacing["min_rmsd"]

    if img_fmax:
        endpoint_fmax = max(img_fmax[0], img_fmax[-1])
        max_fmax      = max(img_fmax)
        ep_ratio      = endpoint_fmax / max_fmax if max_fmax > 0 else 0.0
    else:
        ep_ratio = 0.0

    if min_rmsd < _COLLAPSE_RMSD:
        failure_mode = "image_collapse"
    elif kink_score > _KINK_D2:
        failure_mode = "kinking"
    elif cv > _BUNCHING_CV:
        failure_mode = "bunching"
    elif ep_ratio > _ENDPOINT_RATIO:
        failure_mode = "high_endpoint_forces"
    else:
        failure_mode = "slow_convergence"

    return {
        "failure_mode":     failure_mode,
        "phase":            phase,
        "fmax_final":       fmax_fin,
        "fmax_target":      fmax_tgt,
        "steps_taken":      steps,
        "max_steps":        max_steps,
        "n_images":         n_images,
        "method":           method,
        "spring_constant":  k,
        "per_image_fmax":   img_fmax,
        "energy_smoothness": energy_smoothness,
        "image_spacing":    image_spacing,
    }
