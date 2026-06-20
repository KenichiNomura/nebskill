"""MLIP dispatcher — returns an ASE Calculator for any model in the registry."""
import sys
import torch


def _resolve_device(entry: dict) -> str:
    d = entry.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def make_calculator(mlip_name: str, registry: dict):
    """Instantiate an ASE Calculator for the given MLIP name.

    Parameters
    ----------
    mlip_name : str
        Key into the registry (e.g. "mace-mp", "chgnet", "nequip-oam-l").
    registry : dict
        Loaded from assets/mlip_registry.yaml.
    """
    if mlip_name not in registry:
        raise ValueError(
            f"Unknown MLIP '{mlip_name}'. "
            f"Available: {sorted(registry.keys())}"
        )
    entry = registry[mlip_name]
    pkg = entry["package"]
    device = _resolve_device(entry)

    # ------------------------------------------------------------------ MACE
    if pkg == "mace":
        from mace.calculators import mace_mp, mace_off, mace_omol
        model_size = entry.get("model_size", "medium")
        dtype = entry.get("dtype", "float64")
        model_type = entry.get("model_type", "mace-mp")
        if model_type == "mace-off":
            return mace_off(model=model_size, device=device, default_dtype=dtype)
        if model_type == "mace-omol":
            return mace_omol(model=model_size, device=device, default_dtype=dtype)
        return mace_mp(model=model_size, device=device, default_dtype=dtype)

    # --------------------------------------------------------------- CHGNet
    if pkg == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device=device)

    # ---------------------------------------------------------------- MatGL
    if pkg == "matgl":
        import matgl
        from matgl.ext.ase import PESCalculator
        model_name = entry.get("model_name", "M3GNet-MP-2021.2.8-PES")
        pot = matgl.load_model(model_name)
        return PESCalculator(potential=pot)

    # --------------------------------------------------------------- SevenNet
    if pkg == "sevenn":
        from sevenn.sevennet_calculator import SevenNetCalculator
        checkpoint = entry.get("checkpoint", "7net-0")
        modal = entry.get("modal", None)
        if modal:
            return SevenNetCalculator(checkpoint, device=device, modal=modal)
        return SevenNetCalculator(checkpoint, device=device)

    # ------------------------------------------------------------------ uPET
    if pkg == "upet":
        from upet.calculator import UPETCalculator
        model_name = entry.get("model_name", "pet-mad-s")
        version    = entry.get("version", "latest")
        return UPETCalculator(model=model_name, version=version, device=device)

    # ------------------------------------------------------------- fairchem
    if pkg == "fairchem":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        model_name = entry.get("model_name", "uma-s-1p2")
        task = entry.get("task", "omat")
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        return FAIRChemCalculator(predictor, task_name=task)

    # ------------------------------------------------------------------ TACE
    if pkg == "tace":
        from tace.interface.ase import TACEAseCalc
        model_path = entry.get("model_path")
        if not model_path:
            raise ValueError(f"TACE entry '{mlip_name}' requires 'model_path'")
        dtype = entry.get("dtype", "float32")
        return TACEAseCalc(model_path, device=device, dtype=dtype)

    # ---------------------------------------------- NequIP / Allegro (OAM)
    if pkg == "nequip":
        from pathlib import Path
        # e3nn 0.5.x loads constants.pt with torch.load() without weights_only=False.
        # PyTorch ≥ 2.6 changed the default to weights_only=True, which rejects slice.
        torch.serialization.add_safe_globals([slice])
        from nequip.integrations.ase import NequIPCalculator
        compiled = entry.get("compiled_model")
        if not compiled:
            raise ValueError(
                f"NequIP/Allegro entry '{mlip_name}' requires 'compiled_model'"
            )
        skill_root = Path(__file__).resolve().parent.parent
        compiled_path = (skill_root / compiled).resolve()
        return NequIPCalculator.from_compiled_model(str(compiled_path), device=device)

    raise ValueError(f"Unsupported package '{pkg}' for MLIP '{mlip_name}'")
