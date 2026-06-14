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
        from mace.calculators import mace_mp, mace_off
        model_size = entry.get("model_size", "medium")
        dtype = entry.get("dtype", "float64")
        model_type = entry.get("model_type", "mace-mp")
        if model_type == "mace-off":
            return mace_off(model=model_size, device=device, default_dtype=dtype)
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
        model_name = entry.get("model_name", "pet-oam-xl")
        return UPETCalculator(model=model_name, device=device)

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
        from nequip.integrations.ase import NequIPCalculator
        compiled = entry.get("compiled_model")
        if not compiled:
            raise ValueError(
                f"NequIP/Allegro entry '{mlip_name}' requires 'compiled_model'"
            )
        return NequIPCalculator.from_compiled_model(compiled, device=device)

    # ------------------------------------------------------------------- Orb
    if pkg == "orb":
        import orb_models
        from orb_models.forcefield.calculator import ORBCalculator
        model_name = entry.get("model_name", "orb-v2")
        model = orb_models.load_model(model_name)
        return ORBCalculator(model, device=device)

    # ------------------------------------------------------------- MatterSim
    if pkg == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator
        model_name = entry.get("model_name", "MatterSim-v1.0.0-5M")
        return MatterSimCalculator(load_path=model_name, device=device)

    raise ValueError(f"Unsupported package '{pkg}' for MLIP '{mlip_name}'")
