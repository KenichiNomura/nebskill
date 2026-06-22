"""MLIP dispatcher — returns an ASE Calculator for any model in the registry."""
import sys
import torch


def _resolve_device(entry: dict) -> str:
    d = entry.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def make_calculator(mlip_name: str, registry: dict, device_override: str | None = None):
    """Instantiate an ASE Calculator for the given MLIP name.

    Parameters
    ----------
    mlip_name : str
        Key into the registry (e.g. "mace-mp", "chgnet", "nequip-oam-l").
    registry : dict
        Loaded from assets/mlip_registry.yaml.
    device_override : str, optional
        Use this device string verbatim (e.g. "cuda:2") instead of the
        registry entry's "device" field. Used to pin a specific image's
        calculator to a specific GPU for multi-GPU NEB.
    """
    if mlip_name not in registry:
        raise ValueError(
            f"Unknown MLIP '{mlip_name}'. "
            f"Available: {sorted(registry.keys())}"
        )
    entry = registry[mlip_name]
    pkg = entry["package"]
    device = device_override or _resolve_device(entry)

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
        model_name = entry.get("model_name", "pet-mad-s")
        version    = entry.get("version", "latest")
        return UPETCalculator(model=model_name, version=version, device=device)

    # ------------------------------------------------------------- fairchem
    if pkg == "fairchem":
        from fairchem.core import FAIRChemCalculator
        task = entry.get("task", "omat")
        checkpoint = entry.get("checkpoint")
        # MLIPPredictUnit._setup_device only accepts bare "cpu"/"cuda" (an
        # indexed "cuda:N" trips its assert) and resolves "cuda" to
        # torch.cuda.current_device() once at construction time
        # (fairchem.core.common.distutils.get_device_for_local_rank), so for
        # multi-GPU NEB we pin the current device first, then pass the bare
        # string — same "current device" pattern as the nequip branch below.
        fc_device = "cpu"
        if device.startswith("cuda"):
            if ":" in device:
                torch.cuda.set_device(int(device.split(":", 1)[1]))
            fc_device = "cuda"
        if checkpoint:
            from pathlib import Path
            from fairchem.core.units.mlip_unit import load_predict_unit
            # Same PyTorch >=2.6 weights_only default issue as the nequip
            # branch below: the checkpoint pickles a `slice` object.
            torch.serialization.add_safe_globals([slice])
            predictor = load_predict_unit(str(Path(checkpoint).expanduser()), device=fc_device)
        else:
            from fairchem.core import pretrained_mlip
            model_name = entry.get("model_name", "uma-s-1p2")
            predictor = pretrained_mlip.get_predict_unit(model_name, device=fc_device)
        return FAIRChemCalculator(predictor, task_name=task)

    # ----------------------------------------------------------------- MatRIS
    if pkg == "matris":
        from matris.applications.base import MatRISCalculator
        model_name = entry.get("model_name", "matris_10m_oam")
        task = entry.get("task", "efs")
        return MatRISCalculator(model=model_name, task=task, device=device)

    # --------------------------------------------------------------- DeepMD
    if pkg == "deepmd":
        from pathlib import Path
        from deepmd.calculator import DP
        model_path = entry.get("model_path")
        if not model_path:
            raise ValueError(f"DeepMD entry '{mlip_name}' requires 'model_path'")
        skill_root = Path(__file__).resolve().parent.parent
        model_path = str((skill_root / model_path).resolve())
        head = entry.get("head")
        return DP(model_path, head=head)

    # ------------------------------------------------------------ EquiformerV3
    if pkg == "equiformer_v3":
        import sys
        from pathlib import Path
        repo_root = entry.get("repo_root")
        if not repo_root:
            raise ValueError(f"EquiformerV3 entry '{mlip_name}' requires 'repo_root'")
        exp_dir = str(Path(repo_root).expanduser() / "experimental")
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)
        # Triggers @registry.register_model("equiformer_v3"). fairchem's own
        # setup_imports() only auto-discovers an experimental/ dir relative to
        # the installed package root, not this external clone, so the model
        # class must be imported explicitly before OCPCalculator can resolve it.
        import models.equiformer_v3.equiformer_v3  # noqa: F401
        # Same PyTorch >=2.6 weights_only default issue as the fairchem/nequip
        # branches: the checkpoint pickles a `slice` object.
        torch.serialization.add_safe_globals([slice])
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        checkpoint = entry.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"EquiformerV3 entry '{mlip_name}' requires 'checkpoint'")
        return OCPCalculator(
            checkpoint_path=str(Path(checkpoint).expanduser()),
            cpu=(device == "cpu"),
        )

    # ------------------------------------------------------------------ TACE
    if pkg == "tace":
        from pathlib import Path
        from tace.interface.ase import TACEAseCalc
        model_path = entry.get("model_path")
        if not model_path:
            raise ValueError(f"TACE entry '{mlip_name}' requires 'model_path'")
        skill_root = Path(__file__).resolve().parent.parent
        model_path = str((skill_root / model_path).resolve())
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

        # AOTInductor-compiled .nequip.pt2 artifacts were compiled for the bare
        # "cuda" device string and reject any indexed "cuda:N" string outright
        # (strict equality check against compile-time metadata in
        # nequip/model/inference_models/aotinductor.py). The actual physical
        # GPU is instead selected by whatever device is "current" — both at
        # load time (model.to("cuda") inside the constructor) and on every
        # later calculate() call (self.device is re-resolved each time via
        # AtomicDataDict.to_(data, self.device)), and "current device" is
        # thread-local. So for multi-GPU NEB (one calculator per image,
        # forces computed from per-image threads) we must pin each call,
        # not just the load.
        if device_override and device_override.startswith("cuda:"):
            gpu_index = int(device_override.split(":", 1)[1])
            torch.cuda.set_device(gpu_index)
            calc = NequIPCalculator.from_compiled_model(str(compiled_path), device="cuda")
            original_calculate = calc.calculate

            def _calculate_pinned(*args, **kwargs):
                torch.cuda.set_device(gpu_index)
                return original_calculate(*args, **kwargs)

            calc.calculate = _calculate_pinned
            return calc

        return NequIPCalculator.from_compiled_model(str(compiled_path), device=device)

    raise ValueError(f"Unsupported package '{pkg}' for MLIP '{mlip_name}'")
