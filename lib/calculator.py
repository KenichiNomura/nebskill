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

    # --------------------------------- NequIP 0.6.x / Allegro 0.3.x (deployed .pt)
    if pkg == "nequip-legacy":
        from pathlib import Path
        # e3nn 0.5.x loads constants.pt with torch.load() without weights_only=False.
        # PyTorch ≥ 2.6 changed the default to weights_only=True, which rejects slice.
        torch.serialization.add_safe_globals([slice])
        from nequip.ase.nequip_calculator import NequIPCalculator
        from nequip.data import AtomicData, AtomicDataDict
        from ase.calculators.calculator import all_changes, Calculator

        deployed = entry.get("deployed_model")
        if not deployed:
            raise ValueError(
                f"nequip-legacy entry '{mlip_name}' requires 'deployed_model'"
            )
        skill_root = Path(__file__).resolve().parent.parent
        deployed_path = (skill_root / deployed).resolve()
        if not deployed_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {deployed_path}\n"
                f"Run: python step0-models/download.py --model {mlip_name}"
            )
        species_map = entry.get("species_to_type_name") or None
        base = NequIPCalculator.from_deployed_model(
            str(deployed_path),
            species_to_type_name=species_map,
            device=device,
        )

        # NequIP 0.6.1 + e3nn 0.5.x produces atom_types with shape [N, 1] via
        # TypeMapper, but the saved TorchScript model expects shape [N].  Subclass
        # to squeeze before the forward pass.
        class _AllegroFMCalc(NequIPCalculator):
            def calculate(self, atoms=None, properties=["energy"],
                          system_changes=all_changes):
                Calculator.calculate(self, atoms)
                data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
                for k in AtomicDataDict.ALL_ENERGY_KEYS:
                    if k in data:
                        del data[k]
                data = self.transform(data)
                data = data.to(self.device)
                data = AtomicData.to_AtomicDataDict(data)
                # fix [N, 1] → [N]
                if "atom_types" in data and data["atom_types"].dim() == 2:
                    data["atom_types"] = data["atom_types"].squeeze(-1)
                out = self.model(data)
                self.results = {}
                if AtomicDataDict.TOTAL_ENERGY_KEY in out:
                    self.results["energy"] = self.energy_units_to_eV * (
                        out[AtomicDataDict.TOTAL_ENERGY_KEY]
                        .detach().cpu().numpy().reshape(tuple())
                    )
                    self.results["free_energy"] = self.results["energy"]
                if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
                    self.results["energies"] = self.energy_units_to_eV * (
                        out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                        .detach().squeeze(-1).cpu().numpy()
                    )
                if AtomicDataDict.FORCE_KEY in out:
                    self.results["forces"] = (
                        self.energy_units_to_eV / self.length_units_to_A
                    ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
                if AtomicDataDict.STRESS_KEY in out:
                    from ase.stress import full_3x3_to_voigt_6_stress
                    stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
                    stress = stress.reshape(3, 3) * (
                        self.energy_units_to_eV / self.length_units_to_A ** 3
                    )
                    self.results["stress"] = full_3x3_to_voigt_6_stress(stress)

        calc = _AllegroFMCalc.__new__(_AllegroFMCalc)
        calc.__dict__.update(base.__dict__)
        return calc

    # ---------------------------------------------- NequIP / Allegro (OAM)
    if pkg == "nequip":
        # e3nn 0.5.x loads constants.pt with torch.load() without weights_only=False.
        # PyTorch ≥ 2.6 changed the default to weights_only=True, which rejects slice.
        torch.serialization.add_safe_globals([slice])
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
