# NEB Method Reference

## What NEB does

The Nudged Elastic Band (NEB) method finds the Minimum Energy Path (MEP)
between two known states (reactant and product) on a potential energy surface.
A chain of images is connected by spring forces, then relaxed so that the
images distribute themselves along the MEP.

## Key ASE parameters

| Parameter | Description | Our default |
|---|---|---|
| `k` | Spring constant (eV/Å) | 0.1 |
| `method` | Tangent estimation method | `improvedtangent` |
| `climb` | Enable climbing image | False (phase 1), True (phase 2) |
| `remove_rotation_and_translation` | NEB-TR for non-periodic systems | True |
| `allow_shared_calculator` | Share calculator between images | False |

## Methods

- **`improvedtangent`** (default): smooth tangent using neighbor energies.
  Robust and well-tested. Recommended starting point.
- **`string`**: minimizes the band energy using string method. Works well
  with preconditioning (`precon='Exp'`) for organic molecules with varying
  bond stiffness. Use as fallback on kinking failures.
- **`aseneb`**: legacy method, avoid.
- **`eb`**: full elastic band, rarely needed.

## Interpolation

Always use IDPP (`neb.interpolate('idpp')`) over linear interpolation.
IDPP minimizes a cost function on interatomic distances, producing fewer
atom collisions in the initial path for organic molecules.

## Two-phase CI-NEB protocol

Phase 1 (standard NEB):
- Distributes images along the MEP
- Converge to `fmax < 0.3 eV/Å` before enabling climbing image
- Ensures accurate tangent estimates for the climbing image

Phase 2 (CI-NEB, `climb=True`):
- The highest-energy image feels no spring forces
- Its parallel force component is inverted, pushing it toward the saddle point
- Converge to `fmax < 0.05 eV/Å`

## Optimizer choice

Use FIRE for both phases. BFGS and L-BFGS are unsuitable for CI-NEB because
the NEB force is not a true gradient of any scalar function.

## Common failure modes and interventions

| Symptom | Diagnosis | Fix |
|---|---|---|
| Low inter-image RMSD (< 0.05 Å) | Image collapse | Increase `k` |
| Highly uneven inter-image RMSD | Image bunching | Increase `n_images` |
| Large energy second derivative | Kinking | Switch to `string` method |
| High forces at image 0 or N-1 | Endpoint not at minimum | Tighten endpoint relaxation |
| Steps ≈ cap, fmax slowly decreasing | Almost converged | Increase step cap or reduce `k` |

## References

- Henkelman & Jónsson, J. Chem. Phys. 113, 9978 (2000) — improved tangent NEB
- Henkelman et al., J. Chem. Phys. 113, 9901 (2000) — climbing image NEB
- ASE NEB documentation: https://ase-lib.org/ase/neb.html
