# NEB Method Reference

## What NEB does

The Nudged Elastic Band (NEB) method finds the Minimum Energy Path (MEP)
between two known states (reactant and product) on a potential energy surface.
A chain of images is connected by spring forces, then relaxed so that the
images distribute themselves along the MEP.

## Key ASE parameters

| Parameter | Description | Our default |
|---|---|---|
| `k` | Spring constant (eV/Å) | 0.5 |
| `method` | Tangent estimation method | `improvedtangent` |
| `climb` | Enable climbing image | False (phase 1), True (phase 2) |
| `remove_rotation_and_translation` | NEB-TR for non-periodic systems | True |
| `allow_shared_calculator` | Share calculator between images | True (sequential NEB; all images use one calculator instance) |

## Methods

- **`improvedtangent`** (default): smooth tangent using neighbor energies.
  Robust and well-tested. Recommended starting point.
- **`string`**: minimizes the band energy using string method. Works well
  with preconditioning (`precon='Exp'`) for organic molecules with varying
  bond stiffness. Use as fallback on kinking failures.
- **`aseneb`**: legacy method, avoid.
- **`eb`**: full elastic band, rarely needed.

## Interpolation

`assets/neb_defaults.yaml`'s default is `linear`, not IDPP — linear is
safer for periodic cells (IDPP's distance-cost minimization can fight the
minimum-image convention and produce a worse starting path under periodic
boundary conditions). For isolated/molecular (non-periodic) systems, IDPP
(`neb.interpolate('idpp')`) is generally preferable: it minimizes a cost
function on interatomic distances, producing fewer atom collisions in the
initial path. Set `interpolation: idpp` in the config if your system is
non-periodic and the linear default isn't converging well.

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

FIRE2 is the default optimizer for both phases. MDMin is the only other
option offered to the retry agent (`switch_optimizer`), used when FIRE2
stalls (e.g. kinking). BFGS and L-BFGS are deliberately excluded: they are
unsuitable for CI-NEB because the NEB force is not a true gradient of any
scalar function, and this breaks down further once the climbing image's
parallel force component is inverted in phase 2.

## Common failure modes and interventions

| Symptom | Diagnosis | Fix |
|---|---|---|
| Low inter-image RMSD (< 0.05 Å) | Image collapse | Increase `k` |
| Highly uneven inter-image RMSD | Image bunching | Increase `n_images` |
| Large energy second derivative | Kinking | Switch optimizer to `MDMin` |
| High forces at image 0 or N-1 | Endpoint not at minimum | Tighten endpoint relaxation |
| Steps ≈ cap, fmax slowly decreasing | Almost converged | Increase step cap or reduce `k` |

## References

- Henkelman & Jónsson, J. Chem. Phys. 113, 9978 (2000) — improved tangent NEB
- Henkelman et al., J. Chem. Phys. 113, 9901 (2000) — climbing image NEB
- ASE NEB documentation: https://ase-lib.org/ase/neb.html
