# Step 3 — Run NEB

Interpolates NEB images between the relaxed endpoints and runs the two-phase
NEB calculation: standard NEB followed by Climbing Image NEB (CI-NEB).

## Script

```bash
python step3-neb/neb_runner.py --output-dir outputs/{run_id}/{mlip} \
    --mlip <mlip_name> --config assets/neb_defaults.yaml \
    --registry assets/mlip_registry.yaml
```

Reads `relaxed_endpoints.json` and `endpoints.json` from `--output-dir`.
Writes `neb_result.json` and `neb_trajectory.xyz` (updated after each
phase). Exits with code 4 if either phase fails to converge, signaling
`step4-monitor` to take over.

`--n-images`, `--method`, `--spring-constant`, `--optimizer` override the
config values (used by the adaptive retry loop in step 4).

## Multi-GPU NEB (`--n-gpus`)

`--n-gpus N` (default 4, from `neb.n_gpus` in the config — always use all
A100 GPUs on the node for one NEB job) spreads one NEB run's images across N
GPUs on a single node instead of evaluating them sequentially on one device.
Each image gets its own calculator instance pinned to `cuda:{i % N}`, and
the NEB object is built with `parallel=True, allow_shared_calculator=False`.
Since no MPI is configured on this system, this uses ASE's threaded
fallback (`ase.mep.neb.NEB`, `world.size == 1` branch): one Python thread
per image, each calling its own calculator concurrently — real multi-GPU
parallelism within a single process, no `srun`/`mpirun` wrapping needed.

Hard-capped at 4 (this system's GPUs/node) and requires CUDA; the script
exits with a clear error before doing any work if the request can't be
satisfied. `n_gpus=1` is byte-for-byte the original single-shared-calculator
serial path (still available by passing `--n-gpus 1` or setting `neb.n_gpus:
1` in the config).

Caveat: each GPU holds one full calculator (model weights) per image
scheduled to it, so very large models with many images use more VRAM in
aggregate than the single-shared-calculator path — test with the default
`n_images_min: 9` before scaling up.

## n_images calculation

Unless overridden by the user or agent:
```
path_length = sum of per-atom displacements between reactant and product (Å)
              (minimum-image convention if the system is periodic)
n_images = max(n_images_min, round(path_length / images_per_angstrom))
```

## Phase 1 — Standard NEB

1. Create `n_images` copies of the reactant Atoms object
2. Attach the chosen MLIP's calculator to each image (via
   [lib/calculator.py](../lib/calculator.py))
3. Set endpoint positions (first and last images fixed)
4. Run interpolation (`idpp` or `linear`, `mic=True` if periodic)
5. Create NEB object with `remove_rotation_and_translation` disabled for
   periodic systems, enabled otherwise
6. Run the chosen optimizer (`FIRE2` by default, or `MDMin`), max
   `phase1_max_steps` steps, until `fmax < phase1_fmax`
7. Save trajectory to `neb_trajectory.xyz`

If phase 1 exceeds max steps without converging → trigger step 4
(monitor/retry).

## Phase 2 — Climbing Image NEB (CI-NEB)

Only runs after phase 1 converges. Continues from phase 1 final positions.

1. Set `climb=True` on the same NEB object (same images, same calculators)
2. Run the chosen optimizer, max `phase2_max_steps` steps, until
   `fmax < phase2_fmax`
3. Append to `neb_trajectory.xyz`

If phase 2 exceeds max steps → trigger step 4 (monitor/retry).

## neb_result.json

```json
{
  "n_images": 9,
  "method": "improvedtangent",
  "spring_constant": 0.1,
  "optimizer": "FIRE2",
  "dft_barrier_ev": null,
  "phase1": { "phase": 1, "converged": true, "...": "..." },
  "latest":  { "phase": 2, "converged": true, "...": "..." }
}
```

## Optimizer fallback

If the agent selects `switch_optimizer(optimizer="MDMin")` on retry, NEB is
re-run with `MDMin` instead of the default `FIRE2`, which tends to perform
better when energy kinking is detected (more tolerant of noisy/
discontinuous forces). `method` (tangent scheme) is fixed and not
LLM-adjustable.

See [references/neb_method.md](../references/neb_method.md) for full
details.
