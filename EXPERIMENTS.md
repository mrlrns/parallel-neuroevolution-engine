# Experiment log

Running record of hypotheses, settings and outcomes. Newest last.
Negative results are kept — they are the ones that constrain the search.

---

## E0 — Baseline: reward telescoping produces a degenerate policy

2026/06

**Observation.** Champions maximised score without producing sustained swimming:
a single initial impulse followed by passive drift.

**Diagnosis.** The per-step reward is `progres = distance_t − distance_{t−1}`,
accumulated over the episode. The sum telescopes:

```
Σ (d_t − d_{t−1}) = d_final − d_initial
```

The optimised objective is therefore **net displacement**, indifferent to how it is
obtained. Combined with a low drag coefficient (`water_drag = 0.005`), a launched body
drifts for a long time, so one impulse pays as much as sustained swimming at a fraction
of the energy cost. The agent was doing exactly what was asked.

**Secondary issue.** `apply_action` used `0.85 · base − 0.3 · base · action`, so an action
of zero still commanded a 15% contraction — a free initial impulse, since the energy term
`|base · action|` charged nothing for it.

**Conclusion.** Specification problem, not an optimisation problem.

---

## E1 — Fix the neutral point and the energy metric

2026/08/05

**Changes.**
- `new_lengths = base − 0.3 · base · action` (neutral point back at natural length)
- `energy_step = Σ |new_lengths − base|` (deviation from natural length, not command amplitude)

**Result.** Phase 1 champion reaches **240.8** at generation 14 (11 nodes, 16 links:
9 muscles, 7 bones). Replay confirms **actual swimming**: the body deforms continuously
and translates, no longer a single impulse.

**Caveat found while replaying.** `visualize.py` had not been updated with the new
`apply_action`, so the first replay showed a frozen creature. Training and replay code
must be kept in sync — a replay discrepancy looks exactly like a training failure.

**Side effect to keep in mind.** `|new_lengths − base| = 0.3 · base · |action|`, so the new
penalty is exactly 0.3× the previous one. At unchanged `coeff_energie`, energetic pressure
was divided by ~3.3. Before/after comparisons are not on the same scale.

**Open issue.** The within-generation *mean* degrades even as the population maximum rises
(e.g. gen 16: 81.8 → 57.7 → 30.7 → 26.2). `best_score` is a running maximum over ~900
noisy samples and only ever increases, so it is not evidence of learning. The mean is the
honest metric — and it goes the wrong way.

---

## E2 — Phase 2 refinement degrades the policy

2026/08/10
**Setup.** `train2.py`, champion gen 14, `BATCH_SIZE = 600`, `FRAME_NB = 300`,
truncated BPTT every 60 frames, Adam `lr = 1e-3`, exploration noise annealed 0.030 → 0.020.

| episode | mean | mean distance |
|---|---|---|
| 0 | 116.7 | 126.1 |
| 20 | 94.1 | 101.9 |
| 50 | 82.7 | 90.4 |
| 100 | 72.2 | 79.5 |

**Result.** Mean performance drops ~38% over 100 episodes, near-monotonically. `score max`
stays flat around 380–410: the distribution spreads rather than shifts. Exploration noise
*decreases* over the same window, so it cannot explain the loss.

**Conclusion.** Gradient descent is actively degrading the objective it optimises.

---

## E3 — Control: is the optimiser the cause?

2026/08/12
**Setup.** Identical to E2 with `LEARNING_RATE = 0`. Nothing else changed.

| episode | mean | mean distance |
|---|---|---|
| 0 | 117.5 | 127.1 |
| 20 | 116.1 | 125.8 |
| 30 | 120.7 | 130.4 |
| 50 | 115.5 | 125.0 |
| 60 | 125.4 | 135.3 |

**Result.** Mean is stable at 115–125 across 60 episodes. The only difference from E2 is
whether the gradient is applied.

**Conclusion.** **The Adam updates are the cause of the degradation.** This rules out
state accumulation between episodes, simulator instability, and the exploration schedule.

**Bonus observation.** At `lr = 0` the mean drifts slightly upward (117 → 125) as noise
anneals (0.030 → 0.024) — less perturbation, better performance. The annealing schedule
behaves as intended.

---

## E4 — Learning rate sweep: bad step size or bad direction?

**Setup.** Batch 2000, 50 episodes per run, one parameter changed at a time.
Exploration noise annealed identically across all runs (0.030 → 0.025), so it
introduces no bias between them.

| lr | mean @ 0 | mean @ 50 | shape |
|---|---|---|---|
| 0 (control) | 117.5 | 115.5 | flat |
| 1e-5 | 125.8 | 123.4 | flat, indistinguishable from control |
| 5e-5 | 112.7 | 137.6 | slow, still rising at 50 |
| 1e-4 | 116.6 | 142.9 | rises to ~148 by ep 40 |
| 3e-4 | 116.1 | 132.8 | peaks 147 at ep 20, then declines |
| 5e-4 | 113.5 | 110.3 | peaks 141 at ep 10, then collapses |
| 1e-3 | 116.7 | 82.7 | monotone collapse |

**Conclusion.** Classic step-size signature: too large diverges, too small does
nothing, and there is a working band in between. The gradient **direction** is
sound — at 1e-4 it improves performance by ~23% net of the control drift.
`clip_grad_norm_(1.0)` bounds the gradient norm but preserves its direction, so
it offers no protection against an oversized step.

**Secondary finding.** The larger the step, the earlier the peak and the steeper
the subsequent decline. A fixed learning rate cannot be optimal across the whole
run: the policy improves until the step becomes too large relative to local
curvature, then diverges.

---

## E5 — Long runs: the 50-episode ranking is misleading

**Setup.** 150 episodes, batch 2000, lr = 1e-4 vs lr = 5e-5.

| episode | 1e-4 | 5e-5 |
|---|---|---|
| 0 | 114.5 | 116.7 |
| 50 | 150.3 | 135.8 |
| 90 | 147.4 | 151.7 |
| 150 | 147.4 | 157.0 |

**Result.** `lr = 1e-4` plateaus at ~150 from episode 50 onward — the following
100 episodes gain nothing. `lr = 5e-5` rises steadily to 157.0 and is **still
climbing at episode 150** (+34% from start). The curves cross around episode 90.

**Conclusion.** A sweep truncated at 50 episodes would have selected the wrong
setting. Short-horizon comparisons are unreliable here.

**Caveat.** Two separate runs at lr = 1e-4 (E4 and E5) start at 116.6 and 114.5
and follow visibly different trajectories (145.6 vs 139.5 at episode 20). Run-to-
run variance is real and single runs should not be over-interpreted.

---

## E6 — Planned: 5e-5 to convergence, and a decaying schedule

**Hypotheses.**
1. `lr = 5e-5` has not yet plateaued at 150 episodes. Run to 300 to locate it.
2. A decaying schedule (1e-4 → 2e-5) should capture the fast early rise of 1e-4
   while avoiding the stall, given the peak-then-decline pattern from E4.



## Open questions

- Is the vertical drift observed in replay (the body sinks as it advances) contributing to
  the displacement reward? If the motion is partly a diagonal fall, distance overstates
  swimming performance.
- The `coef_hauteur` penalty introduced at generation 15 penalises `|Δ| of the barycentre's
  vertical position`. Undulatory swimming necessarily moves the barycentre vertically —
  this may be penalising the target behaviour. Worth testing a progressive schedule, or
  removing it entirely.
- Adam is re-instantiated at every generation in `train.py`, so its moment estimates reset
  roughly every 20 episodes. Unavoidable in part (parameter shapes change under mutation),
  but it means phase 1 never runs a warm optimiser.
- Mutation operators are asymmetric: `rate_new_node = 0.1` adds nodes, nothing
  ever removes them. Structural bloat. Needs a mirrored-pair deletion operator
  with a connectivity guard and index remapping, and/or a parsimony penalty.

## Related work to read

- Xu et al., *Accelerated Policy Learning with Parallel Differentiable Simulation* —
  documents exactly the E2/E3 symptom: gradients through stiff differentiable simulators
  are high-variance and biased over long horizons. Proposes short-horizon rollouts with a
  learned critic for the tail.
- Ma et al., *DiffAqua: A Differentiable Computational Design Pipeline for Soft Underwater
  Swimmers* (SIGGRAPH 2021) — the closest published analogue to this project.
