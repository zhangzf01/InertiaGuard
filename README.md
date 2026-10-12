# InertiaGuard

Code for the anonymous submission *Detecting Chain-of-Thought Hijacking in Backdoored Multimodal Reasoning Models*.

InertiaGuard flags a reasoning trace whose answer belief changes without being paid for by the entropy of the
reasoning that changes it. At every sentence end (cut) it reads the model's distribution over the option letters,
$p_k$, and for every span of steps between cuts $i<j$ computes the belief inertia

$$\mathrm{BI}(i,j) = \textstyle\sum_{t\in(i,j]} H_t \;-\; D_{\mathrm{KL}}(p_j \,\|\, p_i),$$

which honest (self-consistent, greedy) reasoning keeps non-negative. The visual belief inertia VBI is the minimum of
BI over spans that contain an image-dependent token, i.e. a token whose next-token distribution moves by more than
$\tau = 1$ nat when the image is replaced by a black frame or by another image. A trace is flagged if $-\mathrm{VBI}$
exceeds a threshold calibrated on clean traces (95th percentile of 50 clean validation traces).
Only inference access to the monitored model is needed.

## Files

| file | content |
|---|---|
| `inertiaguard.py` | the method: generation, teacher-forced passes (entropy, image dependence), cue readout, span search |
| `data.py` | A-OKVQA, ScienceQA, MathVista, MMMU (multiple choice), solver prompt, ReShift logo trigger |
| `run.py` | generate and score the clean and the triggered trace of every item |
| `evaluate.py` | hijack labels and calibrated TPR@5%FPR, realized FPR, AUROC |

## Usage

```bash
pip install -r requirements.txt

# score 200 items, clean and triggered (one GPU, Qwen2.5-VL backbone)
python run.py --model <backdoored checkpoint> --dataset aokvqa --data_root data --target c --out runs/aokvqa.jsonl

# calibrated detection metrics
python evaluate.py runs/aokvqa.jsonl --attack reshift_sft
```

`--target` is the attack's target letter (`c` for ReShift and BadToken, `a` for ReWire); items whose answer already is
the target are skipped. `--attack` selects the signature that marks a hijack (`reshift_sft`, `reshift_rl`, `badtoken`,
`rewire`; or pass `--signature <regex>`). See `data.py` for the expected dataset files.

Using the detector on its own:

```python
from inertiaguard import InertiaGuard
guard = InertiaGuard("<checkpoint>")
r = guard(image, prompt, n_opt=4, other_image=another_image_of_the_task)
r["vbi"]   # lower is more suspicious; compare -r["vbi"] with a threshold calibrated on clean traces
```

Each call returns the trace, its answer, VBI, and the readouts, entropies and image dependence per step, so other
values of $\tau$ can be scored without the GPU (`inertiaguard.vbi`).

Cost per trace, beyond generation: three teacher-forced passes (real image, black frame, another image) and one
short prefill per kept cut; the span search is negligible.
