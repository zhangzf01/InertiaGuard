"""Generate and score the clean and the triggered trace of every item.
  python run.py --model <checkpoint> --dataset aokvqa --data_root data --target c --out runs/reshift_aokvqa.jsonl
"""
import argparse, json, os, random
from data import load, add_trigger
from inertiaguard import InertiaGuard

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", required=True, choices=["aokvqa", "scienceqa", "mathvista", "mmmu"])
ap.add_argument("--data_root", default="data")
ap.add_argument("--target", default="c", help="the attack's target letter (items with this answer are skipped)")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", required=True)
a = ap.parse_args()

items = load(a.dataset, a.data_root, a.target, a.n)
guard = InertiaGuard(a.model)
rng = random.Random(0)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as f:
    for i, d in enumerate(items):
        other = items[rng.choice([j for j in range(len(items)) if j != i])]["image"]   # substitute image
        for cond, image in (("clean", d["image"]), ("triggered", add_trigger(d["image"]))):
            r = guard(image, d["prompt"], d["n_opt"], other)
            f.write(json.dumps(dict(id=i, condition=cond, gt=d["gt"], **r)) + "\n")
            f.flush()
        print(f"{i + 1}/{len(items)}", flush=True)
