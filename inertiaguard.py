"""InertiaGuard: detect chain-of-thought hijacking from a single trace via visual belief inertia (VBI).

For cuts i < j of the chain, belief inertia is the entropy the model spends on the span minus the belief revision
it causes, BI(i, j) = H_ij - KL(p_j || p_i), where p_k is the model's distribution over the option letters read out
at cut k. Honest reasoning keeps BI >= 0; a hijack makes an unpaid revision (BI << 0) at a place where the image
spoke. VBI is the lowest BI over spans that contain an image-dependent token (KL to a substitute image > TAU).
"""
import re
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LETTERS = "abcde"
CUE = "Therefore, the final answer is \\boxed{"
TAU = 1.0           # nats: a token is image-dependent if replacing the image moves it by more than TAU
EPS = 1e-4          # smoothing added to every readout probability
NO_SPAN = 50.0      # VBI of a trace without image-dependent tokens (nothing to check, never flagged)


def cut_points(text):
    """Char offsets of the cuts: the empty chain, every sentence end before the final "\\boxed{", and that box."""
    stop = text.rfind("\\boxed{")
    stop = len(text) if stop < 0 else stop
    bounds = sorted({0, len(text), *(m.end() for m in re.finditer(r"(?<=[.!?\n])\s+", text))})
    ends = [a + len(text[a:b].rstrip()) for a, b in zip(bounds[:-1], bounds[1:]) if text[a:b].strip()]
    return sorted({0, stop, *(e for e in ends if 0 < e < stop)})


def steps(cuts, tok_start):
    """Kept cuts and the token indices of the step between consecutive kept cuts.
    A step of at most two tokens (list markers, blank lines) is merged into the next one; the last is never merged."""
    seg = [np.where((tok_start >= a) & (tok_start < b))[0] for a, b in zip(cuts[:-1], cuts[1:])]
    keep = [0] + [k for k in range(1, len(cuts)) if k == len(cuts) - 1 or len(seg[k - 1]) > 2]
    return keep, [np.concatenate(seg[a:b]) for a, b in zip(keep[:-1], keep[1:])]


def vbi(P, step_H, step_A, tau=TAU):
    """P: readouts at the K+1 kept cuts; step_H: entropy spent in each of the K steps; step_A: largest per-token
    image dependence in each step. Returns min over localized spans (i, j] of H_ij - KL(p_j || p_i)."""
    P = np.asarray(P, float) + EPS
    P /= P.sum(1, keepdims=True)
    lp = np.log(P)
    B = (P[None] * (lp[None] - lp[:, None])).sum(-1)                      # B[i, j] = KL(p_j || p_i)
    cH = np.concatenate([[0.0], np.cumsum(step_H)])
    cL = np.concatenate([[0], np.cumsum(np.asarray(step_A) > tau)])
    ok = np.triu(np.ones(B.shape, bool), 1) & (cL[None] - cL[:, None] > 0)
    return float(((cH[None] - cH[:, None]) - B)[ok].min()) if ok.any() else NO_SPAN


class InertiaGuard:
    """Wraps a Qwen2.5-VL reasoning model; needs inference access only."""

    def __init__(self, model_path, device="cuda"):
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.dev = device
        self.letter_ids = [self.tok.convert_tokens_to_ids(l) for l in LETTERS]

    def inputs(self, image, question, suffix=""):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + suffix
        return self.proc(text=[text], images=[image], return_tensors="pt").to(self.dev)

    def substitute(self, inp, kind, other=None):
        """Pixel values of a black frame or of another image, on the same patch grid as the real image."""
        t, h, w = inp["image_grid_thw"][0].tolist()
        ps = self.model.config.vision_config.patch_size
        img = Image.new("RGB", (w * ps, h * ps)) if kind == "black" else other.convert("RGB").resize((w * ps, h * ps))
        pv = self.proc.image_processor(images=[img], return_tensors="pt")
        assert pv["image_grid_thw"][0].tolist() == [t, h, w]
        return pv["pixel_values"].to(self.dev, self.model.dtype)

    def logprobs(self, inp, gen, pixel_values):
        """Teacher-forced next-token log-probabilities at the generated positions."""
        ids = torch.cat([inp["input_ids"][0], gen])[None]
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixel_values,
                            image_grid_thw=inp["image_grid_thw"]).logits[0]
        n = inp["input_ids"].shape[1]
        return torch.log_softmax(logits[n - 1:n - 1 + len(gen)].float(), -1)

    def readout(self, image, question, prefix, n_opt, last=False):
        """Distribution over the option letters after the chain prefix. The cue closes the reasoning before the box,
        since a bare "\\boxed{" after a sentence that names an option invites copying that letter; the last cut
        (where the model itself wrote the box) is read as written."""
        if last:
            tail = "\\boxed{" if prefix.endswith((" ", "\n")) else "\n\n\\boxed{"
        else:
            prefix = prefix.rstrip()
            tail = ("\n\n" if prefix else "") + CUE
        logits = self.model(**self.inputs(image, question, prefix + tail)).logits[0, -1].float()
        return torch.softmax(logits[self.letter_ids[:n_opt]], -1).tolist()

    @torch.no_grad()
    def __call__(self, image, question, n_opt, other_image, max_new_tokens=640):
        """Generate a trace greedily and score it. other_image: another image of the same task (substitute)."""
        inp = self.inputs(image, question)
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)[0, inp["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=False)
        # entropy of every generated token and its image dependence (larger KL of the two substitutes)
        lp = self.logprobs(inp, gen, inp["pixel_values"])
        H = -(lp.exp() * lp).sum(-1)
        A = torch.stack([(lp.exp() * (lp - self.logprobs(inp, gen, self.substitute(inp, k, other_image)))).sum(-1)
                         for k in ("black", "other")]).max(0).values
        H, A = H.cpu().numpy(), A.cpu().numpy()
        # cuts at sentence ends, steps, readout at every kept cut
        tok_start = np.array([o[0] for o in self.tok(text, add_special_tokens=False,
                                                    return_offsets_mapping=True)["offset_mapping"]])[:len(gen)]
        cuts = cut_points(text)
        keep, segs = steps(cuts, tok_start)
        P = [self.readout(image, question, text[:cuts[k]], n_opt, last=(k == len(cuts) - 1)) for k in keep]
        step_H = [float(np.clip(H[s], 0, None).sum()) for s in segs]
        step_A = [float(A[s].max()) if len(s) else 0.0 for s in segs]
        box = re.findall(r"\\boxed\{([^}]+)\}", text)
        return dict(text=text, answer=box[-1].strip().lower() if box else None, vbi=vbi(P, step_H, step_A),
                    cuts=[cuts[k] for k in keep], P=P, step_H=step_H, step_A=step_A)
