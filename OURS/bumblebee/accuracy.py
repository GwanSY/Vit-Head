"""Plaintext ViT accuracy, signed Linear calibration, and fixed head proposals."""
import csv
import json
from pathlib import Path
import types
import numpy as np
from runtime import ROOT, digest, write


def run(out, backend="cpu"):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoImageProcessor, ViTForImageClassification

    config = json.loads((ROOT / "config.json").read_text())
    device = "cuda" if backend == "gpu" else "cpu"
    model_dir = (ROOT / config["model"]).resolve()
    data = ROOT.parents[1] / "datasets/imagenet2012/val"
    split_dir = ROOT.parent / "blb_base/outputs/heterogeneous_attention_20261009/splits"
    manifests = {name: list(csv.reader((split_dir / f"{name}.csv").open()))
                 for name in ("calibration", "selection")}
    assert len(manifests["calibration"]) == 2000 and len(manifests["selection"]) == 5000
    assert not set(r[0] for r in manifests["calibration"]) & set(r[0] for r in manifests["selection"])
    out.mkdir(parents=True, exist_ok=False)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)

    class Images(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            path, label = self.rows[index]
            with Image.open(data / path) as image:
                pixels = processor(image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            return pixels, int(label)

    def loader(name):
        return DataLoader(Images(manifests[name]), batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=device == "cuda")

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ViTForImageClassification.from_pretrained(
        model_dir, local_files_only=True, attn_implementation="eager").eval().to(device)
    modules = [layer.attention.attention for layer in model.vit.encoder.layer]
    original = [module.forward for module in modules]
    sums = torch.zeros((4, 12, 12), dtype=torch.float64, device=device)
    state = {"calibrate": True, "plan": None}

    def install(layer, module):
        def forward(self, hidden_states, head_mask=None, output_attentions=False):
            q, k, v = [self.transpose_for_scores(project(hidden_states))
                       for project in (self.query, self.key, self.value)]
            if layer == 0 and state.get("capture"):
                state["qkv"] = {name:value.cpu().numpy() for name,value in zip(("q","k","v"),(q,k,v))}
            z = (q / 8) @ k.transpose(-1, -2)
            s = z.softmax(dim=-1)
            if state["calibrate"]:
                a, b = s @ v, z @ v
                r = z.clamp_min(0)
                denominator = r.sum(dim=-1, keepdim=True)
                r = r / torch.where(denominator > 0, denominator, 1)
                residual = r @ v - a
                for index, value in enumerate((a * b, b * b, a * a, residual * residual)):
                    sums[index, layer] += value.sum(dim=(0, 2, 3), dtype=torch.float64)
            plan = state["plan"]
            if plan:
                weights = []
                for head, kind in enumerate(plan["heads"][layer]):
                    if kind == "S":
                        w = s[:, head]
                    elif kind == "R":
                        r = z[:, head].clamp_min(0)
                        denominator = r.sum(dim=-1, keepdim=True)
                        w = r / torch.where(denominator > 0, denominator, 1)
                    else:
                        w = z[:, head] * plan["alpha"][layer][head]
                    weights.append(w)
                s = torch.stack(weights, dim=1)
            context = (s @ v).permute(0, 2, 1, 3).reshape(hidden_states.shape)
            return (context, s) if output_attentions else (context,)
        module.forward = types.MethodType(forward, module)

    for layer, module in enumerate(modules): install(layer, module)
    with torch.inference_mode():
        for pixels, _ in loader("calibration"):
            model(pixels.to(device, non_blocking=True))
    values = sums.cpu().numpy()
    assert np.isfinite(values).all() and (values[1] > 0).all()
    alpha = values[0] / values[1]
    write(out / "alpha.json", dict(alpha=alpha.tolist(),samples=2000,
          manifest_sha256=digest(split_dir / "calibration.csv"),
          formula="sum <ZV,softmax(Z)V> / sum ||ZV||^2; signed; no clipping"))
    proxy = {"R": values[3] / values[2],
             "L": np.maximum(values[2] - values[0] ** 2 / values[1], 0) / values[2]}
    state["calibrate"] = False
    with Image.open((ROOT / config["image"]).resolve()) as image:
        pixels = processor(image.convert("RGB"),return_tensors="pt")["pixel_values"].to(device)
    state["capture"] = True
    with torch.inference_mode(): model(pixels)
    np.savez_compressed(out / "attention.npz",**state.pop("qkv"))
    state["capture"] = False
    plans = {}

    def add(name, changes):
        heads = [["S"] * 12 for _ in range(12)]
        for layer, head, kind in changes: heads[layer][head] = kind
        plans[name] = dict(heads=heads, alpha=alpha.tolist())

    add("S", [])
    add("R1", [(0, 0, "R")])
    add("L1", [(0, 6, "L")])
    add("M2", [(0, 0, "R"), (0, 6, "L")])
    add("R12", [(layer, 0, "R") for layer in range(12)])
    add("L12", [(layer, 6, "L") for layer in range(12)])
    add("M24", [(layer, head, kind) for layer in range(12) for head, kind in ((0,"R"),(6,"L"))])
    # These proposals use calibration errors only. Secure costs rank them later;
    # selection accuracy is an empirical constraint, not a holdout guarantee.
    ranked = sorted((float(proxy[kind][layer,head]),layer,head,kind)
                    for kind in "RL" for layer in range(12) for head in range(12))
    chosen, seen = [], set()
    for _, layer, head, kind in ranked:
        if (layer,head) not in seen:
            chosen.append((layer,head,kind)); seen.add((layer,head))
            if len(chosen) in (2,6,12,24): add(f"A{len(chosen)}",chosen)
            if len(chosen) == 24: break
    write(out / "plans.json", plans)
    write(out / "run.json", dict(status="running",model_sha256=digest(model_dir / "model.safetensors"),
          split_hashes={name:digest(split_dir / f"{name}.csv") for name in manifests},
          samples=5000,holdout_evaluated=False,backend=f"plaintext PyTorch FP32 {device}; TF32 disabled",
          torch=torch.__version__,device=torch.cuda.get_device_name() if device == "cuda" else "CPU",
          qkv_sample=str(config["image"]),qkv_shape=[1,12,197,64],qkv_sha256=digest(out / "attention.npz"),
          proxy= {k:v.tolist() for k,v in proxy.items()},accuracy_budget_pp=1.0,
          plan_selection="Calibration proxy proposes; measured secure cost and selection accuracy choose. No training."))
    predictions = {name:[] for name in plans}
    labels = []
    logits_smoke = {}
    for batch,(pixels,target) in enumerate(loader("selection")):
        labels.extend(target.tolist())
        pixels = pixels.to(device, non_blocking=True)
        with torch.inference_mode():
            for name,plan in plans.items():
                state["plan"] = plan
                logits = model(pixels).logits
                predictions[name].append(logits.topk(5,dim=-1).indices.cpu().numpy())
                if batch == 0: logits_smoke[name] = logits[:3].cpu().numpy()
        if (batch + 1) % 40 == 0: print(f"Accuracy: {min((batch+1)*32,5000)}/5000",flush=True)
    labels = np.asarray(labels)
    predictions = {k:np.concatenate(v) for k,v in predictions.items()}
    baseline = predictions["S"][:,0] == labels
    rows = []
    for name,plan in plans.items():
        correct = predictions[name][:,0] == labels
        top1 = float(correct.mean() * 100)
        counts = {kind:sum(row.count(kind) for row in plan["heads"]) for kind in "SRL"}
        rows.append(dict(plan=name,S=counts["S"],R=counts["R"],L=counts["L"],samples=len(labels),
             top1=top1,top5=float((predictions[name]==labels[:,None]).any(axis=1).mean()*100),
             loss_pp=float(baseline.mean()*100-top1),
             lost=int((baseline & ~correct).sum()),gained=int((~baseline & correct).sum())))
    with (out / "accuracy.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out / "predictions.npz",labels=labels,**predictions)
    (out / "log").mkdir()
    np.savez_compressed(out / "log/smoke.npz",**logits_smoke)
    meta=json.loads((out / "run.json").read_text());meta["status"]="complete"
    write(out / "run.json",meta)
    for module,forward in zip(modules,original): module.forward=forward
    print("Plaintext accuracy complete: accuracy.csv",flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out.resolve())
