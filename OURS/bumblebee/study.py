"""CPU evidence: secure costs, actual head ablations, and accuracy tradeoffs."""
import csv
import json
import random
import statistics
from pathlib import Path
import numpy as np
from runtime import ROOT, digest, write
from experiment import case_config, run_case, save_rows

SPLITS=ROOT.parent / "blb_base/outputs/heterogeneous_attention_20261009/splits"


class Images:
    def __init__(self,rows,processor):self.rows=rows;self.processor=processor
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        from PIL import Image
        name,label=self.rows[index]
        with Image.open(ROOT.parents[1] / "datasets/imagenet2012/val" / name) as image:
            x=self.processor(image.convert("RGB"),return_tensors="pt")["pixel_values"][0]
        return x,int(label)


class Plain:
    """Frozen FP32 ViT; cached prefixes reuse actual unchanged computation."""
    def __init__(self,data):
        import torch
        import types
        from transformers import AutoImageProcessor,ViTForImageClassification
        self.torch=torch;torch.set_num_threads(32)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        config=json.loads((ROOT / "config.json").read_text());model_dir=(ROOT / config["model"]).resolve()
        self.processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True)
        self.model,loaded=ViTForImageClassification.from_pretrained(model_dir,local_files_only=True,attn_implementation="eager",output_loading_info=True)
        assert not any(loaded.values()),f"Checkpoint mismatch: {loaded}"
        self.model.eval()
        c=self.model.config
        assert (c.num_hidden_layers,c.hidden_size,c.num_attention_heads,c.image_size,c.patch_size)==(12,768,12,224,16)
        self.alpha=json.loads((data / "alpha.json").read_text())["alpha"]
        self.a=torch.tensor(self.alpha,dtype=torch.float32)
        self.plans=[None];self.capture=False;self.cached={}
        for layer,block in enumerate(self.model.vit.encoder.layer):
            def forward(module,hidden_states,head_mask=None,output_attentions=False,layer=layer):
                assert head_mask is None and not output_attentions
                q,k,v=[module.transpose_for_scores(project(hidden_states)) for project in (module.query,module.key,module.value)]
                ctx=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.)
                batch=hidden_states.shape[0]//len(self.plans)
                for index,plan in enumerate(self.plans):
                    if plan is None:continue
                    part=slice(index*batch,(index+1)*batch)
                    for kind in "RL":
                        ids=[h for h,t in enumerate(plan["heads"][layer]) if t==kind]
                        if not ids:continue
                        z=(q[part][:,ids]/8) @ k[part][:,ids].transpose(-1,-2)
                        if kind=="R":
                            w=z.clamp_min(0);den=w.sum(-1,keepdim=True);w=w/torch.where(den>0,den,1)
                        else:w=z*self.a[layer,ids][None,:,None,None]
                        ctx[part][:,ids]=w @ v[part][:,ids]
                if self.capture:self.cached[layer]=(q,k,v,ctx)
                return (ctx.permute(0,2,1,3).reshape(hidden_states.shape),)
            block.attention.attention.forward=types.MethodType(forward,block.attention.attention)

    def loader(self,split,batch=4,limit=None):
        from torch.utils.data import DataLoader
        rows=list(csv.reader((SPLITS / f"{split}.csv").open()))
        if limit is not None:rows=rows[:limit]
        return DataLoader(Images(rows,self.processor),batch_size=batch,shuffle=False,num_workers=4)

    def logits(self,pixels,plans):
        self.plans=plans
        x=pixels.repeat(len(plans),1,1,1)
        return self.model(x).logits.reshape(len(plans),len(pixels),1000)

    def suffix(self,hidden,plans):
        begin=min(l for plan in plans for l,row in enumerate(plan["heads"]) if set(row)!={"S"})
        self.plans=plans;x=hidden[begin].repeat(len(plans),1,1)
        for block in self.model.vit.encoder.layer[begin:]:x=block(x)[0]
        logits=self.model.classifier(self.model.vit.layernorm(x)[:,0])
        return logits.reshape(len(plans),len(hidden[0]),1000)

    def single_heads(self,pixels,layer,hidden):
        t=self.torch;q,k,v,base=self.cached[layer]
        z=(q/8) @ k.transpose(-1,-2)
        r=z.clamp_min(0);den=r.sum(-1,keepdim=True)
        alternatives=[(r/t.where(den>0,den,1))@v,(z*self.a[layer][None,:,None,None])@v]
        self.plans=[None];block=self.model.vit.encoder.layer[layer]
        flat=base.permute(0,2,1,3).reshape(hidden[layer].shape)
        results=[]
        for start in range(0,24,4):
            contexts=flat.repeat(4,1,1);prefix=hidden[layer].repeat(4,1,1)
            for j,index in enumerate(range(start,start+4)):
                kind,head=divmod(index,12)
                contexts[j*len(pixels):(j+1)*len(pixels),:,head*64:(head+1)*64]=alternatives[kind][:,head]
            x=prefix+block.attention.output(contexts,prefix)
            x=block.output(block.intermediate(block.layernorm_after(x)),x)
            for following in self.model.vit.encoder.layer[layer+1:]:x=following(x)[0]
            logits=self.model.classifier(self.model.vit.layernorm(x)[:,0])
            results.append(logits.reshape(4,len(pixels),1000))
        return t.cat(results)


def score(name,pred,labels,baseline,plan,split,method="selected",family="legacy",seed=""):
    correct=pred[:,0]==labels;lost=int((baseline & ~correct).sum());gained=int((~baseline & correct).sum())
    rng=np.random.default_rng(20261012)
    counts=rng.multinomial(len(labels),[lost/len(labels),gained/len(labels),1-(lost+gained)/len(labels)],size=10000)
    ci=np.percentile((counts[:,0]-counts[:,1])*100/len(labels),[2.5,97.5])
    return dict(plan=name,split=split,family=family,method=method,seed=seed,
        S=sum(r.count("S") for r in plan["heads"]),R=sum(r.count("R") for r in plan["heads"]),L=sum(r.count("L") for r in plan["heads"]),
        samples=len(labels),top1=int(correct.sum())*100/len(labels),top5=int((pred==labels[:,None]).any(1).sum())*100/len(labels),
        loss_pp=(lost-gained)*100/len(labels),lost=lost,gained=gained,ci_low_pp=float(ci[0]),ci_high_pp=float(ci[1]))


def head_sensitivity(out,data,layers=None):
    import torch
    if (out / "heads.csv").exists():return
    layers=list(range(12)) if layers is None else layers
    plain=Plain(data);predictions=[];labels=[];baselines=[];check_errors=[]
    existing=out / "log/heads.npz"
    if existing.exists():
        saved=np.load(existing);done=len(saved["labels"]);labels=saved["labels"].tolist()
        predictions=[saved["predictions"]];baselines=[saved["baseline"]]
    else:done=0
    old_error=json.loads((out / "run.json").read_text()).get("cache_full_forward_max_abs",0.)
    seen=0
    with torch.inference_mode():
        for pixels,target in plain.loader("calibration"):
            seen+=len(pixels)
            if seen<=done:continue
            plain.plans=[None];plain.capture=True
            result=plain.model(pixels,output_hidden_states=True)
            plain.capture=False;baseline=result.logits.topk(5,-1).indices.numpy()
            all_predictions=[]
            for layer in layers:
                logits=plain.single_heads(pixels,layer,result.hidden_states)
                if seen==len(pixels):
                    # Every head and both replacements are checked against full forward inference.
                    for start in range(0,24,4):
                        plans=[]
                        for index in range(start,start+4):
                            kind,head=divmod(index,12);heads=[["S"]*12 for _ in range(12)];heads[layer][head]="RL"[kind]
                            plans.append(dict(heads=heads,alpha=plain.alpha))
                        actual=plain.logits(pixels,plans)
                        error=float((actual-logits[start:start+4]).abs().max());check_errors.append(error)
                        assert error<1e-4 and torch.equal(actual.argmax(-1),logits[start:start+4].argmax(-1))
                    plain.plans=[None]
                    status(out,"heads-check",checked_layers=layers.index(layer)+1,cache_full_forward_max_abs=max(check_errors))
                all_predictions.append(logits.topk(5,-1).indices.numpy().astype(np.uint16))
            predictions.append(np.stack(all_predictions,axis=1));baselines.append(baseline.astype(np.uint16));labels.extend(target.tolist())
            if seen%16==0:status(out,"heads",head_images=seen)
            if seen%80==0 or seen==2000:
                print(f"Actual single-head accuracy: {seen}/2000 images, all 288 replacements",flush=True)
                np.savez_compressed(existing,labels=np.asarray(labels),baseline=np.concatenate(baselines),predictions=np.concatenate(predictions,axis=2))
                status(out,"heads",head_images=seen,cache_full_forward_max_abs=max(check_errors,default=old_error))
    pred=np.concatenate(predictions,axis=2);base=np.concatenate(baselines)[:,0]==np.asarray(labels)
    assert pred.shape==(24,len(layers),2000,5)
    rows=[]
    for index_layer,layer in enumerate(layers):
        for index in range(24):
            kind,head=divmod(index,12);p=pred[index,index_layer];correct=p[:,0]==np.asarray(labels)
            lost=int((base & ~correct).sum());gained=int((~base & correct).sum())
            rows.append(dict(layer=layer,head=head,kind="RL"[kind],samples=2000,top1=int(correct.sum())*100/2000,
                loss_pp=(lost-gained)*100/2000,lost=lost,gained=gained))
    save_rows(out / "heads.csv",rows)
    status(out,"heads-complete",heads_sha256=digest(out / "heads.csv"),calibration_baseline_top1=int(base.sum())*100/2000)


def make_plans(out,data):
    path=out / "plans.json"
    if path.exists():return json.loads(path.read_text())
    alpha=json.loads((data / "alpha.json").read_text())["alpha"]
    sensitivity=list(csv.DictReader((out / "heads.csv").open()))
    plans=json.loads((data / "plans.json").read_text())
    info={name:dict(family="legacy",method="legacy",seed="") for name in plans}
    def add(name,changes,family,method,seed=""):
        heads=[["S"]*12 for _ in range(12)]
        for l,h,t in changes:assert heads[l][h]=="S";heads[l][h]=t
        plans[name]=dict(heads=heads,alpha=alpha);info[name]=dict(family=family,method=method,seed=seed)
    for family in "RLM":
        for count in (6,12,24,48):
            quotas={"R":count if family=="R" else count//2 if family=="M" else 0,
                    "L":count if family=="L" else count//2 if family=="M" else 0}
            chosen=[];seen=set();remaining=quotas.copy()
            for r in sorted(sensitivity,key=lambda r:(float(r["loss_pp"]),int(r["layer"]),int(r["head"]),r["kind"])):
                l,h,t=int(r["layer"]),int(r["head"]),r["kind"]
                if remaining[t] and (l,h) not in seen:chosen.append((l,h,t));seen.add((l,h));remaining[t]-=1
                if not any(remaining.values()):break
            assert len(chosen)==count
            name=f"{family.lower()}{count}"
            add(name,chosen,family,"selected")
            for seed in range(5):
                rng=random.Random(20261012+seed);changes=[]
                for l,row in enumerate(plans[name]["heads"]):
                    ids=rng.sample(range(12),12);nr,nl=row.count("R"),row.count("L")
                    changes += [(l,h,"R") for h in ids[:nr]]+[(l,h,"L") for h in ids[nr:nr+nl]]
                add(f"{name}-{seed}",changes,family,"random",seed)
            if family=="M":
                changes=[(l,h,t) for j in range(6) for l in range(12) for h,t in ((j,"R"),(j+6,"L"))][:count]
            else:
                order=list(range(12)) if family=="R" else list(range(6,12))+list(range(6))
                changes=[(l,h,family) for h in order for l in range(12)][:count]
            add(name+"-u",changes,family,"uniform")
    write(path,plans);write(out / "log/info.json",info)
    return plans


def evaluate(out,data,split,names):
    import torch
    plans=json.loads((out / "plans.json").read_text());info=json.loads((out / "log/info.json").read_text())
    path=out / "log" / f"{split}.npz";plain=Plain(data)
    predictions={n:[] for n in names};labels=[];done=0;seen=0;errors=[]
    if path.exists():
        saved=np.load(path);done=len(saved["labels"]);labels=saved["labels"].tolist()
        assert set(saved.files)==set(names)|{"labels"}
        predictions={n:[saved[n]] for n in names}
    legacy=json.loads((data / "plans.json").read_text()) if split=="selection" else {}
    old={n:v for n,v in np.load(data / "predictions.npz").items()} if split=="selection" else None
    reuse={n:next((x for x,p in legacy.items() if p==plans[n]),None) for n in names if n!="S"}
    needed=[n for n in names if n!="S" and not reuse.get(n)]
    loader=plain.loader(split);total=len(loader.dataset)
    with torch.inference_mode():
        for pixels,target in loader:
            seen+=len(pixels)
            if seen<=done:continue
            plain.plans=[None];result=plain.model(pixels,output_hidden_states=True)
            base=result.logits.topk(5,-1).indices.numpy()
            predictions["S"].append(base.astype(np.uint16))
            if old is not None:
                assert np.array_equal(base[:,0],old["S"][seen-len(pixels):seen,0]),"CPU baseline differs from old accuracy; do not reuse"
                for n,x in reuse.items():
                    if x is not None:predictions[n].append(old[x][seen-len(pixels):seen].astype(np.uint16))
            for start in range(0,len(needed),4):
                chunk=needed[start:start+4];variants=[plans[n] for n in chunk]
                logits=plain.suffix(result.hidden_states,variants)
                if seen==len(pixels):
                    actual=plain.logits(pixels,variants);error=float((actual-logits).abs().max());errors.append(error)
                    assert error<1e-4 and torch.equal(actual.argmax(-1),logits.argmax(-1))
                pred=logits.topk(5,-1).indices.numpy().astype(np.uint16)
                for j,n in enumerate(chunk):predictions[n].append(pred[j])
            labels.extend(target.tolist())
            if seen%16==0:status(out,split,accuracy_images=seen,accuracy_total=total)
            if seen%80==0 or seen==total:
                arrays={n:np.concatenate(v) for n,v in predictions.items()}
                np.savez_compressed(path,labels=np.asarray(labels),**arrays)
                print(f"Actual {split} accuracy: {seen} images, {len(names)} plans",flush=True)
    arrays={n:np.concatenate(v) for n,v in predictions.items()};labels=np.asarray(labels);baseline=arrays["S"][:,0]==labels
    rows=[score(n,arrays[n],labels,baseline,plans[n],split,**info[n]) for n in names]
    for row in rows:row["source"]=f"out/07:{reuse[row['plan']]}" if reuse.get(row['plan']) else "CPU FP32 actual inference"
    previous=list(csv.DictReader((out / "acc.csv").open())) if (out / "acc.csv").exists() else []
    previous=[r for r in previous if r["split"]!=split]
    save_rows(out / "acc.csv",previous+rows)
    status(out,split+"-complete",accuracy_images=len(labels),accuracy_sha256=digest(out / "acc.csv"),
           **{split+"_cache_max_abs":max(errors,default=0.)})


def combinations(out,data):
    plans=make_plans(out,data)
    rows=list(csv.DictReader((out / "acc.csv").open())) if (out / "acc.csv").exists() else []
    if not any(r["split"]=="selection" for r in rows):parallel_accuracy(out,data,"selection",list(plans))
    rows=[r for r in csv.DictReader((out / "acc.csv").open()) if r["split"]=="selection"]
    freeze=out / "log/frozen.json"
    if not freeze.exists():
        names=["S","R12","A24","r48","l6","l48","m6","m48"]
        for family in "RLM":
            feasible=[r for r in rows if r["family"]==family and r["method"]=="selected" and float(r["loss_pp"])<=1]
            if feasible:
                best=max(feasible,key=lambda r:int(r["R"])+int(r["L"]))["plan"]
                if best not in names:names.append(best)
        write(freeze,dict(holdout=names,timing=["S","r48","l6","l48","m6","m48"],
             plans_sha256=digest(out / "plans.json"),selection_sha256=digest(out / "log/selection.npz"),
             rule="Fixed small/large timing representatives; maximum feasible selected count per family added for holdout only; no holdout-based tuning"))


def accuracy_part(out,data,stage,part,names):
    import os,contextlib,traceback
    os.sched_setaffinity(0,set(range(part*48,(part+1)*48)))
    with (out / "log/worker.log").open("a") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            if stage=="heads":head_sensitivity(out,data,list(range(part,12,2)))
            else:evaluate(out,data,stage,names)
            status(out,stage+"-complete",status="complete")
        except BaseException as exc:
            status(out,"failed",status="failed",error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc();raise SystemExit(1)


def parallel_accuracy(out,data,stage,names=None):
    import multiprocessing as mp
    import shutil,time
    parts=[out / "log" / f"{stage}{i}" for i in range(2)];processes=[]
    context=mp.get_context("spawn")
    for i,part in enumerate(parts):
        part.mkdir(exist_ok=True);(part / "log").mkdir(exist_ok=True)
        meta=json.loads((part / "run.json").read_text()) if (part / "run.json").exists() else {}
        meta.update(status="running",stage=stage);meta.pop("error",None);write(part / "run.json",meta)
        if names is not None:
            shutil.copyfile(out / "plans.json",part / "plans.json");shutil.copyfile(out / "log/info.json",part / "log/info.json")
        assigned=None if names is None else ["S"]+[n for j,n in enumerate(x for x in names if x!="S") if j%2==i]
        process=context.Process(target=accuracy_part,args=(part,data,stage,i,assigned));process.start();processes.append(process)
    try:
        while any(p.is_alive() for p in processes):
            progress=[json.loads((p / "run.json").read_text()) for p in parts]
            status(out,stage,parts=progress)
            for i,p in enumerate(processes):
                if p.exitcode not in (None,0):raise RuntimeError(f"Accuracy worker {i} failed; see {parts[i] / 'log/worker.log'}")
            time.sleep(5)
        for i,p in enumerate(processes):
            p.join()
            if p.exitcode!=0:raise RuntimeError(f"Accuracy worker {i} failed; see {parts[i] / 'log/worker.log'}")
    finally:
        for p in processes:
            if p.is_alive():p.terminate();p.join()
    if stage=="heads":
        saved=[np.load(p / "log/heads.npz") for p in parts]
        assert np.array_equal(saved[0]["labels"],saved[1]["labels"]) and np.array_equal(saved[0]["baseline"],saved[1]["baseline"])
        merged=np.empty((24,12,2000,5),dtype=np.uint16)
        for i,s in enumerate(saved):merged[:,i::2]=s["predictions"]
        np.savez_compressed(out / "log/heads.npz",labels=saved[0]["labels"],baseline=saved[0]["baseline"],predictions=merged)
        rows=[r for p in parts for r in csv.DictReader((p / "heads.csv").open())]
        rows.sort(key=lambda r:(int(r["layer"]),int(r["head"]),r["kind"]))
        save_rows(out / "heads.csv",rows)
        status(out,"heads-complete",heads_sha256=digest(out / "heads.csv"),
               calibration_baseline_top1=(saved[0]["baseline"][:,0]==saved[0]["labels"]).mean()*100)
    else:
        saved=[np.load(p / "log" / f"{stage}.npz") for p in parts]
        assert np.array_equal(saved[0]["labels"],saved[1]["labels"]) and np.array_equal(saved[0]["S"],saved[1]["S"])
        merged={key:s[key] for s in saved for key in s.files}
        np.savez_compressed(out / "log" / f"{stage}.npz",**merged)
        rows=[r for p in parts for r in csv.DictReader((p / "acc.csv").open())]
        rows=[r for i,r in enumerate(rows) if r["plan"]!="S" or i==0]
        previous=list(csv.DictReader((out / "acc.csv").open())) if (out / "acc.csv").exists() else []
        save_rows(out / "acc.csv",[r for r in previous if r["split"]!=stage]+rows)
        status(out,stage+"-complete",accuracy_sha256=digest(out / "acc.csv"),accuracy_images=len(saved[0]["labels"]))


def full_timings(out,native):
    from experiment import full
    frozen=json.loads((out / "log/frozen.json").read_text())
    assert frozen["plans_sha256"]==digest(out / "plans.json")
    plans=json.loads((out / "plans.json").read_text());target=out / "log/full";target.mkdir(exist_ok=True)
    data=target / "data";data.mkdir(exist_ok=True)
    write(data / "plans.json",plans)
    accuracy=[r for r in csv.DictReader((out / "acc.csv").open()) if r["split"]=="selection"]
    save_rows(data / "accuracy.csv",accuracy)
    for name in frozen["timing"]:
        status(out,"full",timing_plan=name)
        full(target,name,"cpu",native,repeats=3,data=data)
    old=ROOT / "out/09"
    previous=[r for r in csv.DictReader((old / "results.csv").open()) if r["layout"]=="joint" and r["case"]!="cpu-S-joint"]
    current=list(csv.DictReader((target / "results.csv").open()))
    rows=[]
    for r in previous+current:
        name=r["case"].split('-')[1]
        base=old if r in previous else target
        raw=json.loads((base / r["metadata"]).read_text())
        assert r["native_sha256"]==digest(native) and raw["config"]["backend"]=="cpu"
        assert raw["config"].get("head_plan")== (None if name=="S" else plans[name])
        times=[m["online_s"] for m in raw["measurements"]]
        rows.append(dict(plan=name,repeats=int(r["repeats"]),seconds=float(r["online_s"]),gib=float(r["total_gib"]),
                         min_seconds=min(times),max_seconds=max(times),
                         numerical_pass=r["numerical_pass"],source="historical" if r in previous else "current",
                         metadata=str((base/r["metadata"]).relative_to(ROOT))))
    save_rows(out / "time.csv",rows)
    status(out,"full-complete",time_sha256=digest(out / "time.csv"))


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42})
    rows=list(csv.DictReader((out / "cost.csv").open()))
    raw_cost=list(csv.DictReader((out / "log/cost.csv").open()))
    fig,axes=plt.subplots(2,2,figsize=(8,5),layout="constrained")
    for ax,(scope,h) in zip(axes.flat,[(s,h) for s in ("map","attention") for h in (1,12)]):
        group=[r for r in rows if r["scope"]==scope and int(r["heads"])==h]
        quartiles=np.array([np.percentile([float(r["online_s"]) for r in raw_cost if r["scope"]==scope and int(r["heads"])==h and r["kind"]==s["kind"]],[25,75]) for s in group]).T
        medians=np.array([float(r["seconds"]) for r in group])
        ax.bar([r["kind"] for r in group],medians,yerr=np.array([medians-quartiles[0],quartiles[1]-medians]),capsize=3,color=["#4c78a8","#f58518","#54a24b"])
        ax.set_title(f"{'Weight map' if scope=='map' else 'Attention core'}: {h} head{'s' if h>1 else ''}");ax.set_ylabel("Secure execution (s)")
        for i,r in enumerate(group):ax.text(i,quartiles[1,i],f"{float(r['seconds']):.3f}",ha="center",va="bottom")
        ax.margins(y=.2)
    fig.suptitle("Median of 5 independent 2PC runs; bars show interquartile range",fontsize=10)
    fig.savefig(out / "cost.pdf");plt.close(fig)
    head_rows=list(csv.DictReader((out / "heads.csv").open()));matrices=[]
    for kind in "RL":
        matrix=np.zeros((12,12))
        for r in head_rows:
            if r["kind"]==kind:matrix[int(r["layer"]),int(r["head"])]=float(r["loss_pp"])
        matrices.append(matrix)
    fig,axes=plt.subplots(1,2,figsize=(8,4),layout="constrained")
    for ax,kind,matrix in zip(axes,"RL",matrices):
        img=ax.imshow(matrix,cmap="RdBu_r",vmin=-1,vmax=1);ax.set_title(f"Single-head {kind}: calibration 2,000")
        for layer,head in zip(*np.where(matrix>1)):
            ax.text(head,layer,f"{matrix[layer,head]:.2f}",ha="center",va="center",color="white",fontsize=7)
        ax.set_xlabel("Head");ax.set_ylabel("Layer");ax.set_xticks(range(12));ax.set_yticks(range(12))
    fig.colorbar(img,ax=axes,label="Top-1 loss (pp); values above 1 are annotated",extend="max");fig.savefig(out / "heads.pdf");plt.close(fig)
    rows=[r for r in csv.DictReader((out / "acc.csv").open()) if r["split"]=="selection"]
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout="constrained")
    for ax,family in zip(axes,"RLM"):
        for method,color in [("selected","#4c78a8"),("uniform","#f58518"),("random","#888888")]:
            group=[r for r in rows if r["family"]==family and r["method"]==method]
            xs=sorted(set(int(r["R"])+int(r["L"]) for r in group));ys=[];dev=[]
            for x in xs:
                vals=[float(r["loss_pp"]) for r in group if int(r["R"])+int(r["L"])==x]
                ys.append(np.mean(vals));dev.append(np.std(vals,ddof=1) if len(vals)>1 else 0)
            ax.plot([0]+xs,[0]+ys,"o-",label=method,color=color)
            if method=="random":ax.fill_between(xs,np.array(ys)-dev,np.array(ys)+dev,color=color,alpha=.2)
        ax.axhline(1,color="black",ls="--",lw=1,label="1 pp reference")
        ax.set_title({"R":"ReLU only","L":"Linear only","M":"ReLU + Linear"}[family]);ax.set_xlabel("Replaced heads");ax.set_ylabel("Top-1 loss (pp)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Selection 5,000 images; random band: mean +/- SD over 5 seeds",fontsize=10)
    fig.savefig(out / "acc.pdf");plt.close(fig)
    if (out / "time.csv").exists():
        from matplotlib.lines import Line2D
        measured=list(csv.DictReader((out / "time.csv").open()));test={r["plan"]:r for r in csv.DictReader((out / "acc.csv").open()) if r["split"]=="holdout"}
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        for r in measured:
            x,y=float(r["seconds"]),float(test[r["plan"]]["loss_pp"])
            raw=json.loads((ROOT / r["metadata"]).read_text());times=[m["online_s"] for m in raw["measurements"]]
            accuracy=test[r["plan"]];prior=r["source"]=="historical"
            color="#888888" if prior else {"S":"#4c78a8","R":"#f58518","L":"#54a24b","M":"#b279a2"}[r["plan"][0].upper()]
            for ax in axes:
                ax.errorbar(x,y,xerr=[[x-min(times)],[max(times)-x]],
                            yerr=[[y-float(accuracy["ci_low_pp"])],[float(accuracy["ci_high_pp"])-y]],
                            fmt="o",color=color,mfc="white" if prior else color,capsize=2)
                if ax is axes[1] or y>1.6 or r["plan"]=="S":
                    ax.annotate(r["plan"].upper()+(" (prior)" if prior else ""),(x,y),xytext=(3,4),textcoords="offset points",fontsize=8)
        for ax in axes:
            ax.axhline(1,color="black",ls="--",lw=1);ax.set_xlabel("Measured complete secure inference (s)");ax.set_ylabel("Holdout Top-1 loss (pp)")
        axes[0].set_title("All measured representatives");axes[1].set_title("Near the 1 pp reference");axes[1].set_ylim(-.15,1.6)
        near=[r for r in measured if float(test[r["plan"]]["loss_pp"])<=1.6]
        axes[1].set_xlim(min(float(r.get("min_seconds",r["seconds"])) for r in near)-2,
                         max(float(r.get("max_seconds",r["seconds"])) for r in near)+4)
        axes[0].legend(handles=[Line2D([],[],marker="o",ls="",color="#4c78a8",label="Current timing"),
                               Line2D([],[],marker="o",ls="",color="#888888",mfc="white",label="Prior timing")],fontsize=8)
        fig.suptitle("Horizontal bars: range of 3 runs; vertical bars: paired 95% CI",fontsize=10)
        fig.savefig(out / "time.pdf");plt.close(fig)
    status(out,"complete",status="complete",accuracy_sha256=digest(out / "acc.csv"))


def status(out,stage,**values):
    path=out / "run.json"
    meta=json.loads(path.read_text());meta.update(stage=stage,**values)
    if values.get("status")=="complete":
        for key in ("parts","timing_plan"):meta.pop(key,None)
    write(path,meta)


def secure_costs(out,data,native):
    target=out / "log/cost";target.mkdir(parents=True,exist_ok=True)
    path=out / "log/cost.csv"
    rows=list(csv.DictReader(path.open())) if path.exists() else []
    alpha=json.loads((data / "alpha.json").read_text())["alpha"][0]
    qkv=data / "attention.npz"
    for repeat in range(1,6):
        cases=[(scope,h,kind) for scope in ("map","attention") for h in (1,12) for kind in "SRL"]
        random.Random(20261012+repeat).shuffle(cases)
        for scope,h,kind in cases:
            name=f"c{repeat}-{scope[0]}{h}-{kind}"
            if any(r["case"]==name for r in rows):continue
            config=case_config("attention","cpu",scope=scope,head_count=h,types=[kind]*h,alpha=alpha,
                   qkv="/work/"+str(qkv.relative_to(ROOT.parents[1])),qkv_sha256=digest(qkv),
                   layout="joint",repeats=1,plan_name=kind,native_library=str(native))
            row=run_case(target,name,config)
            row.update(scope=scope,heads=h,kind=kind,round=repeat)
            row["metadata"]=str(Path("log/cost") / row["metadata"])
            rows.append(row);save_rows(path,rows)
            status(out,"cost",cost_runs=len(rows),cost_total=60)
    summary=[]
    for scope in ("map","attention"):
        for h in (1,12):
            for kind in "SRL":
                group=[r for r in rows if r["scope"]==scope and int(r["heads"])==h and r["kind"]==kind]
                assert len(group)==5 and all(str(r["numerical_pass"])=="True" for r in group)
                times=[float(r["online_s"]) for r in group]
                summary.append(dict(scope=scope,heads=h,kind=kind,repeats=5,
                   seconds=statistics.median(times),iqr_seconds=float(np.subtract(*np.percentile(times,[75,25]))),
                   mib=statistics.median(float(r["total_gib"])*1024 for r in group),
                   relative_l2=max(float(r["relative_l2"]) for r in group),
                   max_abs=max(float(r["max_abs"]) for r in group)))
    save_rows(out / "cost.csv",summary)
    status(out,"cost-complete",cost_sha256=digest(out / "cost.csv"))


def run(out,stage,data,native):
    out.mkdir(parents=True,exist_ok=True);(out / "log").mkdir(exist_ok=True)
    if not (out / "run.json").exists():
        config=json.loads((ROOT / "config.json").read_text())
        write(out / "run.json",dict(status="running",stage=stage,backend="cpu",budget_reference_pp=1.,
            candidate_filter="No accuracy filtering in exploration; budget is a reference line",
            model_sha256=digest((ROOT / config["model"]).resolve() / "model.safetensors"),
            native_sha256=digest(native),source_data=str(data),
            splits={name:digest(SPLITS / f"{name}.csv") for name in ("calibration","selection","holdout")},
            holdout_authorization="Current user request explicitly invokes the annotated plan including frozen holdout evaluation",
            protocol="Original CPU BumbleBee; real 2PC FM64/f18; 25 threads per party; 1 Gbps / 0.25 ms per direction"))
    if stage in ("all","cost"):secure_costs(out,data,native)
    if stage in ("all","heads") and not (out / "heads.csv").exists():parallel_accuracy(out,data,"heads")
    if stage in ("all","acc"):combinations(out,data)
    if stage in ("all","holdout"):
        frozen=json.loads((out / "log/frozen.json").read_text())
        assert frozen["plans_sha256"]==digest(out / "plans.json")
        rows=list(csv.DictReader((out / "acc.csv").open()))
        if not any(r["split"]=="holdout" for r in rows):parallel_accuracy(out,data,"holdout",frozen["holdout"])
    if stage in ("all","full"):full_timings(out,native)
    if stage in ("all","report"):report(out)
