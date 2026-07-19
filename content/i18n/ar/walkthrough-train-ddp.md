# `scripts/train_ddp.py` {#scriptstrainddppy}

## الغرض {#purpose}

يوسّع هذا الملف المدرّب ذي العملية الواحدة بتدريب اختياري متعدد وحدات GPU باستخدام التدريب المتوازي الموزّع (DDP). تحت `torchrun` تملك كل عملية GPU واحدة، وتتدرب على شريحة منفصلة من معرّفات المعقّدات، ويحسب DDP متوسط التدرجات، وتجمع اختزالات صريحة المقاييس؛ أما تحت `python3` العادي فيتبع مسار العملية الواحدة.

## موقعه في المنظومة {#how-it-fits-in}

يستدعيه `run_full_stage2_ddp.sh` ومشغّل v2 مباشرة، باستخدام `torchrun` عند ظهور أكثر من GPU واحدة. يقبل تقريبًا وسائط البيانات والنموذج والمحسّن والاستئناف وW&B نفسها التي يقبلها `train.py`، لكن بلا `--jitter`. ويزوّد `torchrun` أيضًا `WORLD_SIZE` و`RANK` و`LOCAL_RANK`. تقرأ الرتب كلها الرسوم وتشارك في التدريب؛ ولا تكتب إلا الرتبة 0 ملفات `train_log.csv` و`last.pt` و`karmadock_team002.pkl` وملف معرّف تشغيل W&B وسجلاته.

## شرح الشيفرة {#walkthrough}

### الاستيرادات ونموذج DDP {#imports-and-ddp-model}

```python
import argparse
import contextlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.fns import set_random_seed, Early_stopper, karmadock_evaluation
from dataset.graph_obj import PDBBindGraphDataset
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.KarmaDock_architecture import KarmaDock

from seminar_csv import complex_ids
```

يشغّل DDP عملية Python ونسخة نموذج مستقلة لكل GPU. أثناء الانتشار الخلفي تجمع النسخ تدرجات المعاملات المناظرة وتحسب متوسطها جماعيًا، فتجري المحسّنات كلها التحديث نفسه. ويتوسع ذلك عادة أفضل من `DataParallel` ذي العملية الواحدة. يوفّر `torch.distributed` عمليات مجموعة العمليات، بينما يوفّر `contextlib` سياقًا خاملًا للشيفرة المشتركة بين نمطي DDP وغير DDP. استُورد `sys` لكنه غير مستخدم في الملف.

### مساعدات التوزيع {#distributed-helpers}

```python
def ddp_info():
    """(enabled, rank, local_rank, world_size) from the torchrun environment."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        return True, int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0")), world
    return False, 0, 0, 1
```

`WORLD_SIZE` هو العدد الكلي للعمليات، و`RANK` هوية العملية العامة، ويختار `LOCAL_RANK` وحدتها GPU على العقدة الحالية. يفعّل عالم متعدد العمليات DDP، وإلا تبسّط الأصفار الحتمية التفرعات اللاحقة.

```python
def shard(ids, rank, world):
    """Stripe a list across ranks so each rank trains on a disjoint subset."""
    return ids[rank::world] if world > 1 else list(ids)
```

تأخذ الشريحة الموسعة في Python كل معرّف ذي ترتيب `world` بدءًا من `rank`: مع رتبتين تأخذ إحداهما الفهارس 0,2,4… والأخرى 1,3,5…. يحل ذلك محل `DistributedSampler` المعتاد في PyTorch، وهو عيّان يقسم مجموعة البيانات ويعيد خلطها باتساق كل حقبة. يتجاوز التخطيط المخصص محمّل KarmaDock الخاص لكنه يثبت عضوية مجموعة كل رتبة طوال التشغيل.

```python
def all_reduce_sum(values, device):
    """Sum a small list of python floats across all ranks; returns a python list."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()
```

الاختزال الشامل عملية جماعية تساهم فيها كل رتبة بموتر، وتُختزل الموترات عنصرًا بعنصر، وتتلقى كل رتبة الناتج. يسمح جمع مجاميع الخسارة والأعداد لكل رتبة بحساب المتوسطات العامة نفسها واتخاذ قرار الإيقاف نفسه. يقلل Float64 الخطأ العددي في تجميع المقاييس، ولا تشارك هذه الموترات في التدرجات.

### واجهة CLI واختيار الرسوم {#cli-and-graph-selection}

```python
def parse_args():
    p = argparse.ArgumentParser(description="Retrain / fine-tune KarmaDock.")
    p.add_argument("--graph_dir", required=True,
                   help="Dir with per-complex <id>.dgl graphs (from generate_graph.py).")
    p.add_argument("--val_graph_dir", default="",
                   help="Graph dir for the validation complexes when --val_csv points to a "
                        "separately-preprocessed split (LP train/val live in different dirs). "
                        "Defaults to --graph_dir.")
    p.add_argument("--complex_dir", default="",
                   help="KarmaDock-layout complex dir (only needed if a graph must be regenerated).")
    p.add_argument("--csv", required=True,
                   help="Seminar train CSV (proto_train.csv / full_train.csv / lp_train.csv).")
    p.add_argument("--val_csv", default="",
                   help="Optional explicit validation CSV. If set, ALL of --csv is used "
                        "for training and these complexes for validation -- used for "
                        "LP-PDBBind, whose leak-proof val split must NOT be a random "
                        "carve-out of train. If empty, fall back to a --val_frac split.")
    p.add_argument("--out_dir", required=True,
                   help="Where to write checkpoints and the training log.")
    p.add_argument("--init_model", default="",
                   help="Checkpoint to initialise from. Empty = train from scratch; "
                        "set to karmadock_screening.pkl to fine-tune.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <out_dir>/last.pt if present (for cluster reschedules).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Physical minibatch size (limited by GPU memory).")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="Gradient-accumulation steps. Effective batch = batch_size*accum_steps. "
                        "Paper uses an effective batch of 64; on a 16 GB V100 use e.g. "
                        "--batch_size 8 --accum_steps 8.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--pos_r", type=float, default=1.0,
                   help="Weight on the pose (RMSD) loss vs the MDN scoring loss.")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stopping patience (epochs without val improvement).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--random_seed", type=int, default=2023)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--wandb", action="store_true", help="log metrics to W&B")
    p.add_argument("--wandb_project", default="karmadock-seminar")
    p.add_argument("--wandb_entity", default=None, help="W&B team/user (optional)")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()
```

تطابق المعاني `train.py`: الدفعة الفيزيائية لكل عملية وGPU، لذلك تكون الدفعة العالمية الفعّالة المكتملة `batch_size × accum_steps × world_size`. يزن `pos_r` حد الإحداثيات/RMSD مقابل خسارة مسافة MDN وتسجيلها. ولا يوجد خيار لتشويش الإحداثيات في نسخة DDP هذه.

```python
def split_available(ids, graph_dir, val_frac, seed):
    """Keep only ids that actually have a graph, then deterministically split."""
    have = [i for i in ids if os.path.exists(os.path.join(graph_dir, f"{i}.dgl"))]
    missing = len(ids) - len(have)
    if missing:
        print(f"# WARNING: {missing}/{len(ids)} complexes have no .dgl graph "
              f"(rdkit/pocket failures) and are skipped.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(have))
    n_val = max(1, int(round(len(have) * val_frac)))
    val_idx = set(perm[:n_val].tolist())
    train_ids = [have[i] for i in range(len(have)) if i not in val_idx]
    valid_ids = [have[i] for i in range(len(have)) if i in val_idx]
    return train_ids, valid_ids


def available(ids, graph_dir):
    """Keep only ids that actually have a .dgl graph (used with an explicit val CSV)."""
    have = [i for i in ids if os.path.exists(os.path.join(graph_dir, f"{i}.dgl"))]
    missing = len(ids) - len(have)
    if missing:
        print(f"# WARNING: {missing}/{len(ids)} complexes have no .dgl graph and are skipped.")
    return have
```

تطابق الدالتان مساعدات العملية الواحدة: تُتجاوز الرسوم الفاشلة، ويكون التقسيم العشوائي ثابتًا بالبذرة، ولا يُخلط التحقق الصريح بالتدريب. يحدث ذلك قبل التقسيم على الرتب، فتشتق كل رتبة بصورة مستقلة قوائم المعرّفات العامة نفسها.

### تحقق ملائم لـDDP وسياق المزامنة {#ddp-friendly-validation-and-synchronization-context}

```python
def validate_sums(model, loader, device, pos_r):
    """Validation loss SUMS (+count) over a loader -- DDP-friendly (all-reduce the sums).

    Mirrors the loss used in training: loss = pos_r*rmsd_loss + mdn_loss.
    """
    model.eval()
    s_tot = s_rmsd = s_mdn = 0.0
    n = 0
    with torch.no_grad():
        for data in loader:
            if data is None:
                continue
            data = data.to(device)
            rmsd_loss, mdn_loss = model(data, device, pos_r)
            if mdn_loss is None:
                continue
            loss = pos_r * rmsd_loss + mdn_loss
            s_tot += float(loss); s_rmsd += float(rmsd_loss); s_mdn += float(mdn_loss); n += 1
    return s_tot, s_rmsd, s_mdn, n
```

تحوّل `eval()` سلوك الطبقات إلى الاستدلال، وتمنع `no_grad()` بناء رسم autograd فتقلل الذاكرة والحساب. إرجاع المجاميع مع العدد أساسي؛ لأن حساب متوسط كل شريحة أولًا ثم متوسط الشرائح سيحيّز النتيجة عندما تختلف أحجام الشرائح.

```python
def maybe_no_sync(model, active):
    """no_sync() context to skip DDP gradient all-reduce on non-boundary accum steps."""
    if active and hasattr(model, "no_sync"):
        return model.no_sync()
    return contextlib.nullcontext()
```

يزامن DDP التدرجات عادة بعد كل انتشار خلفي. يُقصد بـ`no_sync()` منع الاتصال في خطوات التراكم الوسيطة والمزامنة عند حد تحديث المحسّن فقط. تتيح `nullcontext()` للمستدعي استخدام عبارة `with` واحدة عندما لا يكون المنع نشطًا.

```python
def build_loader(ids, args, dataset_type, shuffle, graph_dir=None):
    ds = PDBBindGraphDataset(
        src_dir=args.complex_dir,
        pdb_ids=ids,
        dst_dir=graph_dir or args.graph_dir,
        dataset_type=dataset_type,
        n_job=1,
        on_the_fly=True,
        verbose=False,
    )
    return PassNoneDataLoader(
        dataset=ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, follow_batch=[], pin_memory=True,
    )
```

يطابق ذلك `train.py`: تدخل معرّفات كل رتبة المقسمة مسبقًا محمّلًا مخصصًا مستقلًا، ويعيد التحميل عند الطلب عشوأة وضعيات البداية. ولا يُستخدم `DistributedSampler` قياسي.

### بدء مجموعة العمليات وW&B {#process-group-startup-and-wb}

```python
def main():
    args = parse_args()

    is_ddp, rank, local_rank, world = ddp_info()
    main_proc = (rank == 0)
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if main_proc:
        os.makedirs(args.out_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()
    set_random_seed(args.random_seed)
    if main_proc:
        print(f"# device: {device} | DDP: {is_ddp} (world_size={world})")
```

NCCL هي خلفية NVIDIA للاتصال الجماعي بين وحدات GPU. تختار كل عملية GPU المحلية قبل بناء موترات CUDA. ويمنع **الحاجز** كل رتبة حتى تصل الرتب كلها، ضامنًا إنشاء الرتبة 0 دليل الخرج. تتلقى الرتب البذرة الأساسية نفسها؛ وبما أن معرّفاتها مختلفة فهي تعالج عينات مختلفة حتى إذا بدأت تدفقات الخلط والعشوائية بصورة متشابهة.

```python
    wb = None
    if args.wandb and main_proc:
        try:
            import wandb
            has_creds = os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc"))
            if not has_creds and os.environ.get("WANDB_MODE") != "offline":
                print("# WARNING: no W&B credentials (WANDB_API_KEY / ~/.netrc) -> forcing offline")
                os.environ["WANDB_MODE"] = "offline"
            eff_batch = args.batch_size * args.accum_steps * world
            stage = "scoring" if args.pos_r == 0 else "docking"
            base = args.wandb_run_name or "kd"
            rid_file = os.path.join(args.out_dir, "wandb_run_id.txt")
            if args.resume and os.path.exists(os.path.join(args.out_dir, "last.pt")) and os.path.exists(rid_file):
                run_id = open(rid_file).read().strip()
            else:
                run_id = f"{base}_{int(time.time())}_{os.getpid()}"
                os.makedirs(args.out_dir, exist_ok=True)
                with open(rid_file, "w") as _rf:
                    _rf.write(run_id)
            run_name = (f"{base}_lr{args.lr:g}_bs{eff_batch}_ep{args.epochs}"
                        f"_pat{args.patience}_seed{args.random_seed}_{run_id.rsplit('_', 2)[-1]}")
            cfg = dict(vars(args))
            cfg.update(effective_batch=eff_batch, stage=stage, world_size=world,
                       init_from=("scratch" if not args.init_model else os.path.basename(args.init_model)))
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            id=run_id, name=run_name, config=cfg,
                            dir=args.out_dir, resume="allow")
            print(f"# W&B run: {run_name}")
        except Exception as e:
            print(f"# WARNING: W&B disabled ({e})")
            wb = None
```

تملك الرتبة 0 وحدها تتبع التجربة، فتمنع تكرار التشغيلات. يسجل الإعداد حجم العالم والدفعة العالمية الفعّالة. وبالمقارنة مع `train.py` تكون اللاحقة المعروضة مكوّن PID في المعرّف لا طابع بداية زمنيًا منسقًا؛ وتظل الهوية محفوظة ولا يعاد استخدامها إلا مع `--resume` ووجود ملفات الاستئناف معًا.

### قوائم البيانات العامة وشرائح الرتب وتغليف النموذج {#global-data-lists-rank-shards-and-model-wrapping}

```python
    val_graph_dir = args.val_graph_dir or args.graph_dir
    if args.val_csv:
        train_ids = available(complex_ids(args.csv), args.graph_dir)
        valid_ids = available(complex_ids(args.val_csv), val_graph_dir)
    else:
        train_ids, valid_ids = split_available(
            complex_ids(args.csv), args.graph_dir, args.val_frac, args.random_seed)
    if main_proc:
        print(f"# train: {len(train_ids)} | valid: {len(valid_ids)} complexes (global)")
    train_ids = shard(train_ids, rank, world)
    valid_ids = shard(valid_ids, rank, world)
    train_loader = build_loader(train_ids, args, "train", shuffle=True)
    valid_loader = build_loader(valid_ids, args, "valid", shuffle=False, graph_dir=val_graph_dir)
```

تحسب كل عملية أولًا التقسيم العام المطابق، ثم تختار شريحتها. يضمن ذلك تغطية منفصلة من دون تنسيق بين الرتب. وقد تختلف الشرائح بمعقّد واحد على الأكثر، مع إمكان اختلاف عدد الدفعات الصالحة أكثر إذا تحولت عينات إلى `None`.

```python
    model = KarmaDock()
    if is_ddp:
        if args.init_model:
            if main_proc:
                print(f"# init weights from {args.init_model}")
            state = torch.load(args.init_model, map_location="cpu")
            sd = {k.replace("module.", "", 1): v for k, v in state["model_state_dict"].items()}
            model.load_state_dict(sd, strict=False)
        model.to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model, device_ids=[0] if device.startswith("cuda") else None)
        model.to(device)
        if args.init_model:
            print(f"# init weights from {args.init_model}")
            state = torch.load(args.init_model, map_location=device)
            model.load_state_dict(state["model_state_dict"], strict=False)
```

تحمّل تهيئة DDP الحالة إلى CPU قبل نسخها على GPU، وتزيل سابقة `module.` واحدة لأن النموذج المجرد يملك مفاتيح بلا السابقة؛ ويضيف DDP السابقة مجددًا بعد التغليف. يلزم `find_unused_parameters=True` لأن Stage 1، أي `pos_r=0`، قد تتجاوز فرع الإرساء فتترك معاملات بلا تدرجات؛ ومن دون كشفها قد ينتظر DDP اختزالات لن تصل. ويحافظ المسار غير الموزّع على توافق نقاط التحقق ذات سابقة `DataParallel`.

```python
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_ckpt = os.path.join(args.out_dir, "karmadock_team002.pkl")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    stopper = None if is_ddp else Early_stopper(model_file=best_ckpt, mode="lower", patience=args.patience)
    best_val = None
    bad_epochs = 0
```

يحافظ نمط العملية الواحدة على آلية الإيقاف الأصلية. ويستخدم DDP حالة يدوية كي تحدّث كل رتبة عداد patience نفسه بينما تلمس الرتبة 0 الملفات وحدها، متجنبة تعارض الكتابة.

```python
    start_epoch = 0
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_score", None)
        bad_epochs = ck.get("bad_epochs", 0)
        if stopper is not None:
            stopper.best_score = best_val
        if main_proc:
            print(f"# resumed from {last_ckpt} at epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, "train_log.csv")
    if start_epoch == 0 and main_proc:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_rmsd,train_mdn,val_loss,val_rmsd,val_mdn,seconds\n")
```

تقرأ الرتب كلها نقطة تحقق الاستئناف التي كتبتها الرتبة 0، ومنها `bad_epochs` في نمط DDP. ولا تهيئ CSV المشترك إلا الرتبة 0.

### التدريب والتراكم والاختزالات {#training-accumulation-and-reductions}

```python
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        model.train()
        tr_tot, tr_rmsd, tr_mdn, n = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()
        accum = 0
        for data in train_loader:
            if data is None:
                continue
            data = data.to(device)
            rmsd_loss, mdn_loss = model(data, device, args.pos_r)
            if mdn_loss is None:
                continue
            loss = args.pos_r * rmsd_loss + mdn_loss
            boundary = ((accum + 1) % args.accum_steps == 0)
            with maybe_no_sync(model, is_ddp and not boundary):
                (loss / args.accum_steps).backward()
```

تتدرب كل رتبة على شريحتها المحلية. والقصد هو تجاوز اتصال التدرجات بين وحدات GPU حتى حد التراكم. يضع المصدر `backward()` فقط داخل `no_sync`؛ وتوضح المحاذير أهمية ذلك الموضع. وكما في `train.py` لا يُستخدم AMP أو autocast.

```python
            accum += 1
            if accum % args.accum_steps == 0:
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            tr_tot += float(loss); tr_rmsd += float(rmsd_loss); tr_mdn += float(mdn_loss); n += 1
        if accum % args.accum_steps != 0:
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
```

تحدّث النوافذ المكتملة بعد قص التدرجات، كما تحدّث النافذة الجزئية الأخيرة. وتُحفظ مجاميع المقاييس المحلية حتى التحقق لتُختزل عالميًا.

```python
        if is_ddp:
            s = list(validate_sums(model, valid_loader, device, args.pos_r))
            tr_tot, tr_rmsd, tr_mdn, n, s0, s1, s2, vn = all_reduce_sum(
                [tr_tot, tr_rmsd, tr_mdn, n, s[0], s[1], s[2], s[3]], device)
            n = max(n, 1); vn = max(vn, 1)
            tr_tot, tr_rmsd, tr_mdn = tr_tot / n, tr_rmsd / n, tr_mdn / n
            val_loss, val_rmsd, val_mdn = s0 / vn, s1 / vn, s2 / vn
        else:
            n = max(n, 1)
            tr_tot, tr_rmsd, tr_mdn = tr_tot / n, tr_rmsd / n, tr_mdn / n
            v_tot, v_rmsd, v_mdn = karmadock_evaluation(model, valid_loader, device, args.pos_r)
            val_loss = float(v_tot.mean()); val_rmsd = float(v_rmsd.mean()); val_mdn = float(v_mdn.mean())
        secs = time.perf_counter() - t0
```

ينتج تحقق DDP مجاميع محلية، ثم يجمع اختزال شامل واحد ثمانية أرقام بكفاءة. تحصل كل رتبة على المقاييس العامة نفسها. ويفوض النمط غير الموزّع إلى المقيّم الأصلي محافظًا على توافق الحلقة الأصلية.

```python
        if main_proc:
            print(f"[epoch {epoch:03d}] train {tr_tot:.3f} (rmsd {tr_rmsd:.3f}/mdn {tr_mdn:.3f}) | "
                  f"val {val_loss:.3f} (rmsd {val_rmsd:.3f}/mdn {val_mdn:.3f}) | {secs:.0f}s")
            with open(log_path, "a") as f:
                f.write(f"{epoch},{tr_tot:.5f},{tr_rmsd:.5f},{tr_mdn:.5f},"
                        f"{val_loss:.5f},{val_rmsd:.5f},{val_mdn:.5f},{secs:.1f}\n")
            if wb is not None:
                wb.log({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                        "train/loss": tr_tot, "train/rmsd": tr_rmsd, "train/mdn": tr_mdn,
                        "val/loss": val_loss, "val/rmsd": val_rmsd, "val/mdn": val_mdn,
                        "epoch_seconds": secs}, step=epoch)
```

يمنع قيد الرتبة تكرار أسطر الطرفية وكتابات الملفات وW&B. تأتي الثواني المسجلة من الرتبة 0، لكن العملية الجماعية السابقة تجعلها تشمل انتظار وصول الرتب الأخرى إلى اختزال المقاييس.

### الإيقاف المبكر المتزامن وإنهاء التشغيل {#lockstep-early-stopping-and-teardown}

```python
        if is_ddp:
            improved = (best_val is None) or (val_loss < best_val)
            if improved:
                best_val = val_loss; bad_epochs = 0
            else:
                bad_epochs += 1
            if main_proc:
                if improved:
                    torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_score": best_val, "bad_epochs": bad_epochs}, last_ckpt)
            if bad_epochs >= args.patience:
                if main_proc:
                    print(f"# early stopping at epoch {epoch}")
                break
```

لأن `val_loss` اختُزلت شاملًا، تحدّث كل رتبة `best_val` و`bad_epochs` بالقيم نفسها وتنهي الحلقة معًا. التحسن هو قيمة أصغر تمامًا؛ ويستهلك التعادل patience. لا تسلسل إلا الرتبة 0، لكن مزامنة تدرجات DDP تجعل النسخ متكافئة.

```python
        else:
            stopper.step(val_loss, model)
            best_val = stopper.best_score
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": stopper.best_score}, last_ckpt)
            if stopper.early_stop:
                print(f"# early stopping at epoch {epoch}")
                break
```

يحافظ Python العادي على سلوك آلية الإيقاف الأصلية وصيغة استئناف العملية الواحدة.

```python
    if wb is not None:
        wb.summary["best_val_loss"] = float(best_val) if best_val is not None else None
        try:
            art = __import__("wandb").Artifact("karmadock_team002", type="model")
            art.add_file(best_ckpt)
            wb.log_artifact(art)
        except Exception as e:
            print(f"# W&B artifact upload skipped ({e})")
        wb.finish()
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if main_proc:
        print(f"# done. best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
```

تنهي الرتبة 0 تشغيل W&B بينما تتقدم الرتب الأخرى إلى الحاجز النهائي، فتنتظر كلها قبل تفكيك NCCL. ويحرر تدمير مجموعة العمليات الموارد الموزعة بصورة سليمة.

## محاذير وملاحظات {#gotchas--notes}

- توثّق PyTorch أن سياق `no_sync()` في DDP ينبغي أن يشمل المرور الأمامي؛ وهنا يحدث `model(...)` قبل السياق ولا يقع داخله إلا `backward()`. لذلك قد لا يعمل توفير الاتصال المقصود. وسيتطلب نقل المرور الأمامي إلى الداخل أيضًا معالجة نافذة التراكم الجزئية الأخيرة كي تتزامن تدرجاتها قبل التحديث.
- مثل `train.py`، تُقسم نافذة التراكم الجزئية على `accum_steps` الكامل، فينخفض مقدارها.
- تقسيم المعرّفات اليدوي ليس `DistributedSampler`: عضوية كل رتبة ثابتة، ولا تفعل `shuffle=True` إلا تبديل الترتيب داخل الشريحة.
- تستخدم الرتب كلها البذرة نفسها. هذا حتمي، لكن تدفقات زيادة البيانات المستقلة عن الرتبة قد تكون مترابطة بين الرتب.
- يمكن استئناف حالة patience اليدوية في DDP؛ أما النمط غير الموزّع فما زال يملك غموض عداد آلية الإيقاف الموصوف في شرح العملية الواحدة.
- يفترض مسار DDP وجود CUDA وNCCL. وستختار قيمة `WORLD_SIZE>1` على جهاز بلا CPU فقط NCCL وCUDA وتفشل.
