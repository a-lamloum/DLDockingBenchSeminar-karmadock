# شرح `scripts/train.py` {#scriptstrainpy}

## الغرض {#purpose}

يوفر هذا الملف حلقة التدريب أحادية العملية المفقودة من KarmaDock الأصلي. يصل الرسوم الجزيئية المحسوبة مسبقًا بتمرير KarmaDock الأمامي الموجود، ويجمع خسارتي إحداثيات الوضعية وتسجيل شبكة الكثافة الخليطية (MDN)، وينفذ التحسين والتحقق، ويكتب نقاط تحقق قابلة للاستئناف وأخرى ذات أفضل تحقق.

## موقعه في المنظومة {#how-it-fits-in}

يستخدمه `run_train.sh` للتدريب الأولي من الصفر والضبط الدقيق، ويستخدمه `run_full_train.sh` لتقسيم التدريب والتحقق المنقح الكبير. يستورد أدوات النموذج والبيانات من `/app/KarmaDock` ومعرّفات CSV من `scripts/seminar_csv.py`. المدخلات الإلزامية هي `--csv` و`--graph_dir` و`--out_dir`؛ وتشمل الاختيارات CSV ومجلد رسوم منفصلين للتحقق، ونقطة تهيئة، وحالة الاستئناف، ومعاملات التحسين، وارتعاش الإحداثيات، وإعدادات W&B. المخرجات هي `karmadock_team002.pkl` للأفضل، و`last.pt` للاستئناف، و`train_log.csv`، وملفات W&B الاختيارية في `out_dir`.

## شرح الشيفرة {#walkthrough}

يوثق الملف أولًا الواجهة الأصلية الحاسمة ويستورد اعتمادياته:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - Retraining / fine-tuning loop for KarmaDock.

The model's own forward pass already returns the two training losses:

    rmsd_loss, mdn_loss = model(data, device, pos_r)
    loss = pos_r * rmsd_loss + mdn_loss

This script wraps those into a complete, checkpointed training loop.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from utils.fns import set_random_seed, Early_stopper, karmadock_evaluation
from dataset.graph_obj import PDBBindGraphDataset
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.KarmaDock_architecture import KarmaDock

from seminar_csv import complex_ids
```

يقيس **الجذر التربيعي لمتوسط مربع الانحراف (RMSD)** متوسط خطأ الإحداثيات بالأنغستروم بعد مطابقة ذرات الربيطة؛ ويوفر التمرير الأمامي الأصلي هنا خسارة وضعية مرتبطة بـRMSD. تتنبأ **MDN** بتوزيع احتمالي خليطي بدل نقطة واحدة، ويستخدمها KarmaDock لتمثيل مسافات تفاعل البروتين والربيطة واستخراج خسارة التسجيل. يتحكم `pos_r` في مساهمة تعلم الإحداثيات. تبقى تفاصيل ترميز GVP والمحوّل وتمريرات تحسين الإحداثيات المتكافئة المعاد تدويرها ثلاث مرات داخل `KarmaDock`؛ وتعني إعادة التدوير تمرير الوضعية المحدثة عبر وحدة التحسين مرارًا.

### تحليل الوسائط {#argument-parsing}

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
```

تصف هذه الوسائط مصدر البيانات ودورة حياتها. يعد `--val_csv` الصريح مهمًا للتقسيمات المنقحة أو المقاومة للتسرب؛ فقد يضع اقتطاع تحقق عشوائي من التدريب أنظمة بروتين أو ربيطة وثيقة الصلة في الجانبين. تعني `store_true` أن `--resume` يكون false افتراضيًا ويصبح true بمجرد ظهوره.

```python
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
    p.add_argument("--jitter", type=float, default=0.0,
                   help="Scale of coordinate jittering (std in Å) for training augmentation.")
```

الدفعة الفيزيائية هي ما يشغل الذاكرة في الوقت نفسه. يجمع **تجميع التدرجات** تدرجات عدة دفعات صغيرة قبل `optimizer.step`؛ ويجعل تقسيم كل خسارة على `accum_steps` النافذة الكاملة تقارب متوسط تدرج الدفعة الفعلية الأكبر. يحد اضمحلال الأوزان من المعاملات الكبيرة، ويقيد قص التدرج معياره العالمي لتقليل التحديثات غير المستقرة. يضيف الارتعاش اضطرابات غاوسية صغيرة إلى إحداثيات الربيطة كي يرى النموذج وضعيات بدء قريبة بدل حفظ الوضعية الدقيقة.

```python
    p.add_argument("--wandb", action="store_true", help="log metrics to W&B")
    p.add_argument("--wandb_project", default="karmadock-seminar")
    p.add_argument("--wandb_entity", default=None, help="W&B team/user (optional)")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()
```

يكون W&B اختياريًا، ولا يُقبل مفتاح API كوسيط CLI، مما يقلل احتمال ظهوره في قوائم العمليات أو السجلات.

### اختيار الرسوم الصالحة وتقسيم البيانات {#selecting-usable-graphs-and-splitting-data}

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
```

تفشل بعض الجزيئات في تحليل RDKit أو بناء الجيب والرسم، لذلك ترشح الحلقة الملفات الفعلية `.dgl` بدل الإخفاق أثناء التدريب. ينتج `RandomState(seed)` تبديلًا قابلًا للتكرار مستقلًا عن مولد NumPy العالمي. يأخذ التحقق الكسر الأول بعد التقريب وعنصرًا واحدًا على الأقل. تجعل المجموعة فحص العضوية سريعًا، وتحافظ القائمتان على ترتيب المجموعة المصفاة داخل كل قسم.

```python
def available(ids, graph_dir):
    """Keep only ids that actually have a .dgl graph (used with an explicit val CSV)."""
    have = [i for i in ids if os.path.exists(os.path.join(graph_dir, f"{i}.dgl"))]
    missing = len(ids) - len(have)
    if missing:
        print(f"# WARNING: {missing}/{len(ids)} complexes have no .dgl graph and are skipped.")
    return have
```

عند وجود CSV تحقق صريح لا يحدث تقسيم؛ يطبق المساعد مرشح التوفر نفسه على كل جانب من التقسيم المنقح.

### إنشاء محملات البيانات {#constructing-loaders}

```python
def build_loader(ids, args, dataset_type, shuffle, graph_dir=None):
    ds = PDBBindGraphDataset(
        src_dir=args.complex_dir,
        pdb_ids=ids,
        dst_dir=graph_dir or args.graph_dir,
        dataset_type=dataset_type,
        n_job=1,
        on_the_fly=True,   # reload each <id>.dgl and re-randomise the start pose per epoch
        verbose=False,
    )
    return PassNoneDataLoader(
        dataset=ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, follow_batch=[], pin_memory=True,
    )
```

تحمّل مجموعة البيانات الأصلية كائنات رسوم البروتين والربيطة. يعيد `on_the_fly=True` تحميل كل رسم محفوظ ويسمح لمنطق KarmaDock بتعشية وضعية بدء الربيطة عند تكرار الوصول، وهي زيادة مناسبة لتعلم نقل ربيطة موضوعة خطأ إلى الجيب. يمكن للمحمل المخصص تجاوز العينات غير الصالحة أو `None`. تحضر العمليات العاملة الدفعات بالتوازي، وقد تسرع الذاكرة المضيفة المثبتة النقل إلى CUDA. يخلط التدريب بياناته، ولا يفعل التحقق.

### البدء وW&B الاختياري {#startup-and-optional-wb}

```python
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    set_random_seed(args.random_seed)
    print(f"# device: {device}")

    wb = None
    if args.wandb:
        try:
            import wandb
            has_creds = os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc"))
            if not has_creds and os.environ.get("WANDB_MODE") != "offline":
                print("# WARNING: no W&B credentials (WANDB_API_KEY / ~/.netrc) -> forcing offline")
                os.environ["WANDB_MODE"] = "offline"
            eff_batch = args.batch_size * args.accum_steps
            stage = "scoring" if args.pos_r == 0 else "docking"
            base = args.wandb_run_name or "kd"
```

تُفضل CUDA مع بقاء CPU بديلًا وظيفيًا. يُتوقع من مساعد البذرة الأصلي تهيئة المولدات المناسبة. يُستورد W&B عند الحاجة فقط، فلا تتطلبه التشغيلات التي لا تستخدمه. تنقل بيانات الاعتماد المفقودة التسجيل إلى وضع غير متصل بدل إيقاف التدريب. تسمية المرحلة وصفية؛ فأي `pos_r` غير صفري يسمى إرساءً.

```python
            rid_file = os.path.join(args.out_dir, "wandb_run_id.txt")
            if args.resume and os.path.exists(os.path.join(args.out_dir, "last.pt")) and os.path.exists(rid_file):
                run_id = open(rid_file).read().strip()
            else:
                run_id = f"{base}_{int(time.time())}_{os.getpid()}"
                with open(rid_file, "w") as _rf:
                    _rf.write(run_id)
            stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(int(run_id.rsplit('_', 2)[1])))
            run_name = (f"{base}_lr{args.lr:g}_bs{eff_batch}_ep{args.epochs}"
                        f"_pat{args.patience}_seed{args.random_seed}_{stamp}")
            cfg = dict(vars(args))
            cfg.update(effective_batch=eff_batch, stage=stage,
                       init_from=("scratch" if not args.init_model else os.path.basename(args.init_model)))
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            id=run_id, name=run_name, config=cfg,
                            dir=args.out_dir, resume="allow")
            print(f"# W&B run: {run_name}")
        except Exception as e:
            print(f"# WARNING: W&B disabled ({e})")
            wb = None
```

يتيح معرّف التشغيل المحفوظ للاستئناف الحقيقي الإلحاق بتشغيل W&B نفسه. تجمع المعرّفات الجديدة الاسم الأساسي وطابع Unix الزمني وPID، ويولد تحليل الطابع اسمًا مقروءًا. يلتقط `vars(args)` إعداد التجربة كله. يعامل الاستثناء الواسع المراقبة كأمر اختياري، فلا ينبغي لإخفاق التسجيل إهدار تدريب مكلف. يطابق `open(...).read()` المجرد المصدر ويعتمد على التنظيف المعتاد لكائن الملف بخلاف `with` اللاحق.

### تهيئة البيانات والنموذج {#data-and-model-initialization}

```python
    val_graph_dir = args.val_graph_dir or args.graph_dir
    if args.val_csv:
        train_ids = available(complex_ids(args.csv), args.graph_dir)
        valid_ids = available(complex_ids(args.val_csv), val_graph_dir)
    else:
        train_ids, valid_ids = split_available(
            complex_ids(args.csv), args.graph_dir, args.val_frac, args.random_seed)
    print(f"# train: {len(train_ids)} | valid: {len(valid_ids)} complexes")
    train_loader = build_loader(train_ids, args, "train", shuffle=True)
    valid_loader = build_loader(valid_ids, args, "valid", shuffle=False, graph_dir=val_graph_dir)
```

يعود مسار رسوم التحقق الفارغ إلى مجلد رسوم التدريب. يستخدم التحقق الصريح جميع معرّفات التدريب الصالحة وجميع معرّفات التحقق الصالحة؛ وإلا يُنشأ تقسيم عشوائي حتمي واحد.

```python
    model = KarmaDock()
    model = nn.DataParallel(model, device_ids=[0] if device.startswith("cuda") else None)
    model.to(device)
    if args.init_model:
        print(f"# init weights from {args.init_model}")
        state = torch.load(args.init_model, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
```

يغلف `DataParallel` النموذج ويضيف `module.` إلى أسماء معاملات نقطة التحقق، وهو ما يتوقعه الاستدلال الأصلي. لا يُدرج هنا إلا GPU 0، لذلك يوحّد الغلاف التنسيق ولا يقدم توسعًا متعدد GPU. يحمّل `map_location` الموترات إلى الجهاز المختار. يسمح `strict=False` بمفاتيح مفقودة أو زائدة، وهو مفيد بين نقاط مترابطة لكنه قد يخفي عدم التوافق. يحتفظ Adam بتقديرات تكيفية للعزمين الأول والثاني لكل معامل.

```python
    best_ckpt = os.path.join(args.out_dir, "karmadock_team002.pkl")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    stopper = Early_stopper(model_file=best_ckpt, mode="lower", patience=args.patience)
    best_val = None

    start_epoch = 0
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_score", None)
        stopper.best_score = best_val
        print(f"# resumed from {last_ckpt} at epoch {start_epoch}")
```

تخدم نقطتا التحقق غرضين مختلفين: يكتب `Early_stopper` الأصلي أفضل نموذج للاستدلال، بينما يحتوي `last.pt` أيضًا حالة المحسّن والحقبة للاستمرار الدقيق. يكون تحميل الاستئناف صارمًا لأنه يعيد البنية نفسها، ويبدأ بعد الحقبة المحفوظة. تستعاد أفضل درجة، لكن عداد الحقب السيئة المتتالية في الموقف لا يُستعاد صراحة.

```python
    log_path = os.path.join(args.out_dir, "train_log.csv")
    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_rmsd,train_mdn,val_loss,val_rmsd,val_mdn,seconds\n")
```

ينشئ التشغيل الجديد CSV بسيطًا للتحليل أو يفرغه، بينما يلحق الاستئناف صفوفه ويحافظ على السابقة.

### التدريب وتجميع التدرجات {#training-and-gradient-accumulation}

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
            if args.jitter > 0.0:
                pos = data['ligand'].pos
                noise = torch.randn_like(pos) * args.jitter
                data['ligand'].pos = pos + noise
            rmsd_loss, mdn_loss = model(data, device, args.pos_r)
            if mdn_loss is None:
                continue
            loss = args.pos_r * rmsd_loss + mdn_loss
            (loss / args.accum_steps).backward()
```

يفعّل `model.train()` سلوك التدريب مثل dropout. تنتقل دفعات الرسوم إلى الجهاز. يسحب الارتعاش إزاحة طبيعية مستقلة بالشكل والجهاز والنوع نفسيهما لمواضع الربيطة. تجمع `.backward()` التدرجات في مخازن `.grad`، ولا يمسحها PyTorch تلقائيًا. لا يوجد AMP أو autocast؛ تستخدم العمليات الدقة العادية للموترات والنموذج، فلا تُطبق وفورات ذاكرة الدقة المختلطة التلقائية.

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

عند حد التجميع تُقص التدرجات، ويحدث Adam المعاملات، وتُمسح المخازن. يستخرج `float(cuda_tensor)` قيمة قياسية ويزامن الجهاز بالمضيف، فيبادل بعض السرعة بتسجيل مباشر. تُفرغ النافذة النهائية غير المكتملة حتى لا تُهمل العينات.

### التحقق والتسجيل {#validation-and-logging}

```python
        n = max(n, 1)
        tr_tot, tr_rmsd, tr_mdn = tr_tot / n, tr_rmsd / n, tr_mdn / n
        v_tot, v_rmsd, v_mdn = karmadock_evaluation(model, valid_loader, device, args.pos_r)
        val_loss = float(v_tot.mean()); val_rmsd = float(v_rmsd.mean()); val_mdn = float(v_mdn.mean())
        secs = time.perf_counter() - t0
```

يمنع `max` القسمة على صفر، لكن غياب دفعات تدريب صالحة سيعطي متوسطات تدريب صفرية ولا يعد قاتلًا هنا. يفوض التحقق إلى الشيفرة الأصلية المتوقع منها تعطيل التدرجات وضبط وضع التقييم وإرجاع مصفوفات أو موترات للخسائر. تلخص المتوسطات الدفعات لا المركّبات صراحة، لذلك يعتمد الوزن على بنية الخرج الأصلية.

```python
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

تستقبل السجلات المقروءة وCSV المحلي والتتبع الاختياري المقاييس نفسها. يعد `perf_counter` مؤقتًا رتيبًا عالي الدقة مناسبًا للمدة المنقضية.

### نقاط التحقق والإيقاف المبكر والإنهاء {#checkpointing-early-stopping-and-shutdown}

```python
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

خسارة التحقق الأقل أفضل. تحفظ آلية الإيقاف الأصلية أفضل نموذج إلى `best_ckpt` وتعلن الإيقاف بعد قاعدة patience. يجعل حفظ `last.pt` المستقل كل حقبة استئناف العنقود ممكنًا حتى عندما لا تكون أحدث حقبة الأفضل.

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
    print(f"# done. best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
```

تسجل تشغيلات W&B الناجحة أفضل خسارة وتحاول رفع أفضل نقطة تحقق كأثر نموذج. لا يكون إخفاق الأثر قاتلًا. يمنع حاجز الوحدة بدء التدريب عند الاستيراد.

## تنبيهات وملاحظات {#gotchas--notes}

- تُقسم نافذة التجميع الجزئية النهائية على `accum_steps` الكامل، فتسهم بتدرج أصغر من نافذة كاملة بدل إعادة تطبيعها بعدد دفعاتها الفعلي.
- يستعيد الاستئناف `best_score` لا عداد الحقب السيئة الصريح لآلية الإيقاف. يعتمد بقاء patience كاملًا بعد إعادة الجدولة على تنفيذ `Early_stopper` الأصلي غير الموجود في المستودع.
- قد يخفي تحميل `strict=False` عدم تطابق مفاتيح نقطة التحقق والنموذج؛ ولا يطبع البرنامج المفاتيح المفقودة أو غير المتوقعة.
- لا تُرفض مجموعات الرسوم الفارغة أو شبه الفارغة صراحة. يفرض `split_available` أيضًا معرّف تحقق واحدًا، ولذلك تترك مجموعة برسم واحد دون معرّفات تدريب.
- لا يوجد مجدول لمعدل التعلم ولا AMP أو autocast. هذه نواقص في التنفيذ الفعلي لا سلوكًا أصليًا ضمنيًا.
