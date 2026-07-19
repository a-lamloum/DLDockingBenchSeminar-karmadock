# شرح `scripts/convert_karmadock_to_seminar.py` {#scriptsconvertkarmadocktoseminarpy}

## الغرض {#purpose}

يحوّل هذا الأمر مخرجات KarmaDock المنفصلة لكل تكرار إلى تنسيق المقيّم الذي يستخدم ملفًا واحدًا لكل مركّب. يرتّب التنبؤات المتكررة وفق درجة شبكة الكثافة الخليطية (MDN)، ثم يكتب ملف SDF متعدد السجلات بحيث تأتي الوضعية الأعلى درجة أولًا.

## موقعه في المنظومة {#how-it-fits-in}

يستدعيه `run_infer.sh` للمخرجات الخام، والمصححة بحقل القوة، والمصححة بالمحاذاة بعد `ligand_docking.py` الأصلي. يقرأ `<input_dir>/<repeat>.csv` وملفات SDF للوضعيات، وملف CSV للمجموعة، و`--mode`، و`--n_repeat` الاختياري ذي القيمة الافتراضية 3، ثم يكتب `<out_dir>/<id>_pred.sdf` وفق ترتيب المقيّم.

## شرح الشيفرة {#walkthrough}

```python
import argparse
import os
import sys

import pandas as pd
from rdkit import Chem

from seminar_csv import complex_records
```

تقرأ pandas الدرجات، بينما تحلل RDKit سجلات SDF الجزيئية وإحداثياتها وتكتبها.

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="KarmaDock out_dir (has 0.csv, 0/, ...)")
    ap.add_argument("--csv", required=True, help="seminar CSV listing the complexes")
    ap.add_argument("--out_dir", required=True, help="seminar results dir")
    ap.add_argument("--mode", default="align_corrected",
                    choices=["uncorrected", "ff_corrected", "align_corrected"],
                    help="which corrected pose to export (paper recommends FF/align correction)")
    ap.add_argument("--n_repeat", type=int, default=3)
    args = ap.parse_args()

    for path in (args.csv, args.input_dir):
        if not os.path.exists(path):
            sys.exit(f"ERROR: not found: {path}")
    os.makedirs(args.out_dir, exist_ok=True)
```

يرفض `choices` أوضاعًا مكتوبة خطأ. يُتحقق من المدخلين ويُنشأ مجلد الخرج بأمان.

```python
    scores = {}
    repeats = []
    for re in range(args.n_repeat):
        csv_path = os.path.join(args.input_dir, f"{re}.csv")
        if not os.path.exists(csv_path):
            continue
        repeats.append(re)
        for _, row in pd.read_csv(csv_path).iterrows():
            scores.setdefault(row["pdb_id"], []).append({"repeat": re, "score": float(row["score"])})
    if not repeats:
        sys.exit("ERROR: no <re>.csv files found in input_dir")
```

يساهم جدول درجات كل تكرار بأزواج التكرار والدرجة لكل مركّب. تمثل MDN احتمالات مسافات التفاعل كتوزيع خليط، وتُستخدم درجتها القياسية لترتيب الوضعيات بحيث تعد القيمة الأعلى أفضل. يُسمح بغياب تكرارات منفردة، لكن غياب جميع جداول التكرار خطأ قاتل.

```python
    records = complex_records(args.csv)
    ok = 0
    for cid, _lig, _prot in records:
        ranked = sorted(scores.get(cid, [{"repeat": r, "score": 0.0} for r in repeats]),
                        key=lambda x: x["score"], reverse=True)
        out_path = os.path.join(args.out_dir, f"{cid}_pred.sdf")
        writer = Chem.SDWriter(out_path)
        written = 0
```

تحدد الدرجة التنازلية ترتيب سجلات SDF. إذا لم تكن للمركّب درجات، تحصل التكرارات على درجات صفرية متساوية، ويحافظ الفرز المستقر في Python على ترتيب التكرارات.

```python
        for info in ranked:
            re = info["repeat"]
            pose = os.path.join(args.input_dir, str(re), f"{cid}_pred_{args.mode}.sdf")
            if not os.path.exists(pose) and args.mode != "uncorrected":
                pose = os.path.join(args.input_dir, str(re), f"{cid}_pred_uncorrected.sdf")
            if not os.path.exists(pose):
                continue
            for mol in Chem.SDMolSupplier(pose, removeHs=False):
                if mol is None:
                    continue
                mol.SetProp("KarmaDock_Score", f"{info['score']:.4f}")
                mol.SetProp("KarmaDock_Repeat", str(re))
                mol.SetProp("KarmaDock_Mode", args.mode)
                writer.write(mol)
                written += 1
```

يُرخي **تصغير حقل القوة (FF)** البنية الهندسية باستخدام ميكانيكا جزيئية، بينما تضع **المحاذاة المصححة** الإحداثيات فوق إطار مرجعي. عند غياب ملف مصحح، يعود البرنامج إلى الوضعية الخام؛ ويتجاوز الملفات أو السجلات المفقودة وغير الصالحة. يحتفظ بالهيدروجينات الصريحة وبخصائص المصدر الثلاث.

```python
        writer.close()
        if written:
            ok += 1
        elif os.path.exists(out_path):
            os.remove(out_path)
    print(f"wrote {ok}/{len(records)} predicted-pose files to {args.out_dir}")


if __name__ == "__main__":
    main()
```

يؤدي الإغلاق إلى تفريغ الخرج. تُحذف الملفات الفارغة كي يعني وجود الملف أن فيه وضعية واحدة على الأقل. يحصي الملخص المركّبات لا سجلات الوضعيات.

## تنبيهات وملاحظات {#gotchas--notes}

- رتبة المقيّم هي ترتيب الملف، لذلك لا يمكن فصل الفرز التنازلي عن ترتيب الكتابة.
- قد يحتوي مجلد مصحح على إحداثيات خام احتياطية؛ تسجل `KarmaDock_Mode` الوضع المطلوب لا مصدر الملف الاحتياطي.
- إذا وُجدت أي درجة لمركّب، فلن تُؤخذ التكرارات الغائبة عن صفوف درجاته في الحسبان حتى لو وُجدت ملفات وضعياتها.
- تُتجاوز أخطاء تحليل RDKit بدل إنهاء التحويل.
