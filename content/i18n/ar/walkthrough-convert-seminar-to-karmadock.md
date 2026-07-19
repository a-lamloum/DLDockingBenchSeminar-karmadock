# شرح `scripts/convert_seminar_to_karmadock.py` {#scriptsconvertseminartokarmadockpy}

## الغرض {#purpose}

يحوّل هذا الأمر تخطيط الملفات المسطح للندوة إلى التخطيط المنفصل لكل مركّب الذي تتطلبه المعالجة المسبقة في KarmaDock الأصلي. ينسخ البنى ولا يحولها كيميائيًا، ولذلك لا يتغير سوى المجلدات وأسماء الملفات.

## موقعه في المنظومة {#how-it-fits-in}

تستدعيه `run_train.sh` و`run_full_train.sh` و`run_infer.sh` قبل استخراج الجيب وتوليد الرسوم البيانية في الشيفرة الأصلية. يأخذ `--csv` و`--src_dir` و`--out_dir`، ثم يكتب `<out_dir>/<id>/<id>_ligand.sdf` و`<id>_protein.pdb`.

## شرح الشيفرة {#walkthrough}

```python
import argparse
import os
import shutil
import sys

from seminar_csv import complex_records
```

ينسخ `shutil.copy2` المحتويات ويحافظ على البيانات الوصفية حيثما أمكن، بينما يضمن مساعد CSV المشترك اتساق المعرّفات.

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="seminar CSV (e.g. proto_train.csv)")
    ap.add_argument("--src_dir", required=True, help="dir with the refined .sdf/.pdb files")
    ap.add_argument("--out_dir", required=True, help="output KarmaDock-layout dir")
    args = ap.parse_args()

    for path in (args.csv, args.src_dir):
        if not os.path.exists(path):
            sys.exit(f"ERROR: not found: {path}")
```

جميع المسارات إلزامية. يُتحقق من المدخلين قبل بدء عمل الخرج، وتطبع `sys.exit(string)` الخطأ وتنهي التنفيذ بحالة غير صفرية.

```python
    os.makedirs(args.out_dir, exist_ok=True)
    records = complex_records(args.csv)
    ok = miss = 0
    for cid, lig, prot in records:
        s_lig, s_prot = os.path.join(args.src_dir, lig), os.path.join(args.src_dir, prot)
        if not (os.path.exists(s_lig) and os.path.exists(s_prot)):
            miss += 1
            continue
```

يمكن لإعادة التشغيل استخدام مجلد الخرج نفسه. لا يكون المركّب صالحًا إلا عند وجود الشريكين الجزيئيين؛ وتُحصى الأزواج الناقصة ثم تُتجاوز.

```python
        d = os.path.join(args.out_dir, cid)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(s_lig, os.path.join(d, f"{cid}_ligand.sdf"))
        shutil.copy2(s_prot, os.path.join(d, f"{cid}_protein.pdb"))
        ok += 1
    print(f"converted {ok}/{len(records)} complexes ({miss} missing source files)")


if __name__ == "__main__":
    main()
```

يحصل كل مركّب ناجح على الأسماء الدقيقة التي تتوقعها الشيفرة الأصلية. يكشف الملخص فقد البيانات من دون جعل بضعة إخفاقات في المعالجة أخطاء قاتلة. يمنع حاجز الوحدة النسخ عند الاستيراد.

## تنبيهات وملاحظات {#gotchas--notes}

- لا تسبب الأزواج المفقودة خروجًا غير صفري؛ يجب فحص عدد الرسوم البيانية اللاحق والملخص.
- تُستبدل الملفات الهدف الموجودة، بينما تبقى مجلدات الخرج القديمة غير المرتبطة.
- استخراج الجيب وبناء الرسم البياني خطوتان أصليتان لاحقتان وليستا جزءًا من هذا المحول.
