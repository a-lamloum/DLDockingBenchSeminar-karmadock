# شرح `scripts/seminar_csv.py` {#scriptsseminarcsvpy}

## الغرض {#purpose}

تعطي هذه الوحدة المنظومة تعريفًا واحدًا **لمعرّف المركّب**: الاسم المستقر لزوج واحد من جيب البروتين والربيطة. تخفي الفرق بين مخطط CSV الأولي الذي يحتوي أسماء الملفات، ومخطط البيانات الكاملة الذي يحتوي أربعة حقول وصفية يجب بناء الأسماء منها.

## موقعه في المنظومة {#how-it-fits-in}

يستورد المحولان `complex_records`، ويستورد المدرّبان `complex_ids`. المدخل مسار CSV، والمخرجات أزواج مرتبة من `(complex_id, ligand_filename, protein_filename)` أو المعرّفات وحدها، ويجب أن تطابق تمامًا جذوع ملفات الرسم `.dgl` في KarmaDock.

## شرح الشيفرة {#walkthrough}

```python
import pandas as pd

ID_COLUMNS = ["PDBID", "Ligand Name", "Ligand Chain", "Ligand Residue Number"]
```

يثبت الثابت الحقول وترتيبها معًا. سيغير تغيير ترتيبها المسارات وأسماء الرسوم.

```python
def _strip_suffix(ligand_filename):
    return ligand_filename.replace("_ligand_refined.sdf", "").replace("_ligand.sdf", "")
```

يستخرج هذا المساعد الخاص المعرّف من أسماء الربيطة المنقحة أو الأقصر. لا يقيد `replace` المطابقة بنهاية الاسم، مع أن المدخل الصحيح يستخدمها كلواحق.

```python
def _rows_to_triples(df):
    """Yield (complex_id, ligand_filename, protein_filename) per CSV row, either schema."""
    if "ligand_file_name" in df.columns:
        for lig, prot in zip(df["ligand_file_name"], df["protein_file_name"]):
            yield _strip_suffix(lig), lig, prot
    else:
        for parts in zip(*[df[c] for c in ID_COLUMNS]):
            cid = "_".join(parts)
            yield cid, f"{cid}_ligand_refined.sdf", f"{cid}_protein_refined.pdb"
```

يختار هذا المولد المخطط الأولي بوجود `ligand_file_name`؛ وإلا يحوّل أعمدة البيانات الوصفية الأربعة إلى صفوف، ويصلها بشرطات سفلية، ويبني أسماء الملفات المنقحة. تمنع مركزة الفرع اختلاف اشتقاق المعرّفات بين التحويل والتدريب.

```python
def complex_records(csv_path):
    """Unique (complex_id, ligand_filename, protein_filename), in first-seen order."""
    df = pd.read_csv(csv_path, dtype=str)
    seen, records = set(), []
    for cid, lig, prot in _rows_to_triples(df):
        if cid not in seen:
            seen.add(cid)
            records.append((cid, lig, prot))
    return records
```

يمنع `dtype=str` تحوّل رقم بقايا مثل `210` إلى `210.0`. تتيح المجموعة فحص التكرار بسرعة، بينما تحفظ القائمة ترتيب الظهور الأول؛ ويُرجع المركّب المكرر مرة واحدة عمدًا.

```python
def complex_ids(csv_path):
    """The unique complex ids (== .dgl graph stems), in first-seen order."""
    return [cid for cid, _lig, _prot in complex_records(csv_path)]
```

يسقط هذا كل ثلاثية إلى جذر الرسم. المتغيرات ذات الشرطة السفلية غير مستخدمة عمدًا.

## تنبيهات وملاحظات {#gotchas--notes}

- يفحص اختيار المخطط `ligand_file_name` فقط؛ ويفشل المخطط الجزئي لاحقًا بخطأ عمود مفقود.
- يستخدم CSV الخاص بـPoseBusters المرفق `ligand_file`، لا أيًا من المخططين المعروفين، ولذلك لا يتوافق حاليًا مع هذا المساعد.
- تستخدم `_strip_suffix` الاستبدال بدل إزالة لاحقة نهائية صارمة.
