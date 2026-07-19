# `evaluation/evaluation.py` {#evaluationevaluationpy}

## الغرض {#purpose}

هذا مقيّم الإرساء على مستوى مجموعة البيانات في المستودع. يربط صفوف CSV بالبنى المرجعية والمتنبأ بها، ويحسب RMSD للذرات الثقيلة مع مراعاة التناظر للوضعيات المرتبة، ويشغّل اختياريًا فحوص الصلاحية الفيزيائية من PoseBusters، ويطبع إحصاءات top-1 والأفضل من N، ويكتب صف نتيجة لكل وضعية مقيّمة.

## موقعه في المنظومة {#how-it-fits-in}

يستهلك `data/<dataset>.csv` والملفات المرجعية تحت `data/<dataset>/` والتنبؤات تحت `results/<dataset>/<complex_id>_pred.sdf`. تستدعيه `scripts/evaluate.sh` مرارًا عبر روابط رمزية مؤقتة لمقارنات النموذج الأولي، ويمكن للمستخدم تشغيله مباشرة. تتطلب واجهة CLI ‏`--dataset`، وتقبل `--top_n` و`--no_pb_valid` و`--output_csv`، ويكون الخرج الافتراضي `results/<dataset>_evaluation.csv`.

## شرح الشيفرة {#walkthrough}

### إعداد الوحدة والتسجيل {#module-setup-and-logging}

```python
#!/usr/bin/env python3
"""
updated_evaluation.py — Updated evaluation script for pocket-conditioned molecular docking tools.

This script evaluates docking predictions against reference structures using CSV-based dataset mapping.
Supports multiple dataset variants (proto_test, posebusters_filtered, full_test) with different naming conventions.

Computes symmetry-corrected heavy-atom RMSD and runs PoseBusters physical validity checks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
```

يوفّر `Path` مسارات قابلة للتركيب؛ وتصف تلميحات الأنواع القواميس والقوائم والجزيئات أو المسارات الاختيارية والثلاثيات المتوقعة، لكنها لا تفرض الأنواع وقت التشغيل. تحلل RDKit الجزيئات وتجري محاذاة الذرات. ويمنح مسجّل الوحدة طوابع زمنية ومستويات شدة مفيدة في تقييمات العنقود الطويلة.

### RMSD المصحح للتناظر {#symmetry-corrected-rmsd}

```python
def calc_symmetry_rmsd(pred_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
    """
    Compute symmetry-corrected heavy-atom RMSD between predicted and reference mol.

    Uses RDKit's GetBestRMS which accounts for molecular symmetry.
    """
    try:
        pred_noH = AllChem.RemoveHs(pred_mol)
        ref_noH = AllChem.RemoveHs(ref_mol)
        rmsd = rdMolAlign.GetBestRMS(pred_noH, ref_noH)
        return rmsd
    except Exception as e:
        logger.warning(f"RMSD calculation failed: {e}")
        return float("nan")
```

RMSD هو الجذر التربيعي لمتوسط مربع المسافة بين الذرات المتناظرة. تُزال ذرات الهيدروجين لأن مواضعها وأعدادها متغيرة ولأن معايير الإرساء تركز عادة على الذرات الثقيلة. يأخذ **تصحيح التناظر** تعيينات الذرات المتكافئة كيميائيًا في الحسبان؛ فلا ينبغي لتبديل ذرات لا يمكن تمييزها في مجموعة متناظرة أن يجعل وضعية مطابقة تبدو خاطئة. تختار `GetBestRMS` التعيين والمحاذاة الصلبة ذوي أقل RMSD. ويصبح أي عدم تطابق أو فشل في RDKit قيمة NaN كي لا ينهي جزيء واحد تقييم المجموعة.

من المهم أن `GetBestRMS` تجري محاذاة مكانية مثلى للمسبار إلى المرجع. يزيل ذلك الانتقال والدوران العامين قبل التقييم، مع أن RMSD للإرساء المشروط بالجيب يُقاس غالبًا في إطار البروتين الثابت. لذلك يقيس هذا التطبيق اتفاق الشكل والتكوين بعد التركيب، لا خطأ التموضع في الجيب وحده.

### تحميل الجزيئات المتنبأ بها والمرجعية {#loading-predicted-and-reference-molecules}

```python
def load_pred_poses(sdf_path: Path, top_n: int) -> List[Chem.Mol]:
    """Load up to top_n conformers from a multi-conformer SDF file."""
    if not sdf_path.exists():
        return []

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    poses = []
    for i, mol in enumerate(supplier):
        if i >= top_n:
            break
        if mol is not None:
            poses.append(mol)
        else:
            logger.warning(f"  Pose {i+1} in {sdf_path.name} could not be parsed, skipping.")
    return poses
```

قد يحتوي SDF عدة سجلات جزيئية، ويعامل المشروع ترتيبها في الملف بوصفه الرتبة. تحلل مورّدات RDKit السجلات كسولًا. يمنع `sanitize=False` رفض الهندسات المتنبأ بها بسبب فحوص الكيمياء، فيتيح محاولة RMSD على مخرجات غير مثالية؛ ويحفظ `removeHs=False` السجل أولًا ثم تُزال ذرات الهيدروجين صراحة عند حساب RMSD. تُتجاوز السجلات غير الصالحة، لذلك ينضغط موضع القائمة المعادة مع أن رتبة التحذير تعرض موضع السجل الأصلي.

```python
def load_ref_ligand(sdf_path: Path) -> Optional[Chem.Mol]:
    """Load single-conformer reference ligand from SDF."""
    if not sdf_path.exists():
        return None

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is not None:
            return mol
    return None
```

يُستخدم أول سجل مرجعي قابل للتحليل فقط. ويعطي إرجاع `None` المستدعي إشارة واحدة واضحة إلى الفقد أو تعذر التحليل.

### مخططات مجموعة البيانات ومساراتها {#dataset-schemas-and-paths}

```python
def get_complex_identifier(ligand_filename: str, dataset_name: str) -> str:
    """Extract complex identifier from ligand filename based on dataset naming convention."""
    if dataset_name in ["proto_test", "full_test", "proto_train", "full_train"]:
        return ligand_filename.replace("_ligand_refined.sdf", "")
    elif dataset_name == "posebusters_filtered":
        return ligand_filename.replace("_ligand.sdf", "")
    else:
        for suffix in ["_ligand_refined.sdf", "_ligand.sdf", ".sdf"]:
            if ligand_filename.endswith(suffix):
                return ligand_filename.replace(suffix, "")
        return ligand_filename
```

يربط معرّف المعقّد صفوف CSV بـ`<id>_pred.sdf`. تستخدم مجموعات المقرر المنقحة وPoseBusters لواحق مختلفة. وفرع الرجوع دفاعي مع أن argparse يقصر الاستدعاءات حاليًا على أسماء معروفة. تزيل `replace` كل ظهور للنص المحدد، لا الظهور النهائي فقط؛ وتحتويه أسماء الملفات المقصودة مرة واحدة عند النهاية.

```python
def load_dataset_csv(csv_path: Path) -> pd.DataFrame:
    """Load and standardize dataset CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "ligand_file_name" in df.columns:
        df = df.rename(columns={
            "ligand_file_name": "ligand_file",
            "protein_file_name": "protein_file"
        })
    elif "ligand_name" in df.columns:
        pass
    else:
        raise ValueError(f"Unrecognized CSV format in {csv_path}")

    return df
```

تتوقع الشيفرة اللاحقة عمودي `ligand_file` و`protein_file` القياسيين. تُعاد تسمية ملفات CSV لتقييم النموذج الأولي والبيانات الكاملة؛ أما PoseBusters فلديه العمودان ويُتعرف عليه عبر `ligand_name`. تفشل المخططات المجهولة مبكرًا بدل بناء مسارات خاطئة.

```python
def get_file_paths(row: pd.Series, dataset_name: str, data_dir: Path, results_dir: Path) -> Tuple[Path, Path, Path]:
    """Get file paths for reference protein, reference ligand, and predicted poses."""
    ligand_file = row["ligand_file"]
    protein_file = row["protein_file"]

    ref_ligand_path = data_dir / dataset_name / ligand_file
    ref_protein_path = data_dir / dataset_name / protein_file

    complex_id = get_complex_identifier(ligand_file, dataset_name)
    pred_sdf_path = results_dir / dataset_name / f"{complex_id}_pred.sdf"

    return ref_protein_path, ref_ligand_path, pred_sdf_path
```

تمركز هذه الدالة اصطلاح الأدلة الثابت. يأتي البروتين أولًا في ثلاثية الإرجاع لأن PoseBusters يشترط الصلاحية على الجيب، بينما يحتاج RMSD إلى مساري الربيطة فقط.

### تكامل PoseBusters {#posebusters-integration}

```python
def run_posebusters_check(pred_sdf: Path, ref_sdf: Path, protein_pdb: Path) -> pd.DataFrame:
    """Run PoseBusters validity checks on predicted poses."""
    try:
        from posebusters import PoseBusters

        buster = PoseBusters(config="dock")
        results = buster.bust(
            mol_pred=str(pred_sdf),
            mol_true=str(ref_sdf),
            mol_cond=str(protein_pdb),
        )
        return results
    except ImportError:
        logger.error("PoseBusters not installed. Install with: pip install posebusters")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"PoseBusters check failed: {e}")
        return pd.DataFrame()
```

يفحص PoseBusters معقولية الوضعية فيزيائيًا وكيميائيًا، مثل الهندسة الداخلية والتصادمات، باستخدام الربيطة المتنبأ بها والربيطة الحقيقية والبروتين الشرطي. يختار إعداد `dock` حزمة اختبارات الإرساء. تُعامل الحزمة المفقودة كخطأ إعداد ينهي التنفيذ، بينما يعيد فشل وقت التشغيل لمعقّد واحد DataFrame فارغًا كي يستمر التقييم.

### تقييم معقّد واحد {#evaluating-one-complex}

```python
def evaluate_complex(
    row: pd.Series,
    dataset_name: str,
    data_dir: Path,
    results_dir: Path,
    top_n: int,
    run_pb: bool
) -> List[Dict]:
    """Evaluate all poses for a single complex."""

    ref_protein_path, ref_ligand_path, pred_sdf_path = get_file_paths(
        row, dataset_name, data_dir, results_dir
    )

    complex_id = get_complex_identifier(row["ligand_file"], dataset_name)
```

صف CSV واحد هو وحدة التقييم. تُشتق المسارات والمعرّف مرة واحدة وتُستخدم في التشخيص وحقول النتائج.

```python
    missing_files = []
    if not pred_sdf_path.exists():
        missing_files.append(f"predicted poses: {pred_sdf_path}")
    if not ref_ligand_path.exists():
        missing_files.append(f"reference ligand: {ref_ligand_path}")
    if run_pb and not ref_protein_path.exists():
        missing_files.append(f"reference protein: {ref_protein_path}")

    if missing_files:
        logger.warning(f"  [{complex_id}] Missing files: {', '.join(missing_files)}")
        return []
```

لا يُطلب البروتين إلا لـPoseBusters، لا لـRMSD الخاص بالربيطة. ويمنح جمع المسارات المفقودة تحذيرًا واحدًا قابلًا للتنفيذ بدل التوقف عند أول مسار.

```python
    ref_mol = load_ref_ligand(ref_ligand_path)
    if ref_mol is None:
        logger.warning(f"  [{complex_id}] Could not parse reference ligand")
        return []

    pred_poses = load_pred_poses(pred_sdf_path, top_n)
    if not pred_poses:
        logger.warning(f"  [{complex_id}] No valid predicted poses found")
        return []
```

يبطل المرجع غير القابل للتحليل المقارنات كلها. ويجب أن يُحلل سجل متنبأ به واحد على الأقل داخل البادئة المطلوبة.

```python
    results = []
    for rank, pose in enumerate(pred_poses, start=1):
        rmsd = calc_symmetry_rmsd(pose, ref_mol)

        result_row = {
            "complex_id": complex_id,
            "dataset": dataset_name,
            "pose_rank": rank,
            "rmsd": round(rmsd, 4) if not np.isnan(rmsd) else np.nan,
            "rmsd_lt2": rmsd < 2.0 if not np.isnan(rmsd) else False,
            "rmsd_lt1": rmsd < 1.0 if not np.isnan(rmsd) else False,
            "ligand_file": row["ligand_file"],
            "protein_file": row["protein_file"]
        }

        results.append(result_row)
```

عتبة نجاح الإرساء المتعارف عليها هي RMSD أقل من 2 Å، وأقل من 1 Å أشد. لا تُعد NaN نجاحًا. يُخزن RMSD إلى أربع منازل عشرية، بينما تُحسب القيم المنطقية من القيمة غير المقربة. وبعد تجاوز سجلات SDF غير الصالحة، تُعاد ترقيم الرتب تتابعيًا من القائمة القابلة للتحليل بدل حفظ فهارس السجلات الأصلية.

```python
    if run_pb:
        pb_results = run_posebusters_check(pred_sdf_path, ref_ligand_path, ref_protein_path)

        if not pb_results.empty:
            for rank_idx in range(min(len(results), len(pb_results))):
                pb_row = pb_results.iloc[rank_idx]

                bool_cols = pb_row.index[pb_row.apply(lambda x: isinstance(x, (bool, np.bool_)))]
                pb_valid = bool(pb_row[bool_cols].all()) if len(bool_cols) > 0 else False
                results[rank_idx]["pb_valid"] = pb_valid

                for col in bool_cols:
                    if col not in results[rank_idx]:
                        results[rank_idx][col] = bool(pb_row[col])
        else:
            for result_row in results:
                result_row["pb_valid"] = None

    return results
```

يُشغّل PoseBusters مرة على ملف SDF المتنبأ به كله، ثم تقترن الصفوف بحسب الموضع مع الوضعيات المقيّمة. تكون **PB-Valid** صحيحة فقط إذا نجح كل اختبار منطقي من PoseBusters في الصف؛ وتُنسخ القيم المنطقية المنفردة أيضًا ما لم يستبدل اسمها مفتاح نتيجة موجودًا. يسجل خرج الفحص الفارغ صلاحية مجهولة باستخدام `None`، وهي متميزة عن الفشل الحقيقي، أي `False`.

### تقييم مجموعة بيانات وتلخيصها {#evaluating-a-dataset-and-summarizing}

```python
def evaluate_dataset(
    dataset_name: str,
    data_dir: Path,
    results_dir: Path,
    top_n: int,
    run_pb: bool,
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    """Run evaluation on a complete dataset."""

    csv_path = data_dir / f"{dataset_name}.csv"
    df = load_dataset_csv(csv_path)

    logger.info(f"Loaded {len(df)} complexes from {csv_path}")
    logger.info(f"Dataset: {dataset_name} | Top-N: {top_n} | PoseBusters: {run_pb}")

    all_results = []
    n_skipped = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i % 50 == 0 or i == len(df):
            logger.info(f"  Progress: {i}/{len(df)}")

        results = evaluate_complex(row, dataset_name, data_dir, results_dir, top_n, run_pb)
        if results:
            all_results.extend(results)
        else:
            n_skipped += 1
```

يمر المقيّم بصفوف pandas بترتيب CSV ويعرض التقدم كل 50. تُسطح `all_results` لأن المعقّد قد ينتج عدة صفوف وضعيات. ولا يُعد المعقّد متجاوزًا إلا إذا لم ينتج أي صف؛ أما وضعية RMSD بقيمة NaN فتظل مقيّمة.

```python
    if not all_results:
        logger.error("No results produced. Check your predicted poses and data paths.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    n_evaluated = results_df["complex_id"].nunique()
    logger.info(f"Evaluated {n_evaluated} complexes ({n_skipped} skipped)")
```

لا يعد إرجاع DataFrame فارغ فشلًا للعملية بذاته. وإلا تكوّن pandas اتحاد المفاتيح، فتتحول أعمدة PoseBusters الغائبة عن بعض الصفوف إلى قيم مفقودة.

```python
    top1 = results_df[results_df["pose_rank"] == 1]
    if not top1.empty:
        rmsd_lt2_pct = top1["rmsd_lt2"].mean() * 100
        rmsd_lt1_pct = top1["rmsd_lt1"].mean() * 100
        median_rmsd = top1["rmsd"].median()

        logger.info(f"Top-1 Results:")
        logger.info(f"  RMSD < 2.0 Å: {rmsd_lt2_pct:.1f}%")
        logger.info(f"  RMSD < 1.0 Å: {rmsd_lt1_pct:.1f}%")
        logger.info(f"  Median RMSD: {median_rmsd:.2f} Å")

        if "pb_valid" in top1.columns:
            pb_valid_pct = top1["pb_valid"].mean() * 100
            logger.info(f"  PB-Valid: {pb_valid_pct:.1f}%")
```

تعني top-1 أول سجل خرج حُمّل بنجاح. ومتوسط القيم المنطقية في pandas هو نسبة القيم الصحيحة، بينما يتجاهل الوسيط قيم NaN لـRMSD. تتحول `None` الخاصة بـPB إلى قيمة مفقودة وتُستبعد عادة من متوسطها، لذلك قد لا تُحسب الفحوص الفاشلة على أنها غير صالحة في المقام.

```python
    if top_n > 1 and not results_df.empty:
        best_poses = results_df.loc[results_df.groupby("complex_id")["rmsd"].idxmin()]
        best_lt2_pct = best_poses["rmsd_lt2"].mean() * 100
        best_lt1_pct = best_poses["rmsd_lt1"].mean() * 100

        logger.info(f"Best-of-{top_n} Results:")
        logger.info(f"  RMSD < 2.0 Å: {best_lt2_pct:.1f}%")
        logger.info(f"  RMSD < 1.0 Å: {best_lt1_pct:.1f}%")

        if "pb_valid" in best_poses.columns:
            best_pb_pct = best_poses["pb_valid"].mean() * 100
            logger.info(f"  PB-Valid: {best_pb_pct:.1f}%")
```

الأفضل من N مقياس وسيط مثالي؛ إذ يختار لكل معقّد الوضعية ذات أقل RMSD مرجعي، لا الأعلى ترتيبًا لدى النموذج. ويقيس ذلك وجود وضعية جيدة ضمن أول N. ثم تُعرض صلاحية PB للوضعية المختارة بحسب RMSD.

```python
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")

    return results_df
```

تُنشأ الأدلة الأم تكراريًا، ويُكتب جدول الوضعيات المسطح من دون عمود فهرس pandas. كما يجعل إرجاعه الدالة قابلة للاستخدام برمجيًا.

### واجهة سطر الأوامر ونقطة الدخول {#command-line-interface-and-entry-point}

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate docking predictions using CSV-based dataset mapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation.py --dataset proto_test
  python evaluation.py --dataset proto_test --no_pb_valid
  python evaluation.py --dataset posebusters_filtered --top_n 5

Expected file structure (from repository root):
  data/{dataset}.csv
  data/{dataset}/{ligand_file}
  data/{dataset}/{protein_file}
  results/{dataset}/{complex_id}_pred.sdf
        """,
    )

    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["proto_test", "proto_train", "posebusters_filtered", "full_test", "full_train"],
        help="Dataset name to evaluate"
    )
    parser.add_argument("--top_n", type=int, default=1,
                        help="Number of ranked poses to evaluate per complex (default: 1)")
    parser.add_argument("--no_pb_valid", action="store_true",
                        help="Skip PoseBusters physical validity checks (PB validation is enabled by default)")
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="Output CSV path (default: results/{dataset}_evaluation.csv)")

    return parser.parse_args()
```

يحافظ تنسيق الوصف الخام على مثال الاستخدام متعدد الأسطر. وتقيّد اختيارات مجموعة البيانات منطق الأدلة والمخطط. يجعل العلم السلبي PoseBusters مفعّلًا افتراضيًا، وتحول `type=Path` وسيط المسار هذا وحده تلقائيًا.

```python
def main():
    args = parse_args()

    data_dir = Path("data")
    results_dir = Path("results")

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    csv_path = data_dir / f"{args.dataset}.csv"
    if not csv_path.exists():
        logger.error(f"Dataset CSV not found: {csv_path}")
        sys.exit(1)

    dataset_dir = data_dir / args.dataset
    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    results_dataset_dir = results_dir / args.dataset
    if not results_dataset_dir.exists():
        logger.warning(f"Results directory does not exist: {results_dataset_dir}")
        logger.warning("Evaluation will skip all complexes without predicted poses.")
```

كل المسارات نسبية إلى دليل العمل الحالي، لذلك يجب تشغيل الأمر من جذر المستودع أو الجذر الاصطناعي في `evaluate.sh`. غياب بيانات الإدخال قاتل، أما غياب التنبؤات فتحذير فقط كي يبلغ منطق التجاوز لكل معقّد عن التغطية.

```python
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = results_dir / f"{args.dataset}_evaluation.csv"

    evaluate_dataset(
        dataset_name=args.dataset,
        data_dir=data_dir,
        results_dir=results_dir,
        top_n=args.top_n,
        run_pb=not args.no_pb_valid,
        output_csv=output_csv
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
```

يقع ملف CSV الافتراضي مباشرة تحت `results/`، منفصلًا عن أدلة الوضعيات لكل معقّد. وتحول `not args.no_pb_valid` العلم السلبي إلى وسيط الدالة الإيجابي. يمنع حارس الوحدة آثار الاستيراد الجانبية باستثناء إعداد التسجيل.

## محاذير وملاحظات {#gotchas--notes}

- تحاذي `GetBestRMS` الربيطة المتنبأ بها مع المرجع محاذاة مثلى. في الإرساء الأعمى قد يخفي ذلك أخطاء الانتقال والدوران نسبة إلى جيب البروتين، وينتج أرقامًا أكثر تفاؤلًا بكثير من RMSD في إطار ثابت.
- تستخدم النسب المجمعة المعقّدات التي أنتجت صفوف نتائج فقط. تُتجاوز التنبؤات والمراجع المفقودة بدل عدّها إخفاقات، لذلك يجب عرض التغطية مع نسب النجاح.
- يقرأ PoseBusters ملف SDF كاملًا، بينما يتوقف تحميل RMSD عند `top_n` ويسقط السجلات غير الصالحة. قد يسيء إقران صفوف PoseBusters بصفوف RMSD المضغوطة بحسب الموضع المحاذاة إذا كانت سجلات SDF المبكرة غير صالحة، كما يفترض حفظ PoseBusters لترتيب السجلات.
- لا يُتحقق من أن `top_n` موجب. تحمل القيمة الصفرية أو السالبة صفر وضعيات وتجعل كل معقّد متجاوزًا.
- إذا فشل PoseBusters وضبط `pb_valid=None`، تستبعد متوسطات pandas عادة تلك القيم المفقودة بدل عدها غير صالحة.
- يستخدم الأفضل من N الدالة `idxmin`؛ وقد تسبب المجموعات التي تكون قيم RMSD فيها كلها NaN اختيارًا إشكاليًا بحسب سلوك pandas.
- تسمي سلسلة توثيق الوحدة `updated_evaluation.py`، بينما المسار الفعلي هو `evaluation/evaluation.py`؛ وهذا أثر تسمية لا ملف ثانٍ.
