#!/usr/bin/env python3
"""
Merge multiple Ultralytics/YOLO (v5→v11) datasets into one master dataset.

Features
- Point at a parent folder of many YOLO datasets: --src-dir /path/to/datasets
- Reads each sub-dataset's data.yaml to learn class names (if present)
- Harmonizes classes (intersection/union/base) and remaps label IDs accordingly
- Supports both layouts:
  A) images/{train,val,test}, labels/{train,val,test}
  B) {train,val,test}/images, {train,val,test}/labels
- Resolves filename collisions and writes a fresh data.yaml

Usage
------
# Folder-of-folders (recommended)
python3 merge_data.py --src-dir ./datasets --out ./master_dataset

# Fine-tune the class policy
python3 merge_data.py --src-dir ./datasets --out ./master_dataset --class-policy union

# Drop images that end up with no labels after class filtering
python3 merge_data.py --src-dir ./datasets --out ./master_dataset --drop-empty-labels
"""

from __future__ import annotations
import argparse, hashlib, os, re, shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------- YAML (minimal, no pyyaml dependency) ----------------

def read_yaml_data_yaml(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    out = {}
    nc_m = re.search(r"^\s*nc\s*:\s*(\d+)\s*$", text, re.M)
    if nc_m:
        out["nc"] = int(nc_m.group(1))

    # names: [ ... ] form
    names_m = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, re.M)
    if names_m:
        raw = names_m.group(1)
        names = []
        for part in re.split(r",(?![^\[]*\])", raw):
            part = part.strip().strip("'\"")
            if part:
                names.append(part)
        out["names"] = names

    # names: (block form)
    block = re.search(r"names\s*:\s*\n(\s*-\s*.*?)(\n\S|\Z)", text, re.S)
    if block and "names" not in out:
        names = []
        for line in block.group(1).splitlines():
            line = line.strip()
            if line.startswith("-"):
                names.append(line[1:].strip().strip("'\""))
        out["names"] = names

    return out or None

# ---------------- Discovery & pairing (two layouts) -------------------

def discover_splits(root: Path) -> Tuple[Set[str], str]:
    """
    Return (splits, layout) where layout in {"A","B","flat"}:
    A: images/{split}, labels/{split}
    B: {split}/images, {split}/labels
    flat: images/, labels/ with files directly inside
    """
    splits: Set[str] = set()
    imgs = root / "images"
    labs = root / "labels"

    # Layout A?
    if imgs.exists() and labs.exists():
        a_splits = set()
        for p in imgs.glob("*/"):
            a_splits.add(p.name.rstrip("/"))
        for p in labs.glob("*/"):
            a_splits.add(p.name.rstrip("/"))
        if a_splits:
            return a_splits, "A"
        # Flat?
        if imgs.exists() and labs.exists():
            return {"."}, "flat"

    # Layout B?
    for p in root.iterdir() if root.exists() else []:
        if p.is_dir() and (p/"images").exists() and (p/"labels").exists():
            splits.add(p.name)
    if splits:
        return splits, "B"

    return set(), ""  # none

def match_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    if not images_dir.exists() or not labels_dir.exists():
        return pairs
    for img in images_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            try:
                rel = img.relative_to(images_dir)
            except ValueError:
                rel = img.name
            lab = labels_dir / rel
            lab = lab.with_suffix(".txt")
            if lab.exists():
                pairs.append((img, lab))
    return pairs

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_like(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ---------------- Dataset collection + present class IDs --------------

def scan_present_class_ids(label_file: Path) -> Set[int]:
    ids: Set[int] = set()
    try:
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line: continue
            tok = line.split()
            if not tok: continue
            try:
                ids.add(int(tok[0]))
            except ValueError:
                pass
    except Exception:
        pass
    return ids

def collect_dataset(root: Path) -> Dict:
    root = root.resolve()
    data_yaml = read_yaml_data_yaml(root / "data.yaml") or {}
    names: Optional[List[str]] = data_yaml.get("names")
    nc = data_yaml.get("nc", len(names) if names else None)

    splits, layout = discover_splits(root)
    if not splits:
        raise FileNotFoundError(
            f"No splits found under {root} as images/<split> & labels/<split>, "
            f"<split>/images & <split>/labels, or flat images/ & labels/."
        )

    splits_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    present_ids_global: Set[int] = set()

    if layout == "flat":
        pairs = match_pairs(root/"images", root/"labels")
        splits_pairs["."] = pairs
        for _, lab in pairs:
            present_ids_global |= scan_present_class_ids(lab)
    elif layout == "A":
        for s in sorted(splits):
            pairs = match_pairs(root/"images"/s, root/"labels"/s)
            if pairs:
                splits_pairs[s] = pairs
                for _, lab in pairs:
                    present_ids_global |= scan_present_class_ids(lab)
    elif layout == "B":
        for s in sorted(splits):
            pairs = match_pairs(root/s/"images", root/s/"labels")
            if pairs:
                splits_pairs[s] = pairs
                for _, lab in pairs:
                    present_ids_global |= scan_present_class_ids(lab)

    if not splits_pairs:
        raise FileNotFoundError(f"No image/label pairs found in {root}.")

    return {
        "root": root,
        "layout": layout,
        "splits": splits_pairs,
        "data_yaml": data_yaml,
        "names": names,
        "nc": nc,
        "present_ids": present_ids_global,
    }

# ---------------- Class harmonization & remapping ---------------------

def build_final_class_list(datasets: List[Dict], policy: str) -> List[str]:
    """
    Returns sorted list of final class NAMES under policy:
      intersection: names present/used in all datasets (prefer names from data.yaml)
      union:        names present/used in any dataset
      base:         dataset #1's names exactly (or IDs if names missing)
    """
    def names_or_ids(ds: Dict) -> Set[str]:
        if ds["names"]:
            used = set()
            for cid in ds["present_ids"]:
                if 0 <= cid < len(ds["names"]):
                    used.add(ds["names"][cid])
            return used or set(ds["names"])
        return {str(i) for i in ds["present_ids"]}

    sets = [names_or_ids(ds) for ds in datasets]

    if policy == "base":
        base = datasets[0]
        final = names_or_ids(base)
    elif policy == "union":
        final = set()
        for s in sets: final |= s
    else:  # intersection
        final = sets[0].copy()
        for s in sets[1:]: final &= s

    return sorted(final)

def per_dataset_remap(ds: Dict, final_names: List[str]) -> Dict[int, Optional[int]]:
    """old id -> new id (or None to drop)"""
    name_to_new = {n: i for i, n in enumerate(final_names)}
    remap: Dict[int, Optional[int]] = {}
    if ds["names"]:
        max_id = max(ds["present_ids"] | {0})
        for old_id in range(max_id + 1):
            new = None
            if 0 <= old_id < len(ds["names"]):
                nm = ds["names"][old_id]
                new = name_to_new.get(nm, None)
            remap[old_id] = new
    else:
        str_to_new = {s: i for i, s in enumerate(final_names)}
        for old_id in ds["present_ids"]:
            remap[old_id] = str_to_new.get(str(old_id), None)
    return remap

def rewrite_label_lines(label_text: str, remap: Dict[int, Optional[int]]) -> List[str]:
    out: List[str] = []
    for line in label_text.splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        new_id = remap.get(old_id, None)
        if new_id is None:
            continue
        parts[0] = str(new_id)
        out.append(" ".join(parts))
    return out

# ---------------- Merge ----------------

def safe_hash(s: str, n: int = 6) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]

def decide_name(dst_images_dir: Path, stem: str, suffix: str) -> str:
    candidate = stem; i = 1
    while (dst_images_dir / f"{candidate}{suffix}").exists() or \
          (dst_images_dir.parent / "labels" / dst_images_dir.name / f"{candidate}.txt").exists():
        candidate = f"{stem}_{i}"; i += 1
    return candidate

def merge(
    src_roots: List[Path],
    out_root: Path,
    mode: str = "copy",
    accept_missing: bool = False,
    class_policy: str = "intersection",
    keep_empty_labels: bool = True,
):
    out_root = out_root.resolve()
    ensure_dir(out_root)

    # gather metadata & pairs
    datasets = [collect_dataset(r) for r in src_roots]

    # decide final classes
    final_names = build_final_class_list(datasets, class_policy)
    final_nc = len(final_names)
    remaps = [per_dataset_remap(ds, final_names) for ds in datasets]

    out_images = out_root / "images"
    out_labels = out_root / "labels"

    all_splits: Set[str] = set()
    for ds in datasets: all_splits.update(ds["splits"].keys())

    report = {s: {"pairs": 0, "img_written": 0, "lab_written": 0,
                  "skipped_pairs": 0, "dropped_anns": 0, "empty_labels": 0}
              for s in all_splits}

    for split in sorted(all_splits):
        dst_img_dir = out_images / (split if split != "." else "")
        dst_lab_dir = out_labels / (split if split != "." else "")
        ensure_dir(dst_img_dir); ensure_dir(dst_lab_dir)

        for ds, remap in zip(datasets, remaps):
            pairs = ds["splits"].get(split, [])
            report[split]["pairs"] += len(pairs)
            for img, lab in tqdm(pairs, desc=f"{split or 'root'}: {ds['root'].name}"):
                if not img.exists() or not lab.exists():
                    if accept_missing:
                        report[split]["skipped_pairs"] += 1; continue
                    raise FileNotFoundError(f"Missing pair: {img} / {lab}")

                stem = img.stem
                ext = img.suffix.lower()
                if ext not in IMG_EXTS:
                    report[split]["skipped_pairs"] += 1; continue

                # rewrite labels
                label_text = lab.read_text(encoding="utf-8", errors="ignore")
                new_lines = rewrite_label_lines(label_text, remap)
                orig_count = sum(1 for _ in label_text.splitlines() if _.strip())
                report[split]["dropped_anns"] += max(0, orig_count - len(new_lines))

                if not new_lines and not keep_empty_labels:
                    report[split]["empty_labels"] += 1
                    continue

                # collision-safe targets
                dst_img = dst_img_dir / f"{stem}{ext}"
                dst_lab = dst_lab_dir / f"{stem}.txt"
                if dst_img.exists() or dst_lab.exists():
                    h = safe_hash(str(img.resolve()))
                    new_stem = f"{stem}_{h}"
                    if (dst_img_dir / f"{new_stem}{ext}").exists() or (dst_lab_dir / f"{new_stem}.txt").exists():
                        new_stem = decide_name(dst_img_dir, stem, ext)
                    dst_img = dst_img_dir / f"{new_stem}{ext}"
                    dst_lab = dst_lab_dir / f"{new_stem}.txt"

                copy_like(img, dst_img, mode)
                dst_lab.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

                report[split]["img_written"] += 1
                report[split]["lab_written"] += 1
                if not new_lines:
                    report[split]["empty_labels"] += 1

    # write merged data.yaml
    data_yaml_path = out_root / "data.yaml"
    names_block = "\n".join([f"  - {n}" for n in final_names])
    yaml = (
        f"# Auto-generated by merge_data.py\n"
        f"path: .\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {final_nc}\n"
        f"names:\n{names_block}\n"
    )
    data_yaml_path.write_text(yaml, encoding="utf-8")

    print("\nMerge complete →", out_root)
    print("Final classes:", final_names)
    print("Splits report:")
    for s, r in sorted(report.items()):
        print(
            f"  {s or '.'}: pairs={r['pairs']}, images={r['img_written']}, "
            f"labels={r['lab_written']}, skipped_pairs={r['skipped_pairs']}, "
            f"dropped_anns={r['dropped_anns']}, empty_labels={r['empty_labels']}"
        )
    print(f"\nWrote: {data_yaml_path}")

# ---------------- CLI ----------------

def find_subdatasets(parent: Path) -> List[Path]:
    roots: List[Path] = []
    if not parent.exists(): return roots
    for child in parent.iterdir():
        if not child.is_dir(): continue
        if (child/"images").exists() and (child/"labels").exists():
            roots.append(child); continue
        if (child/"data.yaml").exists():
            roots.append(child); continue
        # layout B check
        has_b = False
        for p in child.iterdir():
            if p.is_dir() and (p/"images").exists() and (p/"labels").exists():
                has_b = True; break
        if has_b:
            roots.append(child)
    return roots

def main():
    ap = argparse.ArgumentParser(description="Merge YOLO datasets with class reconciliation")
    ap.add_argument("src", nargs="*", type=Path, help="Dataset roots (if not using --src-dir)")
    ap.add_argument("--src-dir", type=Path, help="Directory containing many YOLO datasets")
    ap.add_argument("--out", type=Path, required=True, help="Output master dataset root")
    ap.add_argument("--mode", choices=["copy","move","link"], default="copy", help="Copy, move or hardlink files")
    ap.add_argument("--accept-missing", action="store_true", help="Skip pairs with missing image/label")
    ap.add_argument("--class-policy", choices=["intersection","union","base"], default="intersection",
                    help="How to decide the final class list")
    ap.add_argument("--keep-empty-labels", dest="keep_empty_labels", action="store_true", default=True,
                    help="Keep images even if annotations were all dropped")
    ap.add_argument("--drop-empty-labels", dest="keep_empty_labels", action="store_false",
                    help="Drop images whose labels became empty after class filtering")
    args = ap.parse_args()

    if args.src_dir:
        roots = find_subdatasets(args.src_dir)
        if len(roots) < 2:
            raise SystemExit(f"Found {len(roots)} dataset(s) under {args.src_dir}; need at least 2.")
    else:
        roots = list(args.src)
        if len(roots) < 2:
            raise SystemExit("Provide at least two dataset roots, or use --src-dir.")

    merge(
        src_roots=roots,
        out_root=args.out,
        mode=args.mode,
        accept_missing=args.accept_missing,
        class_policy=args.class_policy,
        keep_empty_labels=args.keep_empty_labels,
    )

if __name__ == "__main__":
    main()
