#!/usr/bin/env python3
"""
Merge multiple YOLO (v5→v11) datasets into one master dataset using a baseline data.yaml
located at the *parent* dataset folder (the --src-dir you pass in).

Directory layout example
------------------------
root_dataset/
  data.yaml            # ← baseline classes live here (authoritative)
  group1/
    images/{train,val,test}
    labels/{train,val,test}
  pre_train_dataset/
    images/...
    labels/...
  other_subdataset/
    train/images ...
    train/labels ...

Behavior
--------
- Reads baseline class list from <SRC_DIR>/data.yaml and **enforces it** in the merged dataset.
- Each sub-dataset is remapped to those baseline classes by NAME; annotations with classes not
  in the baseline are **omitted**.
- Supports two common layouts per sub-dataset:
  A) images/{train,val,test}, labels/{train,val,test}
  B) {train,val,test}/images, {train,val,test}/labels
  Also supports a "flat" dataset (images/, labels/ with files directly) → treated as "train".
- Resolves filename collisions; writes a fresh master data.yaml.

Usage
-----
python3 merge_data.py \
  --src-dir ./root_dataset \
  --out ./master_dataset \
  [--mode copy|move|link] [--drop-empty-labels] [--accept-missing]

Notes
-----
- Baseline names are taken from baseline data.yaml (block or inline list). Order defines final IDs.
- If a sub-dataset has no data.yaml (hence no names), its labels are kept **only** if a
  numeric id happens to map via another source of names (rare). By default we drop unknowns.
- YOLO-SEG text format is supported (first token is class id; rest are floats). Masks themselves are not modified.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
from typing import Dict, List, Optional, Set, Tuple

# tqdm is optional; falls back to no-op if unavailable
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------- YAML (tiny reader; avoids requiring PyYAML) ------------------

def read_yaml_min(p: Path) -> Optional[dict]:
    """Read minimal keys from YAML: nc, names (inline or block)."""
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    out = {}

    # nc: 19
    nc_m = re.search(r"^\s*nc\s*:\s*(\d+)\s*$", text, re.M)
    if nc_m:
        out["nc"] = int(nc_m.group(1))

    # names: [a, b, c]
    names_m = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, re.M)
    if names_m:
        raw = names_m.group(1)
        names = []
        for part in re.split(r",(?![^\[]*\])", raw):
            part = part.strip().strip("'\"")
            if part:
                names.append(part)
        out["names"] = names

    # block list:
    block = re.search(r"names\s*:\s*\n(\s*-\s*.*?)(\n\S|\Z)", text, re.S)
    if block and "names" not in out:
        names = []
        for line in block.group(1).splitlines():
            s = line.strip()
            if s.startswith("-"):
                names.append(s[1:].strip().strip("'\""))
        out["names"] = names

    return out or None


# ---------------- Discovery & pairing (layout A, B, or flat) ------------------

def discover_splits_and_layout(root: Path) -> Tuple[Set[str], str]:
    """
    Returns (splits, layout) where layout in {"A","B","flat",""}.
    A: images/{split}, labels/{split}
    B: {split}/images, {split}/labels
    flat: images/ and labels/ directly contain files (no subfolders)
    """
    splits: Set[str] = set()
    imgs = root / "images"
    labs = root / "labels"

    # Layout A or flat
    if imgs.exists() and labs.exists():
        a_splits = set()
        for p in imgs.glob("*/"):
            a_splits.add(p.name.rstrip("/"))
        for p in labs.glob("*/"):
            a_splits.add(p.name.rstrip("/"))
        if a_splits:
            return a_splits, "A"
        return {"."}, "flat"

    # Layout B
    for p in root.iterdir() if root.exists() else []:
        if p.is_dir() and (p / "images").exists() and (p / "labels").exists():
            splits.add(p.name)
    if splits:
        return splits, "B"

    return set(), ""


def match_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
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
        os.link(src, dst)  # hardlink
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------- Dataset collection + present class IDs ----------------------

def scan_present_class_ids(label_file: Path) -> Set[int]:
    ids: Set[int] = set()
    try:
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            tok = s.split()
            if not tok:
                continue
            try:
                ids.add(int(tok[0]))
            except ValueError:
                pass
    except Exception:
        pass
    return ids


def collect_dataset(root: Path) -> Dict:
    root = root.resolve()
    meta = read_yaml_min(root / "data.yaml") or {}
    names: Optional[List[str]] = meta.get("names")

    splits, layout = discover_splits_and_layout(root)
    if not splits:
        raise FileNotFoundError(
            f"No splits found under {root} as images/<split>&labels/<split>, "
            f"<split>/images&<split>/labels, or flat images/ & labels/."
        )

    splits_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    present_ids: Set[int] = set()

    if layout == "flat":
        pairs = match_pairs(root / "images", root / "labels")
        if pairs:
            splits_pairs["."] = pairs
            for _, lab in pairs:
                present_ids |= scan_present_class_ids(lab)

    elif layout == "A":
        for s in sorted(splits):
            pairs = match_pairs(root / "images" / s, root / "labels" / s)
            if pairs:
                splits_pairs[s] = pairs
                for _, lab in pairs:
                    present_ids |= scan_present_class_ids(lab)

    elif layout == "B":
        for s in sorted(splits):
            pairs = match_pairs(root / s / "images", root / s / "labels")
            if pairs:
                splits_pairs[s] = pairs
                for _, lab in pairs:
                    present_ids |= scan_present_class_ids(lab)

    if not splits_pairs:
        raise FileNotFoundError(f"No image/label pairs found in {root}.")

    return {"root": root, "splits": splits_pairs, "names": names, "present_ids": present_ids}


# ---------------- Baseline reconciliation ------------------------------------

def build_baseline(src_dir: Path) -> List[str]:
    """Load baseline class names from <src_dir>/data.yaml; order defines final IDs."""
    base = read_yaml_min(src_dir / "data.yaml")
    if not base or not base.get("names"):
        raise SystemExit(
            f"Baseline data.yaml with 'names' not found at {src_dir}/data.yaml. "
            f"Please create one, e.g.:\n\nnames:\n  - paper\n  - glass\n  - metal\n  - plastic\n"
        )
    return list(base["names"])  # copy


def per_dataset_remap(ds: Dict, baseline_names: List[str]) -> Dict[int, Optional[int]]:
    """Map old class id -> new class id using NAME alignment to the baseline.
    If ds has no names, we drop unknowns by default (cannot align).
    """
    remap: Dict[int, Optional[int]] = {}
    name_to_new = {n: i for i, n in enumerate(baseline_names)}

    if ds["names"]:
        max_id = max(ds["present_ids"] | {0}) if ds["present_ids"] else -1
        for old_id in range(max_id + 1):
            new_id = None
            if 0 <= old_id < len(ds["names"]):
                nm = ds["names"][old_id]
                new_id = name_to_new.get(nm, None)
            remap[old_id] = new_id
    else:
        # No names → drop everything (cannot align). Users can add a data.yaml to each sub-dataset.
        for old_id in ds["present_ids"]:
            remap[old_id] = None

    return remap


def rewrite_label_lines(label_text: str, remap: Dict[int, Optional[int]]) -> List[str]:
    """Rewrites first token (class id) using 'remap'. Keeps rest of the line as-is."""
    out: List[str] = []
    for line in label_text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        try:
            old = int(parts[0])
        except ValueError:
            continue
        new = remap.get(old, None)
        if new is None:
            continue
        parts[0] = str(new)
        out.append(" ".join(parts))
    return out


# ---------------- Merge -------------------------------------------------------

def safe_hash(s: str, n: int = 6) -> str:
    import hashlib
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def decide_name(dst_images_dir: Path, stem: str, suffix: str) -> str:
    candidate = stem
    i = 1
    while (dst_images_dir / f"{candidate}{suffix}").exists() or \
          (dst_images_dir.parent / "labels" / f"{candidate}.txt").exists():
        candidate = f"{stem}_{i}"
        i += 1
    return candidate


def merge(src_roots: List[Path], out_root: Path, baseline_names: List[str],
          mode: str = "copy", accept_missing: bool = False, keep_empty_labels: bool = True):
    out_root = out_root.resolve()
    ensure_dir(out_root)

    datasets = [collect_dataset(r) for r in src_roots]
    remaps = [per_dataset_remap(ds, baseline_names) for ds in datasets]

    # Collect all splits seen anywhere (e.g., train/val/test)
    all_splits: Set[str] = set()
    for ds in datasets:
        all_splits.update(ds["splits"].keys())

    report = {
        s: {"pairs": 0, "img_written": 0, "lab_written": 0, "skipped_pairs": 0,
            "dropped_anns": 0, "empty_labels": 0}
        for s in all_splits
    }

    for split in sorted(all_splits):
        # Map any "flat" split to "train"
        split_name = "train" if split == "." else split
        dst_img_dir = out_root / split_name / "images"
        dst_lab_dir = out_root / split_name / "labels"
        ensure_dir(dst_img_dir)
        ensure_dir(dst_lab_dir)

        for ds, remap in zip(datasets, remaps):
            pairs = ds["splits"].get(split, [])
            report[split]["pairs"] += len(pairs)
            for img, lab in tqdm(pairs, desc=f"{split_name}: {ds['root'].name}"):
                if not img.exists() or not lab.exists():
                    if accept_missing:
                        report[split]["skipped_pairs"] += 1
                        continue
                    raise FileNotFoundError(f"Missing pair: {img} / {lab}")

                stem, ext = img.stem, img.suffix.lower()
                if ext not in IMG_EXTS:
                    report[split]["skipped_pairs"] += 1
                    continue

                txt = lab.read_text(encoding="utf-8", errors="ignore")
                new_lines = rewrite_label_lines(txt, remap)
                orig_anns = sum(1 for _ in txt.splitlines() if _.strip())
                report[split]["dropped_anns"] += max(0, orig_anns - len(new_lines))

                # Optionally drop images whose labels become empty after filtering
                if not new_lines and not keep_empty_labels:
                    report[split]["empty_labels"] += 1
                    continue

                dst_img = dst_img_dir / f"{stem}{ext}"
                dst_lab = dst_lab_dir / f"{stem}.txt"

                # Resolve collisions across datasets
                if dst_img.exists() or dst_lab.exists():
                    h = safe_hash(str(img.resolve()))
                    new_stem = f"{stem}_{h}"
                    if (dst_img_dir / f"{new_stem}{ext}").exists() or (dst_lab_dir / f"{new_stem}.txt").exists():
                        new_stem = decide_name(dst_img_dir, stem, ext)
                    dst_img = dst_img_dir / f"{new_stem}{ext}"
                    dst_lab = dst_lab_dir / f"{new_stem}.txt"

                copy_like(img, dst_img, mode)
                # Write label (even if empty, when keep_empty_labels=True)
                dst_lab.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

                report[split]["img_written"] += 1
                report[split]["lab_written"] += 1
                if not new_lines:
                    report[split]["empty_labels"] += 1

    # Write master data.yaml from baseline
    data_yaml_path = out_root / "data.yaml"
    names_block = "\n".join(f"  - {n}" for n in baseline_names)
    nc_value = len(baseline_names)
    yaml = (
        f"# Auto-generated by merge_data.py\n"
        f"# Root: {out_root}\n\n"
        f"path: .\n"
        f"train: ../train/images\n"
        f"val: ../valid/images\n"
        f"test: ../test/images\n\n"
        f"nc: {nc_value}\n"
        f"names:\n{names_block}\n"
    )
    data_yaml_path.write_text(yaml, encoding="utf-8")

    # Console report
    print("\nMerge complete →", out_root)
    print("Baseline classes:", baseline_names)
    for s, r in sorted(report.items()):
        split_name = "train" if s == "." else s
        print(
            f"  {split_name}: pairs={r['pairs']} imgs={r['img_written']} labels={r['lab_written']} "
            f"skipped={r['skipped_pairs']} dropped_anns={r['dropped_anns']} empty_labels={r['empty_labels']}"
        )
    print(f"\nWrote: {data_yaml_path}")


# ---------------- CLI ---------------------------------------------------------

def find_subdatasets(parent: Path) -> List[Path]:
    """Enumerate subdirectories under 'parent' that look like YOLO datasets."""
    roots: List[Path] = []
    if not parent.exists():
        return roots
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        # Layout A or flat presence
        if (child / "images").exists() and (child / "labels").exists():
            roots.append(child)
            continue
        # Explicit hint
        if (child / "data.yaml").exists():
            roots.append(child)
            continue
        # Layout B check (train/images, etc.)
        has_b = False
        for p in child.iterdir():
            if p.is_dir() and (p / "images").exists() and (p / "labels").exists():
                has_b = True
                break
        if has_b:
            roots.append(child)
    return roots


def main():
    ap = argparse.ArgumentParser(description="Merge YOLO datasets using baseline classes from <src-dir>/data.yaml")
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="Parent directory that holds sub-datasets and baseline data.yaml")
    ap.add_argument("--out", type=Path, required=True, help="Output master dataset root")
    ap.add_argument("--mode", choices=["copy", "move", "link"], default="copy",
                    help="Copy, move, or hardlink files")
    ap.add_argument("--accept-missing", action="store_true",
                    help="Skip pairs with missing image/label instead of failing")
    ap.add_argument("--keep-empty-labels", dest="keep_empty_labels", action="store_true", default=True,
                    help="Keep images even if annotations were all dropped by remapping")
    ap.add_argument("--drop-empty-labels", dest="keep_empty_labels", action="store_false",
                    help="Drop images whose labels became empty after filtering")
    args = ap.parse_args()

    src_dir: Path = args.src_dir.resolve()
    baseline_names = build_baseline(src_dir)

    roots = [p for p in find_subdatasets(src_dir) if p.resolve() != src_dir]
    if len(roots) < 2:
        raise SystemExit(f"Found {len(roots)} sub-dataset(s) under {src_dir}; need at least 2.")

    merge(
        src_roots=roots,
        out_root=args.out,
        baseline_names=baseline_names,
        mode=args.mode,
        accept_missing=args.accept_missing,
        keep_empty_labels=args.keep_empty_labels,
    )


if __name__ == "__main__":
    main()
