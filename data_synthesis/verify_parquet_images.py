#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify that the image paths stored in a parquet file exist and are readable.

Usage:
    python3 verify_parquet_images.py <parquet_path> [--workers N] [--limit N]
                                     [--no-pil] [--sample-bytes N] [--out FILE]
                                     [--prune [--prune-all-bad] [--no-backup]]

Examples:
    # verify only
    python3 verify_parquet_images.py \
        /path/to/data/syn_positionqa.parquet --workers 64

    # drop rows whose images are missing, updating the parquet in place
    # (a .bak backup is kept by default)
    python3 verify_parquet_images.py \
        /path/to/data/syn_positionqa.parquet --workers 64 --prune

Run on the training machine only. Dependencies: pandas, pyarrow (parquet I/O).
PIL is optional: with `Pillow` installed, image decoding is verified as well,
which is stricter.

--prune behavior:
    By default only rows whose images are "missing" (path does not exist) are
    dropped. Add --prune-all-bad to also drop unusable rows (empty/unreadable/
    not_image). The update is in place: a temporary file is written and then
    atomically replaces the original. By default the original file is first
    copied to <parquet>.bak; pass --no-backup to skip the backup.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_parquet(parquet_path):
    try:
        import pandas as pd
    except ImportError as e:
        sys.exit(f"[FATAL] pandas is required: pip install pandas  ({e})")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        sys.exit(f"[FATAL] failed to read parquet: {e}")
    if "images" not in df.columns:
        sys.exit(f"[FATAL] parquet has no 'images' column, actual columns: {list(df.columns)}")
    return df


def row_image_paths(v):
    """Extract the list of paths from an images cell (accepts ndarray/list/str)."""
    if isinstance(v, str):
        return [v]
    try:
        return [p for p in list(v) if isinstance(p, str) and p]
    except TypeError:
        s = str(v)
        return [s] if s else []


def extract_paths(parquet_path):
    """Read the parquet, return (df, [(path, [row_idxs]), ...]). Paths are deduplicated for parallelism."""
    df = load_parquet(parquet_path)
    seen = {}
    for i, v in enumerate(df["images"].tolist()):
        for p in row_image_paths(v):
            seen.setdefault(p, []).append(i)
    return df, list(seen.items())


def try_pil():
    try:
        from PIL import Image  # noqa
        return Image
    except Exception:
        return None


def check_one(path, sample_bytes, use_pil, Image):
    """Return (path, status, detail). status in ok/missing/not_file/empty/unreadable/not_image."""
    if not os.path.exists(path):
        return path, "missing", "no such file or directory"
    if not os.path.isfile(path):
        return path, "not_file", "path is not a regular file"
    try:
        st = os.stat(path)
    except OSError as e:
        return path, "unreadable", f"stat failed: {e}"
    if st.st_size == 0:
        return path, "empty", "file size is 0"

    # Read the first few bytes first as a minimal readability check
    try:
        with open(path, "rb") as fh:
            head = fh.read(sample_bytes)
    except OSError as e:
        return path, "unreadable", f"read failed: {e}"
    if not head:
        return path, "empty", "read 0 bytes"

    if use_pil and Image is not None:
        try:
            with Image.open(path) as im:
                im.verify()  # header check only, pixels are not decoded
            # Decode for real as well, in case a corrupt image slips past verify
            with Image.open(path) as im:
                im.load()
        except Exception as e:
            return path, "not_image", f"PIL verify/load failed: {e}"
    return path, "ok", f"size={st.st_size}"


def main():
    ap = argparse.ArgumentParser(description="Verify parquet image paths")
    ap.add_argument("parquet", help="path to the parquet file")
    ap.add_argument("--workers", type=int, default=32, help="number of concurrent threads (default 32)")
    ap.add_argument("--limit", type=int, default=0, help="verify only the first N unique paths (0=all)")
    ap.add_argument("--sample-bytes", type=int, default=1024, help="bytes to read for the minimal readability check (default 1024)")
    ap.add_argument("--no-pil", action="store_true", help="disable PIL image-decoding checks")
    ap.add_argument("--out", default="", help="write unreadable paths to this file (one per line)")
    ap.add_argument("--prune", action="store_true",
                    help="drop rows with missing images and update the parquet in place (only 'missing' by default; add --prune-all-bad to drop all unusable rows)")
    ap.add_argument("--prune-all-bad", action="store_true",
                    help="with --prune: also drop empty/unreadable/not_image rows (not only missing)")
    ap.add_argument("--no-backup", action="store_true",
                    help="with --prune: do not keep a <parquet>.bak backup")
    args = ap.parse_args()
    if args.prune_all_bad and not args.prune:
        sys.exit("[FATAL] --prune-all-bad requires --prune")

    use_pil = not args.no_pil
    Image = try_pil() if use_pil else None
    if use_pil and Image is None:
        print("[WARN] Pillow is not installed; skipping PIL image-decoding checks (existence + prefix read only). Install with: pip install Pillow", file=sys.stderr)
        use_pil = False

    if not os.path.isfile(args.parquet):
        sys.exit(f"[FATAL] parquet does not exist: {args.parquet}")

    t0 = time.time()
    print(f"[*] reading {args.parquet} ...")
    df, path_rows = extract_paths(args.parquet)
    total_unique = len(path_rows)
    path_rows_dict = dict(path_rows)  # path -> [row_idxs]
    print(f"[*] rows: {len(df)}  unique paths: {total_unique}")
    if args.limit > 0:
        path_rows = path_rows[: args.limit]
        print(f"[*] limited to the first {len(path_rows)} unique paths")

    stats = {
        "ok": 0, "missing": 0, "not_file": 0,
        "empty": 0, "unreadable": 0, "not_image": 0,
    }
    bad = []  # (path, status, detail)
    n_done = 0
    n_total = len(path_rows)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(check_one, p, args.sample_bytes, use_pil, Image): p
            for p, _ in path_rows
        }
        for fut in as_completed(futs):
            path, status, detail = fut.result()
            stats[status] += 1
            n_done += 1
            if status != "ok":
                bad.append((path, status, detail))
            if n_done % 1000 == 0 or n_done == n_total:
                dt = time.time() - t0
                rate = n_done / dt if dt else 0
                print(
                    f"  [...] {n_done}/{n_total}  "
                    f"ok={stats['ok']} bad={sum(v for k, v in stats.items() if k != 'ok')}  "
                    f"{rate:.0f} path/s  {dt:.1f}s"
                )

    dt = time.time() - t0
    print()
    print("=" * 60)
    print(f"total unique paths : {n_total}")
    print(f"OK         : {stats['ok']}")
    print(f"missing    : {stats['missing']}")
    print(f"not_file   : {stats['not_file']}")
    print(f"empty      : {stats['empty']}")
    print(f"unreadable : {stats['unreadable']}")
    if use_pil:
        print(f"not_image  : {stats['not_image']}  (PIL decode failed)")
    print(f"elapsed    : {dt:.1f}s")
    print("=" * 60)

    if bad:
        bad.sort(key=lambda x: (x[1], x[0]))
        print(f"\nfirst {min(20, len(bad))} unreadable samples:")
        for path, status, detail in bad[:20]:
            print(f"  [{status}] {path}  ({detail})")
        if args.out:
            with open(args.out, "w") as fh:
                for path, status, detail in bad:
                    fh.write(f"{status}\t{path}\t{detail}\n")
            print(f"\nfull unreadable list written to: {args.out}")

    # ---- Drop missing rows and update in place ----
    if args.prune:
        if args.limit > 0:
            sys.exit("[FATAL] --prune and --limit cannot be combined (this would break the row-set integrity)")

        prune_statuses = {"missing"}
        if args.prune_all_bad:
            prune_statuses = {"missing", "not_file", "empty", "unreadable", "not_image"}

        bad_path_status = {p: s for p, s, _ in bad}
        bad_rows = set()
        for path, status in bad_path_status.items():
            if status in prune_statuses:
                for ri in path_rows_dict.get(path, []):
                    bad_rows.add(ri)
        # Safety net: also drop any row whose images cell is empty
        for i, v in enumerate(df["images"].tolist()):
            if not row_image_paths(v):
                bad_rows.add(i)

        keep_idx = [i for i in range(len(df)) if i not in bad_rows]
        n_before = len(df)
        n_drop = n_before - len(keep_idx)
        print(f"\n[PRUNE] dropping statuses: {sorted(prune_statuses)}")
        print(f"[PRUNE] rows to drop: {n_drop} / {n_before}  kept: {len(keep_idx)}")

        if n_drop == 0:
            print("[PRUNE] nothing to drop, parquet left unchanged.")
            sys.exit(0 if not bad else 1)

        df_keep = df.iloc[keep_idx].reset_index(drop=True)

        # Backup
        if not args.no_backup:
            bak = args.parquet + ".bak"
            print(f"[PRUNE] backing up original -> {bak}")
            try:
                import shutil
                shutil.copy2(args.parquet, bak)
            except OSError as e:
                sys.exit(f"[FATAL] backup failed, original file left unchanged: {e}")

        # Atomic write: write a temporary file, then rename over the original
        tmp = args.parquet + ".tmp." + str(os.getpid())
        try:
            df_keep.to_parquet(tmp, index=False)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            sys.exit(f"[FATAL] failed to write temporary file, original left unchanged: {e}")
        os.replace(tmp, args.parquet)
        print(f"[PRUNE] in-place update done: {args.parquet}")
        print(f"[PRUNE] rows {n_before} -> {len(df_keep)}  (dropped {n_drop})")
        sys.exit(0)
    else:
        sys.exit(1 if stats["ok"] != n_total else 0)


if __name__ == "__main__":
    main()