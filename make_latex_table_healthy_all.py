#!/usr/bin/env python
"""Build a LaTeX table from evaluation_results_healthy_all/summary_table.csv.

Default columns include core reconstruction, centerline, local cross-section,
physiology, and distributional metrics:
pos_mae, centerline_cd, cross_section_2d_chamfer, cross_section_area_error,
cross_section_circularity_error, |r-1|, Murray, BifAngle, Taper, MMD,
MMD_radius, MMD_edge_length, MMD_tortuosity, COV, 1-NNA, Diversity.

GT row is shown at the bottom. Optional literature/SOTA rows can be appended
from a sidecar CSV with partial values.
"""
from __future__ import annotations
import csv, sys, os, argparse, math

PRETTY = {
    "FM v11b healthy":               "FM v11b",
    "FM v12 abspos healthy":         "FM v12 (abs)",
    "Physio v5 healthy":             "Physio v5",
    "Physio v6 healthy":             "Physio v6",
    "Physio v8 abspos healthy":      "Physio v8 (abs)",
    "TreeGNN v1 healthy":            "TreeGNN v1",
    "TreeGNN v2 abspos healthy":     "TreeGNN v2 (abs)",
    "FM-TwoStage v2 healthy":        "FM-TwoStage v2",
    "FM-TwoStage v3 abspos healthy": "FM-TwoStage v3 (abs)",
    "TwoStageFM v1 healthy":         "TwoStageFM v1",
    "AR FM v2 healthy":              "AR-FM v2",
    "AR FM v3 abspos healthy":       "AR-FM v3 (abs)",
    "HierarchicalFM v1 healthy":     "HierFM v1",
    "WFM v1 healthy":                "WFM v1",
    "Latent v1 healthy":             "Latent v1",
    "AneuCond v3 abspos healthy":    "AneuCond v3 (abs)",
    "Branch v2 abspos healthy":      "Branch v2 (abs)",
    "HierFM-Strahler v1":              "HierFM-S v1",
    "HierFM-Strahler v2":              "HierFM-S v2",
    "HierFM-Strahler v3 bifdepth+rel": "HierFM-S v3 (bd+rel)",
    "HierFM-Strahler v4 bifdepth+abs": "HierFM-S v4 (bd+abs)",
    "HierFM-Strahler v5 flat+rel":     "HierFM-S v5 (flat+rel)",
    "HierFM-Strahler v6 flat+abs":     "HierFM-S v6 (flat+abs)",
    "HierFM-Strahler v7 flat+abs+nosc":"HierFM-S v7 (flat+abs, no-SC)",
    "HierFM-Strahler v1 compat genfix": "HierFM-S v1 compat",
    "GT (validation)":                "\\textit{GT (val.)}",
}

# (csv_col, header, fmt, score_fn) — score_fn: smaller better
COLS = [
    ("pos_mae",        r"pos MAE $\downarrow$",       "{:.3f}", lambda v, gt: v),
    ("centerline_cd",  r"CD$_\mathrm{ctr}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("cross_section_2d_chamfer", r"CD$_\mathrm{xsec}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("cross_section_2d_hausdorff", r"HD$_\mathrm{xsec}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("cross_section_area_error", r"Area$_\mathrm{xsec}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("cross_section_perimeter_error", r"Perim$_\mathrm{xsec}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("cross_section_circularity_error", r"Circ$_\mathrm{xsec}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("cross_section_eccentricity_error", r"Ecc$_\mathrm{xsec}$ $\downarrow$", "{:.3f}", lambda v, gt: v),
    ("|radius-1|",     r"$|r-1|$ $\downarrow$",       "{:.3f}", lambda v, gt: v),
    ("murray_viol",    r"Murray $\to$ GT",            "{:.3f}", lambda v, gt: abs(v - gt["murray_viol"])),
    ("bif_angle_deg",  r"BifAng [$^\circ$] $\to$ GT", "{:.3f}", lambda v, gt: abs(v - gt["bif_angle_deg"])),
    ("tapering_viol",  r"Taper $\downarrow$",         "{:.3f}", lambda v, gt: v),
    ("MMD",            r"MMD $\downarrow$",           "{:.3f}", lambda v, gt: v),
    ("MMD_radius",     r"MMD$_r$ $\downarrow$",       "{:.3f}", lambda v, gt: v),
    ("MMD_edge_length",r"MMD$_\ell$ $\downarrow$",    "{:.3f}", lambda v, gt: v),
    ("MMD_tortuosity", r"MMD$_\tau$ $\downarrow$",    "{:.3f}", lambda v, gt: v),
    ("COV",            r"COV $\uparrow$",             "{:.3f}", lambda v, gt: -v),
    ("1-NNA",          r"1-NNA $\to .5$",             "{:.3f}", lambda v, gt: abs(v - 0.5)),
    ("diversity",      r"Div. $\to$ GT",              "{:.3f}", lambda v, gt: abs(v - gt["diversity"])),
]


def parse_float_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def load_optional_sota_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        row.setdefault("model", raw.get("label", "SOTA"))
        row.setdefault("label", row["model"])
        row["rankable"] = str(raw.get("rankable", "0")).strip() == "1"
        rows.append(row)
    return rows


def has_visible_values(row):
    for csv_col, _, _, _ in COLS:
        if parse_float_or_none(row.get(csv_col)) is not None:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation_results_healthy_all/summary_table.csv")
    ap.add_argument("--out", default="evaluation_results_healthy_all/summary_table.tex")
    ap.add_argument("--sota_csv", default="evaluation_results_healthy_all/sota_reference_rows.csv")
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    sota_rows = [r for r in load_optional_sota_rows(args.sota_csv) if has_visible_values(r)]

    gt = next(r for r in rows if r["model"] == "GT (validation)")
    gt_vals = {c: float(gt[c]) for c, *_ in COLS}
    models = [r for r in rows if r["model"] != "GT (validation)"]
    rankable_rows = models + [r for r in sota_rows if r.get("rankable", False)]

    # ranking per column
    ranks = {}
    for csv_col, _, _, score in COLS:
        scored = []
        for i, r in enumerate(rankable_rows):
            value = parse_float_or_none(r.get(csv_col))
            if value is None:
                continue
            scored.append((score(value, gt_vals), i))
        scored.sort()
        order = [i for _, i in scored]
        ranks[csv_col] = (order[0], order[1] if len(order) > 1 else None) if order else (None, None)

    def cell(val, csv_col, fmt, idx):
        if val is None:
            return "--"
        s = fmt.format(val)
        best, second = ranks[csv_col]
        if idx is not None and idx == best:
            return r"\textbf{" + s + "}"
        if idx is not None and idx == second:
            return r"\underline{" + s + "}"
        return s

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Healthy-vessel test split ($N=57$). "
                 r"Best per column \textbf{bold}, second \underline{underlined}. "
                 r"Murray, BifAngle and Diversity report distance-to-GT. "
                 r"Optional literature rows are shown without affecting ranking unless explicitly enabled.}")
    lines.append(r"\label{tab:healthy_all}")
    lines.append(r"\setlength{\tabcolsep}{2.5pt}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l" + "c" * len(COLS) + "}")
    lines.append(r"\toprule")
    lines.append("Model & " + " & ".join(h for _, h, _, _ in COLS) + r" \\")
    lines.append(r"\midrule")
    for i, r in enumerate(models):
        name = PRETTY.get(r["model"], r["model"])
        cells = [cell(parse_float_or_none(r.get(c)), c, fmt, i) for c, _, fmt, _ in COLS]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    if sota_rows:
        lines.append(r"\midrule")
        for r in sota_rows:
            name = r.get("label", r.get("model", "SOTA"))
            idx = None
            if r.get("rankable", False):
                idx = len(models) + [x for x in sota_rows if x.get("rankable", False)].index(r)
            cells = [cell(parse_float_or_none(r.get(c)), c, fmt, idx) for c, _, fmt, _ in COLS]
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    gt_cells = [fmt.format(float(gt[c])) for c, _, fmt, _ in COLS]
    lines.append(PRETTY["GT (validation)"] + " & " + " & ".join(gt_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out)
    print(out)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
