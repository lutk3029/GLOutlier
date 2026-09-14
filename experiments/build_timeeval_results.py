#!/usr/bin/env python3
"""Rebuild Table 6, Table B7, and Figure B1 from per-dataset results.

The input is the long-form CSV committed at results/timeeval_per_dataset.csv.
Only the Python standard library is required.  Generated tables are CSV files and
the Bonferroni--Dunn control diagram is an SVG, so the results can be inspected
without LaTeX, pandas, scipy, or a plotting package.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SOURCE_ORDER = ["SMD", "SVDB", "MITDB", "Exathlon", "Daphnet"]
METHOD_ORDER = [
    "k-means", "OCSVM", "MAD-GAN", "OmniAnomaly", "USAD", "GDN", "AT",
    "TranAD", "FuSAGNet", "MTGFlow", "GCAD", "STAMP", "DAMP", "Ours",
]


def mean(values):
    return sum(values) / len(values)


def average_ranks_desc(values):
    """Return 1-based ranks for larger-is-better values, averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def normal_ppf(p):
    """Acklam's inverse-normal approximation (absolute error < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    a = [-3.969683028665376e1, 2.209460984245205e2,
         -2.759285104469687e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1,
         -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838, -2.549732539343734,
         4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996, 3.754408661907416]
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > high:
        return -normal_ppf(1.0 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def regularized_gamma_q(a, x):
    """Upper regularized incomplete gamma, used for the chi-square p-value."""
    if x < 0 or a <= 0:
        raise ValueError("invalid gamma arguments")
    if x == 0:
        return 1.0
    eps = 1e-14
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def load_results(path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    expected = 48 * len(METHOD_ORDER)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    scores = {}
    sources = {}
    for row in rows:
        key = (row["source"], row["dataset"])
        sources[key] = row["source"]
        pair = (key, row["method"])
        if pair in scores:
            raise ValueError(f"duplicate result: {pair}")
        scores[pair] = float(row["f1_pct"])
    datasets = list(dict.fromkeys((r["source"], r["dataset"]) for r in rows))
    if len(datasets) != 48:
        raise ValueError(f"expected 48 datasets, found {len(datasets)}")
    for dataset in datasets:
        missing = [m for m in METHOD_ORDER if (dataset, m) not in scores]
        if missing:
            raise ValueError(f"{dataset} is missing methods: {missing}")
    return datasets, sources, scores


def write_table6(path, datasets, sources, scores, exact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + [f"{s}_mean_f1_pct" for s in SOURCE_ORDER]
                        + ["all_48_mean_f1_pct"])
        for method in METHOD_ORDER:
            source_means = []
            for source in SOURCE_ORDER:
                values = [scores[(d, method)] for d in datasets if sources[d] == source]
                source_means.append(mean(values))
            overall = mean([scores[(d, method)] for d in datasets])
            source_format = ".6f" if exact else ".1f"
            overall_format = ".6f" if exact else ".2f"
            writer.writerow([method] + [format(v, source_format) for v in source_means]
                            + [format(overall, overall_format)])


def significance(datasets, scores):
    k, n = len(METHOD_ORDER), len(datasets)
    rank_rows = []
    tie_sum = 0.0
    for dataset in datasets:
        values = [scores[(dataset, m)] for m in METHOD_ORDER]
        rank_rows.append(average_ranks_desc(values))
        counts = defaultdict(int)
        for value in values:
            counts[value] += 1
        tie_sum += sum(t**3 - t for t in counts.values() if t > 1)
    avg_ranks = [mean([row[j] for row in rank_rows]) for j in range(k)]
    chi2 = (12.0 * n / (k * (k + 1.0))) * sum(r*r for r in avg_ranks) \
           - 3.0 * n * (k + 1.0)
    tie_correction = 1.0 - tie_sum / (n * (k**3 - k))
    chi2 /= tie_correction
    friedman_p = regularized_gamma_q((k - 1.0) / 2.0, chi2 / 2.0)

    control = METHOD_ORDER.index("Ours")
    se = math.sqrt(k * (k + 1.0) / (6.0 * n))
    tests = []
    for j, method in enumerate(METHOD_ORDER):
        if j == control:
            continue
        diff = avg_ranks[j] - avg_ranks[control]
        z = diff / se
        raw_p = math.erfc(abs(z) / math.sqrt(2.0))
        tests.append({"method": method, "diff": diff, "z": z, "raw": raw_p})
    ordered = sorted(tests, key=lambda row: row["raw"])
    running = 0.0
    for i, row in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - i) * row["raw"])
        running = max(running, adjusted)
        row["holm"] = running
    by_method = {row["method"]: row for row in tests}
    means = {m: mean([scores[(d, m)] for d in datasets]) for m in METHOD_ORDER}
    return avg_ranks, means, by_method, chi2, friedman_p, se


def write_table_b7(path, avg_ranks, means, tests):
    path.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(METHOD_ORDER)), key=lambda i: avg_ranks[i])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "average_rank", "mean_f1_pct",
                         "rank_difference_vs_ours", "z", "raw_p_value",
                         "holm_p_value", "significant_at_0.05"])
        for j in order:
            method = METHOD_ORDER[j]
            if method == "Ours":
                writer.writerow([method, f"{avg_ranks[j]:.2f}", f"{means[method]:.2f}",
                                 "", "", "", "", ""])
                continue
            result = tests[method]
            writer.writerow([method, f"{avg_ranks[j]:.2f}", f"{means[method]:.2f}",
                             f"{result['diff']:.2f}", f"{result['z']:.2f}",
                             f"{result['raw']:.8g}", f"{result['holm']:.8g}",
                             "Yes" if result["holm"] < 0.05 else "No"])


def write_cd_svg(path, avg_ranks, cd):
    width, height = 1200, 520
    left, right, axis_y = 105, 1095, 115
    x = lambda rank: left + (rank - 1.0) / 13.0 * (right - left)
    sorted_methods = sorted(zip(METHOD_ORDER, avg_ranks), key=lambda p: p[1])
    left_items, right_items = sorted_methods[:7], sorted_methods[7:]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111}.label{font-size:16px}.small{font-size:14px}.title{font-size:19px;font-weight:bold}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">Bonferroni–Dunn control diagram (α = 0.05, CD = {cd:.2f})</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}" stroke="#111" stroke-width="2"/>',
        f'<text x="{width/2}" y="75" text-anchor="middle" class="small">Average rank (lower is better)</text>',
    ]
    for rank in range(1, 15):
        xx = x(rank)
        lines += [f'<line x1="{xx:.1f}" y1="{axis_y-7}" x2="{xx:.1f}" y2="{axis_y+7}" stroke="#111"/>',
                  f'<text x="{xx:.1f}" y="{axis_y-14}" text-anchor="middle" class="small">{rank}</text>']
    cd_x1, cd_x2, cd_y = x(1), x(1 + cd), 55
    lines += [f'<line x1="{cd_x1:.1f}" y1="{cd_y}" x2="{cd_x2:.1f}" y2="{cd_y}" stroke="#b2182b" stroke-width="4"/>',
              f'<line x1="{cd_x1:.1f}" y1="{cd_y-7}" x2="{cd_x1:.1f}" y2="{cd_y+7}" stroke="#b2182b" stroke-width="3"/>',
              f'<line x1="{cd_x2:.1f}" y1="{cd_y-7}" x2="{cd_x2:.1f}" y2="{cd_y+7}" stroke="#b2182b" stroke-width="3"/>']
    for side, items in (("left", left_items), ("right", right_items)):
        for row, (method, rank) in enumerate(items):
            yy = 175 + row * 47
            xx = x(rank)
            if side == "left":
                text_x, elbow_x, anchor = 24, 82, "start"
            else:
                text_x, elbow_x, anchor = width - 24, width - 82, "end"
            lines += [
                f'<circle cx="{xx:.1f}" cy="{axis_y}" r="4" fill="#2166ac"/>',
                f'<polyline points="{xx:.1f},{axis_y+5} {xx:.1f},{yy} {elbow_x},{yy}" fill="none" stroke="#555"/>',
                f'<text x="{text_x}" y="{yy+5}" text-anchor="{anchor}" class="label">{method} ({rank:.2f})</text>',
            ]
    lines += [f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="small">Control: Ours. Statistical decisions in Table B7 use Holm’s step-down procedure.</text>',
              '</svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/timeeval_per_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    args = parser.parse_args()
    datasets, sources, scores = load_results(args.input)
    write_table6(args.output_dir / "table6.csv", datasets, sources, scores)
    write_table6(args.output_dir / "table6_exact.csv", datasets, sources, scores, exact=True)
    ranks, means, tests, chi2, p_value, se = significance(datasets, scores)
    write_table_b7(args.output_dir / "table_b7.csv", ranks, means, tests)
    # Bonferroni-Dunn for 13 two-sided control comparisons.
    cd = normal_ppf(1.0 - 0.05 / (2.0 * 13.0)) * se
    write_cd_svg(args.output_dir / "figure_b1_cd_diagram.svg", ranks, cd)
    ours_first = sum(
        scores[(dataset, "Ours")] == max(scores[(dataset, method)] for method in METHOD_ORDER)
        for dataset in datasets
    )
    ours_beats_at = sum(
        scores[(dataset, "Ours")] > scores[(dataset, "AT")] for dataset in datasets
    )
    summary = (f"datasets=48, methods=14\n"
               f"friedman_chi_square={chi2:.5f}\n"
               f"friedman_df=13\n"
               f"friedman_p_value={p_value:.8g}\n"
               f"rank_standard_error={se:.8f}\n"
               f"bonferroni_dunn_cd_alpha_0.05={cd:.5f}\n"
               f"ours_best_f1_dataset_count={ours_first}\n"
               f"ours_f1_greater_than_at_dataset_count={ours_beats_at}\n")
    (args.output_dir / "statistical_summary.txt").write_text(summary, encoding="utf-8")
    print(f"wrote Table 6, exact means, Table B7, Figure B1, and statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
