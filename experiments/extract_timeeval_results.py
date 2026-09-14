#!/usr/bin/env python3
"""Extract the per-dataset results (Tables A1-A6 of the paper appendix) from the
plain-text dump of the submitted PDF and write them to a CSV file.

This script documents the provenance of results/timeeval_per_dataset.csv:
every number in the CSV is copied verbatim from the per-dataset tables
(Tables A1-A6) of the paper "Unifying Global and Local Anomaly Detection for
Time Series". The aggregates reported in the paper (Table 6, Table B7 and
Figure B1) are re-derived from this CSV by build_timeeval_results.py.

Usage:
    pdftotext -layout "KAIS2025_..._v3_ submitted(1).pdf" paper.txt
    python3 experiments/extract_timeeval_results.py [paper.txt] [out.csv]

Only the Python 3 standard library is required.
"""

import csv
import re
import sys
from pathlib import Path

# Column order of Tables A1-A6 (left to right), identical for all six tables.
METHODS = [
    "k-means", "OCSVM", "MAD-GAN", "OmniAnomaly", "USAD", "GDN", "AT",
    "TranAD", "FuSAGNet", "MTGFlow", "GCAD", "STAMP", "DAMP", "Ours",
]

# Datasets in the exact order in which they appear in Tables A1-A6.
DATASETS = (
    # Table A1 + A2: SMD (23 server monitoring sequences)
    [("SMD", "machine-1-1"), ("SMD", "machine-1-2"), ("SMD", "machine-1-3"),
     ("SMD", "machine-1-5"), ("SMD", "machine-1-8"), ("SMD", "machine-2-1"),
     ("SMD", "machine-2-3"), ("SMD", "machine-2-4"), ("SMD", "machine-2-5"),
     ("SMD", "machine-2-6"), ("SMD", "machine-2-7"), ("SMD", "machine-2-8"),
     ("SMD", "machine-2-9"), ("SMD", "machine-3-1"), ("SMD", "machine-3-10"),
     ("SMD", "machine-3-11"), ("SMD", "machine-3-3"), ("SMD", "machine-3-4"),
     ("SMD", "machine-3-5"), ("SMD", "machine-3-6"), ("SMD", "machine-3-7"),
     ("SMD", "machine-3-8"), ("SMD", "machine-3-9")]
    # Table A3: Exathlon (2 Spark system log sequences)
    + [("Exathlon", "5_1_100000_63-64"), ("Exathlon", "5_1_100000_64-63")]
    # Table A4: SVDB (16 ECG records)
    + [("SVDB", r) for r in
       ("803 820 825 827 842 845 853 857 858 864 871 872 873 886 888 894").split()]
    # Table A5: MITDB (4 ECG records)
    + [("MITDB", r) for r in ("103 111 115 117").split()]
    # Table A6: Daphnet (3 Parkinson's gait records)
    + [("Daphnet", "S08R01E2"), ("Daphnet", "S09R01E0"),
       ("Daphnet", "S09R01E4")]
)

# A metric row looks like:  "  P(%) 61.8 61.2 ... 96.8"  (14 values),
# optionally preceded by dataset-name fragments (e.g. in the Exathlon table).
ROW_RE = re.compile(r"(P|R|F)\(%\)\s+((?:\d+(?:\.\d+)?\s+){13}\d+(?:\.\d+)?)\s*$")
TIME_RE = re.compile(r"T\(s\)\s+((?:\d+(?:\.\d+)?\s+){13}\d+(?:\.\d+)?)\s*$")


def parse_rows(text):
    """Return the list of (metric, [14 values]) rows in file order."""
    rows = []
    for line in text.splitlines():
        m = ROW_RE.search(line)
        if m:
            metric, blob = m.group(1), m.group(2)
        else:
            m = TIME_RE.search(line)
            if not m:
                continue
            metric, blob = "T", m.group(1)
        values = [float(v) for v in blob.split()]
        assert len(values) == len(METHODS), (metric, values)
        rows.append((metric, values))
    return rows


def main():
    paper_txt = sys.argv[1] if len(sys.argv) > 1 else "paper.txt"
    out_csv = (sys.argv[2] if len(sys.argv) > 2
               else "results/timeeval_per_dataset.csv")

    with open(paper_txt, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Only the appendix tables: from "Table A1" to "Appendix B".
    start = text.index("Table A1")
    end = text.index("Appendix B")
    rows = parse_rows(text[start:end])

    assert len(rows) == 4 * len(DATASETS), (
        "expected %d metric rows (4 x %d datasets), got %d"
        % (4 * len(DATASETS), len(DATASETS), len(rows)))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "dataset", "method",
                    "precision_pct", "recall_pct", "f1_pct", "time_s"])
        for i, (source, dataset) in enumerate(DATASETS):
            (p_m, p), (r_m, r), (f_m, f1), (t_m, t) = rows[4 * i:4 * i + 4]
            assert (p_m, r_m, f_m, t_m) == ("P", "R", "F", "T"), (
                dataset, p_m, r_m, f_m, t_m)
            for j, method in enumerate(METHODS):
                w.writerow([source, dataset, method,
                            f"{p[j]:g}", f"{r[j]:g}", f"{f1[j]:g}", f"{t[j]:g}"])

    print("wrote %d rows (%d datasets x %d methods) to %s"
          % (len(DATASETS) * len(METHODS), len(DATASETS), len(METHODS), out_csv))


if __name__ == "__main__":
    main()
