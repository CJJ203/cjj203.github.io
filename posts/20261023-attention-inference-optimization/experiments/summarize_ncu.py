"""Aggregate the saved NCU CSV metrics without adding per-kernel percentages."""
import csv
import io
import json
from pathlib import Path


summary = {}
for file in sorted(Path(__file__).parent.glob("ncu-*.csv")):
    rows = csv.DictReader(io.StringIO("\n".join(
        line for line in file.read_text().splitlines() if line.startswith('"'))))
    kernels = {}
    for row in rows:
        kernel = kernels.setdefault(row["ID"], {"kernel": row["Kernel Name"], "metrics": {}})
        kernel["metrics"][row["Metric Name"]] = float(row["Metric Value"].replace(",", ""))
    totals = {metric: sum(k["metrics"][metric] for k in kernels.values())
              for metric in ("gpu__time_duration.sum", "dram__bytes_read.sum",
                             "dram__bytes_write.sum")}
    summary[file.stem] = {"kernel_count": len(kernels), "totals": totals,
                          "kernels": list(kernels.values())}
print(json.dumps(summary, indent=2))
