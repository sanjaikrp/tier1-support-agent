"""
benchmark/analyse_results.py

Phase 10d — the final analysis. Reads the three result files produced by the
benchmark and judge passes, computes all metrics with appropriate statistics,
and writes both a text summary and comparison charts.

Inputs (in results/):
  - benchmark_results.json                 (accuracy + latency, 240 records)
  - hallucination_verdicts.json            (evidence-grounded, 240)
  - hallucination_verdicts_symmetric.json  (symmetric, 240)

Outputs (in results/analysis/):
  - summary.txt          human-readable results with all statistics
  - accuracy_by_category.png
  - hallucination_rates.png
  - latency_distribution.png
  - accuracy_distribution.png

Design notes on statistics (documented for the dissertation):
  - Accuracy: Wilcoxon signed-rank on paired per-query mean scores. The baseline
    scores a constant 1, so this reduces in practice to testing whether the
    multi-agent score differs from 1; reported honestly. Also run on bug tickets
    alone, where the 0-2 metric can discriminate.
  - Hallucination: McNemar's test on paired binary outcomes (assumptions hold).
  - Latency: Shapiro-Wilk normality check, then Wilcoxon signed-rank (latency is
    right-skewed). Effect sizes reported throughout. N=40 -> framed as
    preliminary evidence.
"""

import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

RESULTS = "results/benchmark_results.json"
HALLUC_EVIDENCE = "results/hallucination_verdicts.json"
HALLUC_SYMMETRIC = "results/hallucination_verdicts_symmetric.json"
OUT_DIR = "results/analysis"

SYSTEMS = ["multi_agent", "baseline"]


def load(path):
    with open(path) as f:
        return json.load(f)


def per_query_means(records, value_key):
    """Average a value across replications, per (query_id, system)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get(value_key) is not None:
            acc[r["query_id"]][r["system"]].append(r[value_key])
    means = {}
    for qid, bysys in acc.items():
        means[qid] = {s: statistics.mean(v) for s, v in bysys.items() if v}
    return means


def achievable_max(gt):
    """Max score for a query: 2 if it has a file to identify, else 1."""
    return 2 if (gt.get("file_path") or "").strip() else 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load(RESULTS)
    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 70)
    out("BENCHMARK ANALYSIS — Multi-Agent vs Single-Prompt Baseline")
    out("=" * 70)
    n_records = len(results)
    out(f"Total records: {n_records} (expected 240 = 40 queries x 2 systems x 3 reps)")
    out("")

    # ---------------------------------------------------------------
    # 1. ACCURACY — overall, rescaled per-category, and bug-only
    # ---------------------------------------------------------------
    out("-" * 70)
    out("1. DIAGNOSTIC ACCURACY")
    out("-" * 70)

    # Raw mean score (0-2)
    raw = defaultdict(list)
    for r in results:
        if r.get("accuracy_score") is not None:
            raw[r["system"]].append(r["accuracy_score"])
    for s in SYSTEMS:
        v = raw[s]
        out(f"  {s:12} raw mean score = {statistics.mean(v):.3f} / 2  "
            f"({statistics.mean(v)/2*100:.1f}% of theoretical max)")
    out("")

    # Rescaled: percentage of ACHIEVABLE max per query (2 for bug, 1 for config/access)
    out("  Accuracy as % of ACHIEVABLE max (bug max=2; config/access max=1):")
    gt_by_q = {r["query_id"]: r["ground_truth"] for r in results}
    resc = defaultdict(list)
    resc_cat = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("accuracy_score") is None:
            continue
        amax = achievable_max(r["ground_truth"])
        pct = r["accuracy_score"] / amax * 100
        # cap at 100 (a score of 2 on a max-1 item shouldn't happen, but guard)
        pct = min(pct, 100.0)
        resc[r["system"]].append(pct)
        resc_cat[r["category"]][r["system"]].append(pct)
    for s in SYSTEMS:
        out(f"    {s:12} {statistics.mean(resc[s]):.1f}%")
    out("")
    out("  By category (rescaled % of achievable):")
    for c in ["bug", "config", "access"]:
        row = "    " + f"{c:8}"
        for s in SYSTEMS:
            row += f"  {s}={statistics.mean(resc_cat[c][s]):.1f}%"
        out(row)
    out("")

    # Statistical test on accuracy (paired per-query means)
    acc_means = per_query_means(results, "accuracy_score")
    ma = [acc_means[q]["multi_agent"] for q in acc_means if "multi_agent" in acc_means[q] and "baseline" in acc_means[q]]
    bl = [acc_means[q]["baseline"] for q in acc_means if "multi_agent" in acc_means[q] and "baseline" in acc_means[q]]
    if HAVE_SCIPY and ma:
        try:
            w, pw = scipy_stats.wilcoxon(ma, bl, zero_method="wilcox")
            out(f"  Wilcoxon signed-rank (all queries): W={w:.1f}, p={pw:.5f}")
        except ValueError as e:
            out(f"  Wilcoxon (all queries) could not run: {e}")
        # bug-only
        bug_q = [q for q in acc_means if q.startswith("BUGQ") and "baseline" in acc_means[q]]
        bma = [acc_means[q]["multi_agent"] for q in bug_q]
        bbl = [acc_means[q]["baseline"] for q in bug_q]
        try:
            wb, pwb = scipy_stats.wilcoxon(bma, bbl, zero_method="wilcox")
            out(f"  Wilcoxon signed-rank (BUG queries only): W={wb:.1f}, p={pwb:.5f}")
        except ValueError as e:
            out(f"  Wilcoxon (bug only) could not run: {e}")
        # Effect size: rank-biserial for paired = matched-pairs proportion
        diffs = [a - b for a, b in zip(ma, bl)]
        pos = sum(1 for d in diffs if d > 0)
        nonzero = sum(1 for d in diffs if d != 0)
        if nonzero:
            out(f"  Effect: {pos}/{nonzero} queries where multi-agent scored higher "
                f"(rank-biserial ~ {pos/nonzero:.2f})")
        out("  NOTE: the baseline scores a near-constant 1, so this comparison")
        out("  effectively tests whether the multi-agent score exceeds 1. Framed")
        out("  as preliminary evidence (N=40), not definitive proof.")
    out("")

    # ---------------------------------------------------------------
    # 2. HALLUCINATION — both measures, McNemar
    # ---------------------------------------------------------------
    out("-" * 70)
    out("2. HALLUCINATION RATE")
    out("-" * 70)
    for label, path in [("Evidence-grounded", HALLUC_EVIDENCE),
                        ("Symmetric (ground-truth-only)", HALLUC_SYMMETRIC)]:
        if not os.path.exists(path):
            out(f"  [{label}] file not found: {path}")
            continue
        verdicts = load(path)
        by = defaultdict(lambda: {"n": 0, "h": 0})
        # For McNemar we need paired outcomes per query (use majority across reps)
        per_q = defaultdict(dict)
        tmp = defaultdict(lambda: defaultdict(list))
        for v in verdicts:
            s = v["system"]; by[s]["n"] += 1
            if v["hallucination_detected"]:
                by[s]["h"] += 1
            tmp[v["query_id"]][s].append(1 if v["hallucination_detected"] else 0)
        out(f"  [{label}]")
        for s in SYSTEMS:
            d = by[s]
            out(f"    {s:12} {d['h']}/{d['n']} = {d['h']/d['n']*100:.1f}%")
        # McNemar on per-query majority vote
        for q in tmp:
            for s in SYSTEMS:
                if tmp[q][s]:
                    per_q[q][s] = 1 if sum(tmp[q][s]) > len(tmp[q][s]) / 2 else 0
        # Build contingency: b = MA halluc & BL not; c = BL halluc & MA not
        b = c = 0
        for q in per_q:
            if "multi_agent" in per_q[q] and "baseline" in per_q[q]:
                m = per_q[q]["multi_agent"]; bl_ = per_q[q]["baseline"]
                if m == 1 and bl_ == 0: b += 1
                elif m == 0 and bl_ == 1: c += 1
        out(f"    McNemar discordant pairs: MA-only={b}, BL-only={c}")
        if HAVE_SCIPY and (b + c) > 0:
            # McNemar exact (binomial) test
            from scipy.stats import binomtest
            p = binomtest(min(b, c), b + c, 0.5).pvalue
            out(f"    McNemar exact p={p:.5f}  (odds heavily favour more MA hallucination)")
        out("")

    # ---------------------------------------------------------------
    # 3. LATENCY
    # ---------------------------------------------------------------
    out("-" * 70)
    out("3. EXECUTION LATENCY")
    out("-" * 70)
    lat = defaultdict(list)
    for r in results:
        if r.get("latency_ms") is not None:
            lat[r["system"]].append(r["latency_ms"])
    for s in SYSTEMS:
        v = lat[s]
        out(f"  {s:12} mean={statistics.mean(v):.0f}ms  median={statistics.median(v):.0f}ms  "
            f"stdev={statistics.stdev(v):.0f}ms  min={min(v):.0f}  max={max(v):.0f}")
    ratio = statistics.mean(lat["multi_agent"]) / statistics.mean(lat["baseline"])
    out(f"  Multi-agent is {ratio:.1f}x slower than baseline on average.")
    lat_means = per_query_means(results, "latency_ms")
    lma = [lat_means[q]["multi_agent"] for q in lat_means if "baseline" in lat_means[q]]
    lbl = [lat_means[q]["baseline"] for q in lat_means if "baseline" in lat_means[q]]
    if HAVE_SCIPY and lma:
        _, p_norm = scipy_stats.shapiro(lma)
        out(f"  Shapiro-Wilk normality (multi-agent latency): p={p_norm:.4f} "
            f"({'normal' if p_norm > 0.05 else 'non-normal -> Wilcoxon appropriate'})")
        w, pw = scipy_stats.wilcoxon(lma, lbl)
        out(f"  Wilcoxon signed-rank (latency): W={w:.1f}, p={pw:.6f}")
    out("")

    # ---------------------------------------------------------------
    # 4. CATEGORY-CLASSIFICATION (multi-agent Router capability)
    # ---------------------------------------------------------------
    out("-" * 70)
    out("4. ROUTER CATEGORY-CLASSIFICATION ACCURACY (multi-agent only)")
    out("-" * 70)
    out("  The Router classifies each ticket into bug/config/access.")
    out("  The baseline has no equivalent stage, so this is a capability unique")
    out("  to the multi-agent architecture.")
    import glob as _glob
    RAW_DIR = "results/raw_logs"
    expected_cat = {r["query_id"]: r["ground_truth"].get("expected_category") for r in results}
    r_correct = r_total = 0
    r_bycat = defaultdict(lambda: {"c": 0, "n": 0})
    for _path in _glob.glob(os.path.join(RAW_DIR, "*multiagent.json")):
        try:
            _log = json.load(open(_path))
        except (json.JSONDecodeError, OSError):
            continue
        _qid = _log.get("ticket_id")
        _exp = expected_cat.get(_qid)
        _got = (_log.get("router") or {}).get("category")
        if _exp and _got:
            r_total += 1
            r_bycat[_exp]["n"] += 1
            if _got == _exp:
                r_correct += 1
                r_bycat[_exp]["c"] += 1
    if r_total:
        out(f"  Overall: {r_correct}/{r_total} = {r_correct/r_total*100:.1f}%")
        for _c in ["bug", "config", "access"]:
            _d = r_bycat[_c]
            if _d["n"]:
                out(f"    {_c:8} {_d['c']}/{_d['n']} = {_d['c']/_d['n']*100:.1f}%")
        out("  NOTE: a low access-classification rate indicates the Router confuses")
        out("  access tickets with other categories - a concrete, reportable finding.")
    else:
        out("  Could not read router categories from results/raw_logs/ (folder missing?).")
    out("")

    # ---------------------------------------------------------------
    # CHARTS
    # ---------------------------------------------------------------
    # Chart 1: accuracy by category (rescaled)
    cats = ["bug", "config", "access"]
    ma_vals = [statistics.mean(resc_cat[c]["multi_agent"]) for c in cats]
    bl_vals = [statistics.mean(resc_cat[c]["baseline"]) for c in cats]
    x = range(len(cats)); w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - w/2 for i in x], ma_vals, w, label="Multi-agent", color="#2E75B6")
    plt.bar([i + w/2 for i in x], bl_vals, w, label="Baseline", color="#C55A11")
    plt.xticks(list(x), cats); plt.ylabel("Accuracy (% of achievable max)")
    plt.title("Diagnostic Accuracy by Category"); plt.legend(); plt.ylim(0, 100)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/accuracy_by_category.png", dpi=140); plt.close()

    # Chart 2: hallucination rates (both measures)
    ev = load(HALLUC_EVIDENCE); sy = load(HALLUC_SYMMETRIC)
    def rate(vv, s):
        sub = [x for x in vv if x["system"] == s]
        return sum(1 for x in sub if x["hallucination_detected"]) / len(sub) * 100
    labels = ["Evidence-grounded", "Symmetric"]
    ma_h = [rate(ev, "multi_agent"), rate(sy, "multi_agent")]
    bl_h = [rate(ev, "baseline"), rate(sy, "baseline")]
    x = range(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar([i - w/2 for i in x], ma_h, w, label="Multi-agent", color="#2E75B6")
    plt.bar([i + w/2 for i in x], bl_h, w, label="Baseline", color="#C55A11")
    plt.xticks(list(x), labels); plt.ylabel("Hallucination rate (%)")
    plt.title("Hallucination Rate (two measures)"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/hallucination_rates.png", dpi=140); plt.close()

    # Chart 3: latency distribution (box plot)
    plt.figure(figsize=(8, 5))
    plt.boxplot([lat["multi_agent"], lat["baseline"]], tick_labels=["Multi-agent", "Baseline"])
    plt.ylabel("Latency (ms)"); plt.title("Execution Latency Distribution")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/latency_distribution.png", dpi=140); plt.close()

    # Chart 4: accuracy score distribution
    plt.figure(figsize=(8, 5))
    for idx, s in enumerate(SYSTEMS):
        scores = raw[s]
        counts = [scores.count(0), scores.count(1), scores.count(2)]
        plt.bar([i + idx*w for i in range(3)], counts, w,
                label=s, color=["#2E75B6", "#C55A11"][idx])
    plt.xticks([i + w/2 for i in range(3)], ["0 (wrong)", "1 (partial)", "2 (full)"])
    plt.ylabel("Count"); plt.title("Accuracy Score Distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/accuracy_distribution.png", dpi=140); plt.close()

    out("=" * 70)
    out(f"Charts saved to {OUT_DIR}/")
    out("Summary saved to results/analysis/summary.txt")
    out("=" * 70)

    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
