#!/usr/bin/env python3
"""
ollama_comparative_benchmark.py

Benchmark LLaMA-2 7B vs its Ollama-provided quantized variants,
compute normalized metrics & composite score, and generate comparative plots.
"""
import os, time, subprocess, psutil, argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker

# ─── CONFIG ────────────────────────────────────────────────────────────────

MODELS = [
    {"tag": "llama2:7b",          "desc": "LLaMA-2 7B FP16"},
    {"tag": "llama2:7b-text-q4_0", "desc": "LLaMA-2 7B Q4_0 (4-bit)"},
    {"tag": "llama2:7b-text-q8_0", "desc": "LLaMA-2 7B Q8_0 (8-bit)"},
]

PROMPT = """
You are a debugging assistant specializing in RAG systems.

Analyze this error and suggest a solution:

Traceback (most recent call last):
  File "rag_system.py", line 123, in retrieve_documents
    results = vector_db.similarity_search(query, k=3)
IndexError: Embedding dimension mismatch. Expected 768, got 384
"""

RESULT_DIR = "benchmark_results"
PLOTS_DIR  = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── BENCHMARK LOOP ─────────────────────────────────────────────────────────

def run_benchmark(models, repeat: int, show: bool):
    rows = []
    for m in models:
        tag  = m["tag"]
        desc = m["desc"]
        print(f"\n=== {tag} ({desc}) ===")
        # ensure model is present
        subprocess.run(["ollama", "pull", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for r in range(1, repeat+1):
            print(f" Run {r}/{repeat}…", end="", flush=True)
            proc = psutil.Process(os.getpid())
            mem0 = proc.memory_info().rss / 1e6

            tracker = EmissionsTracker(project_name=f"bench_{tag}", log_level="error")
            tracker.start()

            t0 = time.time()
            result = subprocess.run(
                ["ollama","run", tag, PROMPT],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="ignore"
            )
            t1 = time.time()
            carbon = tracker.stop()

            mem1 = proc.memory_info().rss / 1e6
            output = result.stdout.strip()
            tokens = len(output.split())

            # save raw output
            out_dir = os.path.join(RESULT_DIR, tag.replace(":", "_"))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"run_{r}.txt"), "w", encoding="utf-8") as f:
                f.write(output)

            if show:
                print("\n" + output + "\n" + "-"*40)
            else:
                print(" done")

            rows.append({
                "model": tag,
                "description": desc,
                "time_s": t1 - t0,
                "mem_mb": mem1 - mem0,
                "carbon_kg": carbon or 0.0,
                "token_rate": tokens / (t1 - t0) if (t1 - t0)>0 else 0.0
            })
    return pd.DataFrame(rows)

# ─── AGGREGATE & NORMALIZE ───────────────────────────────────────────────────

def aggregate_and_score(df: pd.DataFrame) -> pd.DataFrame:
    # average metrics per model
    avg = df.groupby(["model","description"], as_index=False).mean()

    # normalize each metric to 0-1
    for col in ["time_s","mem_mb","carbon_kg","token_rate"]:
        minv, maxv = avg[col].min(), avg[col].max()
        norm = (avg[col] - minv) / (maxv - minv) if maxv>minv else 0.0
        avg[f"norm_{col}"] = norm

    # invert metrics where lower is better
    for col in ["time_s","mem_mb","carbon_kg"]:
        avg[f"inv_norm_{col}"] = 1 - avg[f"norm_{col}"]

    # composite score weights
    avg["performance_score"] = (
        avg["inv_norm_time_s"]   * 0.3 +
        avg["inv_norm_mem_mb"]   * 0.3 +
        avg["inv_norm_carbon_kg"]* 0.3 +
        avg["norm_token_rate"]   * 0.1
    )

    # save CSV
    avg.to_csv(os.path.join(RESULT_DIR, "average_results_scored.csv"), index=False)
    return avg

# ─── PLOTTING ───────────────────────────────────────────────────────────────

def plot_comparisons(avg: pd.DataFrame):
    sns.set(style="whitegrid")

    # melt for comparative bar chart
    melt = avg.melt(
        id_vars=["description"],
        value_vars=["time_s","mem_mb","carbon_kg","token_rate"],
        var_name="metric", value_name="value"
    )
    # rename for readability
    metric_labels = {
        "time_s":   "Time (s)",
        "mem_mb":   "Memory Δ (MB)",
        "carbon_kg":"CO₂ (kg)",
        "token_rate":"Tokens/s"
    }
    melt["metric"] = melt["metric"].map(metric_labels)

    plt.figure(figsize=(10,5))
    ax = sns.barplot(x="metric", y="value", hue="description", data=melt, palette="mako")
    plt.title("Comparative Metrics by Model")
    plt.xlabel("")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparative_metrics.png"))
    plt.close()

    # radar chart
    metrics = ["inv_norm_time_s","inv_norm_mem_mb","inv_norm_carbon_kg","norm_token_rate"]
    labels  = ["Speed","Memory","Carbon","Throughput"]
    angles  = np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    for _, row in avg.iterrows():
        vals = [row[c] for c in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, label=row["description"])
        ax.fill(angles, vals, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "radar_comparison.png"))
    plt.close()

    # performance score plot
    plt.figure(figsize=(8,4))
    ax = sns.barplot(x="description", y="performance_score", data=avg, palette="mako")
    plt.title("Composite Performance Score (higher is better)")
    plt.xlabel(""); plt.ylabel("Score")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "performance_score.png"))
    plt.close()

# ─── MAIN ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ollama comparative benchmark")
    parser.add_argument("-r","--repeat", type=int, default=2, help="runs per model")
    parser.add_argument("--show", action="store_true", help="print outputs")
    args = parser.parse_args()

    raw_df = run_benchmark(MODELS, args.repeat, args.show)
    if raw_df.empty:
        print("No data collected. Check your model tags.")
        return

    raw_df.to_csv(os.path.join(RESULT_DIR,"raw_results.csv"), index=False)
    avg_df = aggregate_and_score(raw_df)
    plot_comparisons(avg_df)

    print("✅ Done. Results in `benchmark_results/` and plots in `benchmark_results/plots/`")

if __name__ == "__main__":
    main()
