#!/usr/bin/env python3
"""
ollama_benchmark_clean.py

Benchmark LLaMA-2 7B vs its Ollama quantized variants using the Ollama HTTP API.
– Limits each response to --max-new-tokens via the API.
– Saves raw responses.
– Generates per-metric bar charts + a composite score.
"""
import os, time, subprocess, psutil, argparse, requests
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
API_URL = "http://localhost:11434/api/generate"
RESULT_DIR = "benchmark_results"
PLOTS_DIR   = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── BENCHMARK HELPER ──────────────────────────────────────────────────────

def generate_via_api(model: str, prompt: str, max_new_tokens: int) -> str:
    """Call Ollama API to generate a completion capped at max_new_tokens."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"max_new_tokens": max_new_tokens}
    }
    resp = requests.post(API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]  # final completion text


def benchmark_model(tag: str, desc: str, repeat: int, max_new_tokens: int):
    # 1) Pull locally
    subprocess.run(["ollama", "pull", tag], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 2) Warm-up (ignore output)
    subprocess.run(["ollama", "run", tag, PROMPT],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    records = []
    proc = psutil.Process(os.getpid())
    out_dir = os.path.join(RESULT_DIR, tag.replace(":", "_"))
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, repeat + 1):
        mem0 = proc.memory_info().rss / 1024**2
        tracker = EmissionsTracker(project_name=f"bench_{tag}", log_level="error")
        tracker.start()

        t0 = time.time()
        try:
            text = generate_via_api(tag, PROMPT, max_new_tokens)
        except Exception as e:
            print(f"[ERROR] API call failed for {tag} run {i}: {e}")
            continue
        t1 = time.time()
        carbon = tracker.stop()

        mem1 = proc.memory_info().rss / 1024**2
        tokens = len(text.split())

        # Save raw response
        path = os.path.join(out_dir, f"response_{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        records.append({
            "tag":        tag,
            "desc":       desc,
            "time_s":     t1 - t0,
            "mem_mb":     mem1 - mem0,
            "carbon_kg":  carbon or 0.0,
            "throughput": tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0
        })
        print(f"  Run {i}/{repeat}: {tokens} tokens in {t1-t0:.2f}s")

    return records


def run_all(repeat: int, max_new_tokens: int):
    rows = []
    for m in MODELS:
        print(f"\n▶ Benchmarking {m['tag']} ({m['desc']})")
        recs = benchmark_model(m["tag"], m["desc"], repeat, max_new_tokens)
        rows.extend(recs)
    return pd.DataFrame(rows)


# ─── AGGREGATE & SCORE ────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Drop fastest+slowest to reduce outliers when repeat>=3
    def trim(g):
        return (g
                .nsmallest(len(g)-2, 'time_s')
                .nlargest(len(g)-2, 'time_s')) if len(g)>2 else g

    df = df.groupby('tag', group_keys=False).apply(trim)
    avg = df.groupby(['tag','desc'], as_index=False).mean()

    # Normalize metrics
    for col in ('time_s','mem_mb','carbon_kg','throughput'):
        mn, mx = avg[col].min(), avg[col].max()
        avg[f"norm_{col}"] = (avg[col]-mn)/(mx-mn) if mx>mn else 0.0

    # Invert lower-is-better
    for col in ('time_s','mem_mb','carbon_kg'):
        avg[f"inv_norm_{col}"] = 1 - avg[f"norm_{col}"]

    # Composite performance score
    avg['perf_score'] = (
        avg['inv_norm_time_s']*0.3 +
        avg['inv_norm_mem_mb']*0.3 +
        avg['inv_norm_carbon_kg']*0.3 +
        avg['norm_throughput']*0.1
    )

    avg.to_csv(os.path.join(RESULT_DIR, "average_results.csv"), index=False)
    return avg


# ─── PLOTTING ─────────────────────────────────────────────────────────────

def bar_chart(avg: pd.DataFrame, col: str, ylabel: str, fname: str, fmt="{:.2f}"):
    plt.figure(figsize=(6,4))
    sns.barplot(x="desc", y=col, data=avg, palette="mako")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel); plt.xlabel("")
    for p in plt.gca().patches:
        plt.gca().annotate(
            fmt.format(p.get_height()),
            (p.get_x()+p.get_width()/2, p.get_height()),
            ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{fname}.png"))
    plt.close()

def plot_all(avg: pd.DataFrame):
    bar_chart(avg, "time_s",     "Inference Time (s)",    "time")
    bar_chart(avg, "mem_mb",     "Memory Δ (MB)",         "memory")
    bar_chart(avg, "carbon_kg",  "CO₂ Emissions (kg)",    "carbon")
    bar_chart(avg, "throughput","Tokens per Second",     "throughput")
    bar_chart(avg, "perf_score","Performance Score",     "performance_score", fmt="{:.3f}")

    # Radar chart
    metrics = ['inv_norm_time_s','inv_norm_mem_mb','inv_norm_carbon_kg','norm_throughput']
    labels  = ['Speed','Memory','Carbon','Throughput']
    angles  = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    for _, row in avg.iterrows():
        vals = [row[c] for c in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, label=row['desc'])
        ax.fill(angles, vals, alpha=0.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4,1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "radar.png"))
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama comparative benchmark")
    parser.add_argument("-r","--repeat",      type=int, default=5,   help="runs per model")
    parser.add_argument("--max-new-tokens",   type=int, default=50, help="API max_new_tokens")
    args = parser.parse_args()

    df = run_all(args.repeat, args.max_new_tokens)
    if df.empty:
        print("❌ No data collected; check your model tags.")
        exit(1)

    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULT_DIR, "raw_results.csv"), index=False)

    avg = aggregate(df)
    plot_all(avg)

    print(f"\n✅ Done!")
    print(f"• Raw responses: {RESULT_DIR}/<model_tag>/response_*.txt")
    print(f"• Metrics & plots: {RESULT_DIR}/ and {PLOTS_DIR}/")
