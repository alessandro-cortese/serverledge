#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

REQ = {
    "formula_mode","reward_prior_weight","exploration_prior_weight","horizon",
    "target_count","macro_mean_latency_gain_pct","macro_latency_gain_ci95_low",
    "macro_latency_gain_ci95_high","macro_mean_wrong_choices_saved",
    "macro_mean_wrong_before_convergence_saved",
    "macro_mean_convergence_probability","macro_mean_convergence_requests_saved",
    "macro_first_arm_optimal_probability","macro_mean_reward_pseudo_regret"
}
GRID = (0.25, 0.50, 1.00)

def close(a,b): return math.isclose(a,b,rel_tol=0,abs_tol=1e-9)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--horizon",type=int,default=10)
    ap.add_argument("--selected-wr",type=float,default=0.25)
    ap.add_argument("--selected-we",type=float,default=1.0)
    ap.add_argument("--output-dir",required=True,type=Path)
    a=ap.parse_args()

    with a.input.open(newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f)
        missing=REQ-set(rd.fieldnames or [])
        if missing: raise RuntimeError("Missing columns: "+", ".join(sorted(missing)))
        src=list(rd)

    out=[]
    for r in src:
        if int(float(r["horizon"])) != a.horizon: continue
        wr=float(r["reward_prior_weight"]); we=float(r["exploration_prior_weight"])
        if not any(close(wr,x) for x in GRID) or not any(close(we,x) for x in GRID):
            continue
        out.append({
            "horizon":a.horizon,
            "formula_mode":r["formula_mode"],
            "w_R":wr,"w_E":we,
            "target_count":int(float(r["target_count"])),
            "latency_gain_pct":float(r["macro_mean_latency_gain_pct"]),
            "latency_gain_ci95_low":float(r["macro_latency_gain_ci95_low"]),
            "latency_gain_ci95_high":float(r["macro_latency_gain_ci95_high"]),
            "wrong_choices_saved":float(r["macro_mean_wrong_choices_saved"]),
            "wrong_before_convergence_saved":float(r["macro_mean_wrong_before_convergence_saved"]),
            "convergence_probability":float(r["macro_mean_convergence_probability"]),
            "convergence_requests_saved":float(r["macro_mean_convergence_requests_saved"]),
            "first_arm_optimal_probability":float(r["macro_first_arm_optimal_probability"]),
            "reward_pseudo_regret":float(r["macro_mean_reward_pseudo_regret"]),
            "selected_final_config": close(wr,a.selected_wr) and close(we,a.selected_we),
        })

    out.sort(key=lambda x:(x["w_R"],x["w_E"]))
    if len(out)!=9:
        raise RuntimeError(f"Expected 9 rows at H={a.horizon}, found {len(out)}")

    a.output_dir.mkdir(parents=True,exist_ok=True)
    cp=a.output_dir/f"transfer-weight-tuning-h{a.horizon}.csv"
    mp=a.output_dir/f"transfer-weight-tuning-h{a.horizon}.md"

    with cp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    lines=[
        f"# Transfer Learning weight tuning — H={a.horizon}","",
        f"Source artifact: `{a.input}`","",
        "| w_R | w_E | Latency gain | CI95 | Wrong choices saved | Convergence requests saved | P(convergence) | First arm optimal | Pseudo-regret | Final |",
        "|---:|---:|---:|:---|---:|---:|---:|---:|---:|:---:|"
    ]
    for r in out:
        lines.append(
            f"| {r['w_R']:.2f} | {r['w_E']:.2f} | {r['latency_gain_pct']:+.4f}% | "
            f"[{r['latency_gain_ci95_low']:+.4f}%, {r['latency_gain_ci95_high']:+.4f}%] | "
            f"{r['wrong_choices_saved']:+.4f} | {r['convergence_requests_saved']:+.4f} | "
            f"{r['convergence_probability']:.4f} | {r['first_arm_optimal_probability']:.4f} | "
            f"{r['reward_pseudo_regret']:.4f} | {'YES' if r['selected_final_config'] else ''} |"
        )
    lines += [
        "","## Interpretation","",
        "- `w_R`: equivalent-observation weight of the transferred reward prior.",
        "- `w_E`: pseudo-count used only in the UCB exploration term.",
        "- `H`: evaluation horizon, i.e. the first H target requests/decisions in the replay.",
        "- Lower pseudo-regret is better.",
        "- The final row is marked because it is the frozen configuration, not because one scalar metric defines a universal winner.",
    ]
    mp.write_text("\n".join(lines),encoding="utf-8")
    print(f"PASS csv={cp.resolve()} markdown={mp.resolve()}")

if __name__=="__main__":
    main()
