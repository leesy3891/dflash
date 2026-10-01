"""Reproduce the long-context/research figures from the cited records (CPU only).

Styling follows visualization_selective; inputs are explicitly pinned so legacy,
allocator-control and two-GPU records cannot silently enter the main comparison.
"""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash-long-context-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "visualization_selective"))
from _style import MODEL_COLORS, MODEL_MARKERS, HILITE, HILITE_EDGE

MODELS = ["qwen3-8b", "qwen3.5-9b", "qwen3.5-35b-a3b"]
SHORT = dict(zip(MODELS, ["8B", "9B", "35B-A3B"]))
COLORS = {**MODEL_COLORS, MODELS[2]: "#6a51a3"}
MARKERS = {**MODEL_MARKERS, MODELS[2]: "^"}
STAMPS = dict(zip(MODELS, ["113850", "103640", "023904"]))
CONTEXTS = [4096, 8192, 16384, 32768, 65536]
FOOT = ("B=1 · 3 × RTX A6000, layer sharding · greedy · 256 output tokens · causal_conv1d+fla\n"
        "One source document, three timing repeats. Hybrid results use stock rollback; lossless speedup is unverified.\n"
        "64K: 8B exceeds configured RoPE range; 35B drafter exceeds stated training context. No compute-bound claim.")
SOURCES = []


def read_json(path):
    SOURCES.append(str(path.relative_to(ROOT)))
    return json.loads(path.read_text())


def read_csv(path):
    SOURCES.append(str(path.relative_to(ROOT)))
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def record_path(model, stamp):
    return ROOT / "record_arch_main" / model / f"sweep_causal_conv1d+fla_3gpu_20260930-{stamp}.json"


def export(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load():
    records, main, blocks, components = {}, [], [], []
    for model in MODELS:
        path = record_path(model, STAMPS[model])
        rec = read_json(path)
        assert rec["measurement_protocol"]["version"] == 2
        assert rec["gpu_selection"]["num_gpus"] == 3
        records[model] = rec
        for c in rec["conditions"]:
            row = metrics(model, c, path)
            if c["sweep"] == "sequence":
                main.append(row)
            if c["input_tokens"] == 32768:
                blocks.append(row)
        components.extend(read_csv(path.parent / "csv" / path.stem / "component.csv"))
    path = record_path(MODELS[2], "091225")
    block_record = read_json(path)
    # Use one record for all three 35B block points; never mix its b16 with R35.
    blocks = [r for r in blocks if r["model"] != MODELS[2]]
    for c in block_record["conditions"]:
        if c["input_tokens"] == 32768:
            blocks.append(metrics(MODELS[2], c, path))
    assert len(main) == 15 and len(blocks) == 9
    assert len({(r["model"], r["context"], r["block"]) for r in main}) == 15
    export("performance_metrics.csv", main)
    export("block_metrics.csv", blocks)
    # Preserve the original units, moment and condition keys in this export.
    export("memory_components.csv", components)
    return records, main, blocks, components


def metrics(model, c, path):
    p = c["passes"]["perf"]["median"]
    ar = c["ar"]["native"]["median"]
    mem = c["passes"]["memory"]["median"]
    assert not p["profiler_used_before"] and not ar["profiler_used_before"]
    assert not ar["drafter_resident"]
    assert p["num_output_tokens"] == ar["num_output_tokens"] == 256
    row = dict(model=model, context=c["input_tokens"], block=c["block_size"],
               condition=c["key"], source=str(path.relative_to(ROOT)),
               speedup=c["speedup"]["native"]["tpot_speedup"],
               ar_ms=ar["time_per_output_token_s"] * 1000,
               df_ms=p["time_per_output_token_s"] * 1000,
               df_min_ms=p["time_per_output_token_s__min"] * 1000,
               df_max_ms=p["time_per_output_token_s__max"] * 1000,
               verify_ms=p["target_verify_mean_s"] * 1000,
               verify_over_ar=c["speedup"]["native"]["verify_over_ar_step"],
               acceptance=p["acceptance_rate"], committed=p["mean_committed_per_step"],
               break_even_committed=p["decode_latency_s"] / p["num_verify_steps"] / ar["time_per_output_token_s"],
               first_draft_ms=p["draft_forward_first_s"] * 1000,
               steady_draft_ms=p["draft_forward_steady_mean_s"] * 1000,
               first_draft_share=100 * p["draft_forward_first_s"] / p["decode_latency_s"],
               simultaneous_peak_gb=mem["peak_memory_simultaneous_bytes"] / 1e9,
               sum_device_peaks_gb=p["peak_memory_sum_device_maxima_bytes"] / 1e9,
               peak_site=mem["peak_site"]["operation"],
               native_overhead_gb=c["speedup"]["native"]["peak_memory_sum_device_maxima_bytes_overhead"] / 1e9,
               flags=";".join(c["flags"]))
    recording = c["speedup"].get("recording", {})
    row["recording_overhead_gb"] = (recording["peak_memory_sum_device_maxima_bytes_overhead"] / 1e9
                                            if recording.get("available") else None)
    phases = {"Verify": p["target_verify_total_s"], "First draft": p["draft_forward_first_s"],
              "Steady draft": p["draft_forward_total_s"] - p["draft_forward_first_s"],
              "Draft logits": p["draft_logits_total_s"],
              "Context feature": p["context_feature_decode_total_s"]}
    # Derive a closing remainder from the same median row. Independently reduced
    # per-field medians need not add exactly to the recorded unattributed median.
    remainder = p["decode_latency_s"] - sum(phases.values())
    assert remainder >= -1e-3
    phases["Remainder"] = max(0, remainder)
    row.update({"share_" + k: 100 * v / p["decode_latency_s"] for k, v in phases.items()})
    return row


def canvas(title, rows=2, cols=3, foot=FOOT):
    fig, axes = plt.subplots(rows, cols, figsize=(16 if cols == 3 else 12, 9.8), squeeze=False)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=.975)
    fig.subplots_adjust(left=.065, right=.97, top=.85, bottom=.17, hspace=.52, wspace=.3)
    fig.text(.5, .04, foot, ha="center", va="bottom", fontsize=9, linespacing=1.5,
             bbox=dict(facecolor=HILITE, edgecolor=HILITE_EDGE, boxstyle="round,pad=.65"))
    for i, ax in enumerate(axes.flat):
        ax.grid(axis="y", color="#e6e6e6", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    return fig, list(axes.flat)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", pad=12)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {name}.png / .pdf")


def context_axis(ax):
    ax.set_xticks(range(5), ["4K", "8K", "16K", "32K", "64K†"])
    ax.set_xlabel("Context length († range limits apply)")


def line(ax, data, key, dashed=False, labels=False):
    for m in MODELS:
        rows = sorted([r for r in data if r["model"] == m], key=lambda r: r["context"])
        ys = [r[key] for r in rows]
        ax.plot(range(len(rows)), ys, color=COLORS[m], marker=MARKERS[m],
                ls="--" if dashed else "-", lw=1.8, label=SHORT[m])
        if labels:
            for x, y in enumerate(ys):
                offset = {MODELS[0]: -14, MODELS[1]: 7, MODELS[2]: 17}[m] if key == "verify_over_ar" else (16 if m == MODELS[2] and x == 4 else 6)
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, offset), textcoords="offset points",
                            ha="center", fontsize=8, color=COLORS[m])
    context_axis(ax)


def model_legend(fig):
    fig.legend([Line2D([], [], color=COLORS[m], marker=MARKERS[m]) for m in MODELS],
               ["Qwen3-8B · full attention + dense", "Qwen3.5-9B · GDN + dense",
                "Qwen3.5-35B-A3B · GDN + MoE"], loc="upper center", bbox_to_anchor=(.5,.935),
               ncol=3, frameon=False, fontsize=10)


def overview(data):
    fig, a = canvas("Long-context DFlash: stock performance and the cost of verification")
    model_legend(fig)
    title(a[0], "A  Decode speedup vs native AR")
    line(a[0], data, "speedup", labels=True)
    a[0].axhline(1, color="#666", ls=":"); a[0].set_ylabel("AR TPOT / DFlash TPOT (×)"); a[0].set_ylim(0,6.4)
    title(a[1], "B  Per-token latency (min–max over 3 repeats)")
    for m in MODELS:
        rows = [r for r in data if r["model"] == m]
        a[1].fill_between(range(5), [r["df_min_ms"] for r in rows], [r["df_max_ms"] for r in rows],
                          color=COLORS[m], alpha=.2)
    line(a[1], data, "df_ms"); line(a[1], data, "ar_ms", dashed=True)
    a[1].set_yscale("log"); a[1].set_ylabel("ms / output token"); a[1].text(.03,.96,"Solid: DFlash · dashed: native AR",transform=a[1].transAxes,va="top",fontsize=9)
    title(a[2], "C  Verify cost grows with context")
    line(a[2], data, "verify_over_ar", labels=True); a[2].set_ylabel("Verify interval / AR token step (×)"); a[2].set_ylim(1,5)
    title(a[3], "D  Committed tokens vs break-even cost")
    line(a[3], data, "committed"); line(a[3], data, "break_even_committed", dashed=True)
    a[3].set_ylabel("tokens / verify step");a[3].text(.03,.97,"Solid: observed commit · dashed: cost threshold\nThreshold includes amortised first draft",transform=a[3].transAxes,va="top",fontsize=9)
    title(a[4], "E  First and steady draft forwards")
    line(a[4], data, "first_draft_ms"); line(a[4], data, "steady_draft_ms", dashed=True)
    a[4].set_yscale("log");a[4].set_ylabel("ms / call");a[4].text(.03,.96,"Solid: first · dashed: steady\nFirst draft occurs after TTFT closes",transform=a[4].transAxes,va="top",fontsize=9)
    title(a[5], "F  Decode composition at 4K and 64K")
    selected = [r for m in MODELS for s in (4096,65536) for r in data if r["model"]==m and r["context"]==s]
    phase_colors = ["#c0504d","#3b4d8f","#7fb3d5","#e8a33d","#6aa84f","#b0b0b0"]
    bottom = np.zeros(6)
    for phase, color in zip(["Verify","First draft","Steady draft","Draft logits","Context feature","Remainder"],phase_colors):
        vals = np.array([r["share_"+phase] for r in selected]);a[5].bar(range(6),vals,bottom=bottom,color=color,label=phase,width=.7);bottom+=vals
    a[5].set_xticks(range(6),[f"{SHORT[r['model']].replace('-A3B','')}\n{r['context']//1024}K" for r in selected]);a[5].set_ylabel("% of decode wall interval");a[5].set_ylim(0,106)
    for i,r in enumerate(selected):a[5].text(i, r["share_Verify"]/2,f"{r['share_Verify']:.1f}%",ha="center",color="white",fontsize=8)
    a[5].legend(ncol=3,fontsize=7,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.2))
    save(fig,"long_context_performance")


def component(rows, model, context, moment):
    found = [r for r in rows if r["model"]==model and int(r["input_tokens"])==context
             and int(r["block_size"])==16 and r["moment"]==moment]
    assert len(found)==1, (model,context,moment,len(found))
    return found[0]


def memory(data, comp):
    fig,a=canvas("Long-context memory: prompt recording dominates hybrid speculation state")
    model_legend(fig)
    title(a[0],"A  Attention KV vs prompt conv recording")
    title(a[1],"B  Prompt context feature persists at first draft")
    for m in MODELS:
        pre=[component(comp,m,s,"prefill_end") for s in CONTEXTS]
        a[0].plot(range(5),[float(r["target_attention_kv_bytes"])/1e9 for r in pre],color=COLORS[m],marker=MARKERS[m])
        a[0].plot(range(5),[float(r["target_gdn_conv_recording_bytes"])/1e9 for r in pre],color=COLORS[m],ls="--",marker=MARKERS[m])
        a[1].plot(range(5),[float(r["context_feature_bytes"])/1e9 for r in pre],color=COLORS[m],marker=MARKERS[m])
    for ax in a[:2]:context_axis(ax);ax.set_ylabel("GB (decimal)")
    a[0].text(.02,.98,"Solid: attention KV · dashed: conv recording",transform=a[0].transAxes,va="top",fontsize=9)
    title(a[2],"C  64K storage at three recorded moments")
    keys=[("target_attention_kv_bytes","Target attention KV","#c0504d"),
          ("target_gdn_state_bytes","GDN working state","#a6a6a6"),
          ("target_gdn_conv_recording_bytes","Conv recording","#b39ddb"),
          ("context_feature_bytes","Context feature","#6aa84f"),
          ("draft_kv_bytes","Draft KV","#7fb3d5")]
    states=[component(comp,m,65536,moment) for m in MODELS for moment in ("prefill_end","first_draft","steady_decode")]
    bottom=np.zeros(9)
    for k,label,color in keys:
        vals=np.array([float(r[k])/1e9 for r in states]);a[2].bar(range(9),vals,bottom=bottom,color=color,label=label);bottom+=vals
    a[2].set_xticks(range(9),[f"{SHORT[m].replace('-A3B','')}\n{moment}" for m in MODELS for moment in ("P","F","S")],fontsize=8)
    a[2].set_xlabel("P: prefill end · F: first draft · S: steady decode",fontsize=8)
    a[2].set_ylabel("GB; excludes weights and activations");a[2].legend(fontsize=7,frameon=False,ncol=2,loc="upper left")
    title(a[3],"D  Draft KV / target attention KV")
    for m in MODELS:
        for moment,ls in [("first_draft","-"),("steady_decode","--")]:
            ys=[float(component(comp,m,s,moment)["draft_kv_over_target_attention_kv"]) for s in CONTEXTS]
            a[3].plot(range(5),ys,color=COLORS[m],marker=MARKERS[m],ls=ls)
    context_axis(a[3]);a[3].set_ylabel("ratio (×)");a[3].set_ylim(0,1.45)
    a[3].text(.03,.97,"Solid: first · dashed: steady\nHybrid drafter retains one full-attention layer",transform=a[3].transAxes,va="top",fontsize=9)
    title(a[4],"E  64K peak definitions are different")
    rows=[r for m in MODELS for r in data if r["model"]==m and r["context"]==65536]
    for x,r in enumerate(rows):
        for offset,key,color in [(-.2,"simultaneous_peak_gb",COLORS[r["model"]]),(.2,"sum_device_peaks_gb","#bdbdbd")]:
            y=r[key];a[4].bar(x+offset,y,.38,color=color);a[4].text(x+offset,y+1,f"{y:.1f}",ha="center",fontsize=9)
    a[4].set_xticks(range(3),[SHORT[m] for m in MODELS]);a[4].set_ylabel("GB");a[4].set_ylim(0,160)
    a[4].legend(handles=[Patch(color="#555",label="Simultaneous peak (memory pass)"),Patch(color="#bdbdbd",label="Sum of device maxima (perf pass)")],frameon=False,fontsize=8,loc="upper left")
    title(a[5],"F  64K overhead depends on AR policy")
    for x,r in enumerate(rows):
        for offset,key,color in [(-.2,"native_overhead_gb",COLORS[r["model"]]),(.2,"recording_overhead_gb","#bdbdbd")]:
            y=r[key]
            if y is None:a[5].text(x+offset,1,"OOM",ha="center",fontsize=9);continue
            a[5].bar(x+offset,y,.38,color=color);a[5].text(x+offset,y+.5,f"{y:.2f}",ha="center",fontsize=9)
    a[5].set_xticks(range(3),[SHORT[m] for m in MODELS]);a[5].set_ylabel("DFlash − AR sum of device maxima, GB");a[5].set_ylim(0,42)
    a[5].legend(handles=[Patch(color="#555",label="Native AR"),Patch(color="#bdbdbd",label="Recording AR")],frameon=False,fontsize=8)
    save(fig,"long_context_memory")


def trace_rows(data):
    rows=[]
    for m in MODELS:
        for s in (4096,65536):
            for mode in ("dflash","ar"):
                phase="decode_target_verify" if mode=="dflash" else "decode_ar_step"
                path=ROOT/"record_arch_main"/m/"traces"/f"20260930-{STAMPS[m]}"/f"{m}_S{s}_b16_o256_shape_controlled_p0__{mode}.{phase}.breakdown.json"
                if not path.exists():
                    # Some AR endpoints were not exported; never invent a zero.
                    continue
                t=read_json(path);parts=t["per_occurrence_ms"]
                row=dict(model=m,context=s,mode=mode,source=str(path.relative_to(ROOT)),occurrences=t["occurrences"])
                row.update({k:sum(v.values()) for k,v in parts.items()})
                row["expert_gemm"]=parts.get("moe_experts",{}).get("gemm",0)
                row["expert_dispatch"]=row.get("moe_experts",0)-row["expert_gemm"]
                row["total_kernel_ms"]=sum(sum(v.values()) for v in parts.values())
                row["kernels"]=sum(sum(v.values()) for v in t["kernels_per_occurrence"].values())
                row["expert_loop_kernels"]=sum(t["kernels_per_occurrence"].get("moe_experts",{}).values())
                p=next(r for r in data if r["model"]==m and r["context"]==s)
                row["perf_interval_ms"]=p["verify_ms"] if mode=="dflash" else p["ar_ms"]
                row["cross_pass_ratio"]=row["total_kernel_ms"]/row["perf_interval_ms"]
                rows.append(row)
    export("trace_metrics.csv",rows)
    return rows


def routing():
    path=record_path(MODELS[2],"091225")
    raw=read_csv(path.parent/"csv"/path.stem/"moe_routing.csv")
    out=[]
    for block in (4,8,16):
        rows=[r for r in raw if r["path"]=="dflash" and r["phase"]=="decode: target verify"
              and f"/b{block}/" in r["condition"]]
        assert rows
        out.append(dict(block=block,layer_step_rows=len(rows),source=str((path.parent/"csv"/path.stem/"moe_routing.csv").relative_to(ROOT)),
                        unique_experts=np.mean([float(r["unique_experts_hit"]) for r in rows]),
                        tokens_per_hit_expert=np.mean([float(r["num_tokens"])*float(r["top_k"])/float(r["unique_experts_hit"]) for r in rows]),
                        load_imbalance=np.mean([float(r["load_imbalance"]) for r in rows])))
    export("moe_routing_metrics.csv",out)
    return out


def verification(trace,blocks,route):
    fig,a=canvas("Verification: residual attention growth and sparse expert execution")
    model_legend(fig)
    title(a[0],"A  35B verify GPU kernel breakdown")
    rows=[r for s in (4096,65536) for r in trace if r["model"]==MODELS[2] and r["mode"]=="dflash" and r["context"]==s]
    parts=[("full_attention","Full attention","#c0504d"),("expert_gemm","Expert GEMM","#6a51a3"),
           ("expert_dispatch","Expert dispatch","#bcbddc"),("moe_router","Router","#e8a33d"),
           ("moe_shared_expert","Shared expert","#6aa84f"),("gdn","GDN mixer","#7fb3d5"),
           ("lm_head","LM head","#3b4d8f")]
    bottom=np.zeros(2)
    for k,label,color in parts:
        vals=np.array([r.get(k,0) for r in rows]);a[0].bar(range(2),vals,bottom=bottom,color=color,label=label);bottom+=vals
    rest=np.array([r["total_kernel_ms"] for r in rows])-bottom
    a[0].bar(range(2),rest,bottom=bottom,color="#b0b0b0",label="Other modules")
    for i,r in enumerate(rows):a[0].text(i,r["total_kernel_ms"]+2,f"{r['total_kernel_ms']:.1f} ms",ha="center",fontsize=9)
    a[0].set_xticks(range(2),["4K","64K†"]);a[0].set_ylabel("GPU kernel ms / verify occurrence");a[0].set_ylim(0,170);a[0].legend(ncol=2,frameon=False,fontsize=7,loc="upper left")
    title(a[1],"B  Full-attention mixer vs GDN mixer")
    for m in MODELS:
        rows=[r for r in trace if r["model"]==m and r["mode"]=="dflash"]
        for key,ls in [("full_attention","-"),("gdn","--")]:
            if any(key in r for r in rows):a[1].plot([0,1],[r.get(key,0) for r in rows],color=COLORS[m],marker=MARKERS[m],ls=ls)
    a[1].set_xticks([0,1],["4K","64K†"]);a[1].set_ylabel("GPU kernel ms / verify occurrence");a[1].text(.02,.96,"Solid: full-attention mixer\nDashed: GDN mixer (no GDN in 8B)",transform=a[1].transAxes,va="top",fontsize=9)
    title(a[2],"C  35B GPU kernel sums and perf intervals")
    rows=[r for mode,s in [("ar",4096),("dflash",4096),("dflash",65536)] for r in trace if r["model"]==MODELS[2] and r["mode"]==mode and r["context"]==s]
    for i,r in enumerate(rows):
        a[2].bar(i-.18,r["total_kernel_ms"],.34,color="#6a51a3")
        a[2].bar(i+.18,r["perf_interval_ms"],.34,color="#bdbdbd")
        a[2].text(i,r["perf_interval_ms"]+5,f"{r['kernels']:,.0f} kernels",ha="center",fontsize=8)
    a[2].set_xticks(range(3),["AR 4K","Verify 4K","Verify 64K†"]);a[2].set_ylabel("ms / occurrence");a[2].set_ylim(0,330)
    a[2].legend(handles=[Patch(color="#6a51a3",label="Trace kernel sum"),Patch(color="#bdbdbd",label="Separate perf interval")],frameon=False,fontsize=8,loc="center left",bbox_to_anchor=(0,.72))
    a[2].text(.02,.98,"Difference is not measured CPU time\nRatio is not GPU utilization",transform=a[2].transAxes,va="top",fontsize=9)
    title(a[3],"D  32K block sweep: latency per output token")
    for m in MODELS:
        rows=sorted([r for r in blocks if r["model"]==m],key=lambda r:r["block"])
        ys=[r["df_ms"] for r in rows];a[3].plot(range(3),ys,color=COLORS[m],marker=MARKERS[m])
        for i,y in enumerate(ys):a[3].annotate(f"{y:.1f}",(i,y),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9,color=COLORS[m])
    a[3].set_xticks(range(3),["4","8","16"]);a[3].set_xlabel("Block width (includes anchor)");a[3].set_ylabel("Stock DFlash TPOT, ms");a[3].set_ylim(0,165)
    title(a[4],"E  35B: expert union broadens with block width")
    ys=[r["unique_experts"] for r in route];a[4].bar(range(3),ys,color="#6a51a3",width=.6)
    for i,y in enumerate(ys):a[4].text(i,y+1,f"{y:.2f}",ha="center")
    a[4].set_xticks(range(3),["4","8","16"]);a[4].set_xlabel("Block width at 32K");a[4].set_ylabel("Mean unique experts / layer × step");a[4].set_ylim(0,65)
    title(a[5],"F  Each hit expert still receives few tokens")
    a[5].plot(range(3),[r["tokens_per_hit_expert"] for r in route],color="#6a51a3",marker="^",label="Tokens / hit expert")
    a[5].plot(range(3),[r["load_imbalance"] for r in route],color="#e8a33d",marker="s",ls="--",label="Load imbalance (max / mean)")
    a[5].set_xticks(range(3),["4","8","16"]);a[5].set_xlabel("Block width at 32K");a[5].set_ylabel("Layer × step arithmetic mean");a[5].set_ylim(0,5)
    a[5].legend(frameon=False,fontsize=8);a[5].text(.03,.1,"Includes final short block; no compute\nsaturation or rejected-work estimate",transform=a[5].transAxes,fontsize=9)
    save(fig,"long_context_verification")


def correctness():
    legacy=[]
    for model,stamp in [(MODELS[1],"110754"),(MODELS[2],"120532")]:
        path=ROOT/"record_arch_main"/model/f"sweep_causal_conv1d+fla_20260929-{stamp}.json"
        for c in read_json(path)["conditions"]:
            lengths = (4096,65536) if model == MODELS[1] else (4096,8192)
            if c["input_tokens"] not in lengths or c["block_size"]!=16:continue
            l=c["lossless"]
            legacy.append(dict(model=model,context=c["input_tokens"],source=str(path.relative_to(ROOT)),
                               stock_acceptance=l["acceptance_rate_stock"],replay_acceptance=l["acceptance_rate_exact"],
                               stock_common_prefix=l["stock_vs_ar"]["common_prefix"],
                               replay_common_prefix=l["exact_vs_ar"]["common_prefix"],
                               state_error=l["audit"]["incremental"]["rejecting"]["state_rel_err_mean"]["median"]))
    export("legacy_correctness_metrics.csv",legacy)
    rollback=read_csv(ROOT/"record_arch_main/rollback/csv/rollback.csv")
    selected=[r for r in rollback if r["reference"]=="segmented" and r["mixer"]=="gdn"]
    export("rollback_metrics.csv",selected)
    foot=("Correctness evidence: RB9 (prompt 512, block 16) and separate 2026-09-29 legacy FLA audits L9/L35.\n"
          "Legacy acceptance is never combined with v2 timings. Replay-labelled paths do not guarantee AR agreement.\n"
          "The shadow audit is not an independent full-cache oracle; remaining numerical differences require controls.")
    fig,a=canvas("Rollback correctness: stock state and replay evidence",2,2,foot)
    title(a[0],"A  Rollback probe: recurrent state stays different")
    xs=[int(r["accepted"]) for r in selected];ys=[float(r["max_abs_diff"]) for r in selected]
    a[0].plot(xs,ys,color=COLORS[MODELS[1]],marker="s");a[0].set_xticks(xs);a[0].set_xlabel("Accepted inputs in isolated rollback test");a[0].set_ylabel("Max absolute GDN state difference")
    a[0].text(.03,.08,"At accepted=0: 24/24 recurrent states differ;\nconv and attention KV match the reference",transform=a[0].transAxes,fontsize=9)
    title(a[1],"B  Next-token probe logits also change")
    for x,r in zip(xs,selected):
        y=float(r["logits_max_abs_diff"]);a[1].bar(x,y,width=.8,color=COLORS[MODELS[1]])
        a[1].text(x,y+.12,"top-1 "+("same" if r["next_token_argmax_match"]=="True" else "differs"),ha="center",fontsize=8)
    a[1].set_xticks(xs);a[1].set_xlabel("Accepted inputs in isolated rollback test");a[1].set_ylabel("Max absolute probe logit difference");a[1].set_ylim(0,6.5)
    labels=[f"{SHORT[r['model']]}\n{r['context']//1024}K" for r in legacy]
    title(a[2],"C  Legacy acceptance changes in either direction")
    title(a[3],"D  Legacy common prefix with native AR")
    for i,r in enumerate(legacy):
        color=COLORS[r["model"]]
        for offset,prefix,alpha in [(-.18,"stock",1),(.18,"replay",.35)]:
            a[2].bar(i+offset,r[prefix+"_acceptance"],.34,color=color,alpha=alpha)
            a[3].bar(i+offset,r[prefix+"_common_prefix"],.34,color=color,alpha=alpha)
            a[2].text(i+offset,r[prefix+"_acceptance"]+.015,f"{r[prefix+'_acceptance']:.3f}",ha="center",fontsize=8)
            a[3].text(i+offset,r[prefix+"_common_prefix"]+4,str(r[prefix+"_common_prefix"]),ha="center",fontsize=8)
    for ax in a[2:]:
        ax.set_xticks(range(len(legacy)),labels);ax.legend(handles=[Patch(color="#555",label="Stock"),Patch(color="#bbb",label="Replay correction")],frameon=False,fontsize=8,loc="upper left")
    a[2].set_ylabel("Accepted proposals / all proposals");a[2].set_ylim(0,.8)
    a[3].set_ylabel("Matching output prefix, tokens (max 256)");a[3].set_ylim(0,320)
    save(fig,"long_context_correctness")


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":9,
                         "xtick.labelsize":9,"ytick.labelsize":9,"pdf.fonttype":42})
    records,data,blocks,components=load()
    overview(data)
    memory(data,components)
    trace=trace_rows(data)
    route=routing()
    verification(trace,blocks,route)
    correctness()
    (OUT/"sources.json").write_text(json.dumps({"records":sorted(set(SOURCES)),
        "reports":["LONG_CONTEXT_BOTTLENECK_REPORT.md","DFLASH_ARCHITECTURE_RESEARCH_REPORT.md"],
        "style_reference":"visualization_selective/_style.py"},indent=2)+"\n")


if __name__=="__main__":
    main()
