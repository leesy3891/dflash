"""All-context phase memory, wall latency and trace execution breakdowns.

Only recorded measurements are used. Native AR has no phase-memory probe and
the arch records discard draft stage timers: those omissions are explicit.
"""
from plot_results import (
    ROOT, OUT, MODELS, SHORT, COLORS, STAMPS, CONTEXTS, FOOT,
    record_path, read_json, export, save, plt, np, Patch, Line2D,
)

PHASES = [
    ("prefill: target forward", "P1 Target fwd"),
    ("prefill: context-feature build", "P2 Feature build"),
    ("prefill: first token", "P3 First token"),
    ("prefill: context-feature concat", "P4 Feature concat"),
    ("prefill: rollback/crop", "P5 Crop"),
    ("decode: first draft forward", "D1 First draft"),
    ("decode: draft forward", "D1s Steady draft"),
    ("decode: draft rollback/crop", "D2 Draft crop"),
    ("decode: draft logits", "D3 Draft logits"),
    ("decode: target verify", "D4 Verify"),
    ("decode: verify rollback/crop", "D5 Verify crop"),
    ("decode: context-feature build", "D6 Feature build"),
]
MEMORY = [
    ("target_weights_gb", "Target weights", "#8c8c8c"),
    ("target_attention_kv_gb", "Target attention KV", "#c0504d"),
    ("gdn_recurrent_gb", "GDN recurrent", "#de9c9c"),
    ("gdn_conv_working_gb", "GDN conv working", "#e8a33d"),
    ("gdn_conv_recording_gb", "GDN conv recording", "#b39ddb"),
    ("draft_weights_gb", "Draft weights", "#3b4d8f"),
    ("draft_kv_gb", "Draft KV", "#7fb3d5"),
    ("selected_hidden_gb", "Selected hidden", "#fdd0a2"),
    ("context_feature_gb", "Context feature", "#6aa84f"),
    ("unclassified_cache_gb", "Other cache", "#a1d99b"),
    ("peak_minus_probed_gb", "Peak minus probed components", "#d9b3d9"),
]
TIME = [
    ("prefill_rest_s", "Prefill / first token / other", "#8c8c8c"),
    ("prefill_feature_s", "Prefill feature build + concat", "#a1d99b"),
    ("first_draft_s", "First draft", "#3b4d8f"),
    ("steady_draft_s", "Steady draft (all calls)", "#7fb3d5"),
    ("draft_logits_s", "Draft logits", "#e8a33d"),
    ("verify_s", "Target verify (all calls)", "#c0504d"),
    ("decode_feature_s", "Decode feature build", "#6aa84f"),
    ("decode_remainder_s", "Decode remainder", "#d9b3d9"),
    ("ar_decode_s", "AR decode (all steps)", "#555555"),
]
MODULES = [
    ("full_attention", "Full attention", "#c0504d"),
    ("gdn", "GDN mixer", "#7fb3d5"),
    ("ffn_dense", "Dense FFN", "#3b4d8f"),
    ("moe_experts", "Routed expert loop", "#6a51a3"),
    ("moe_router", "MoE router", "#e8a33d"),
    ("moe_shared_expert", "Shared expert", "#6aa84f"),
    ("ffn_moe+shared", "MoE wrapper", "#bcbddc"),
    ("lm_head", "LM head", "#756bb1"),
    ("outside_target_modules", "Outside target modules", "#b0b0b0"),
]
KERNELS = [
    ("gemm", "GEMM", "#3b4d8f"),
    ("attention_flash", "Flash attention", "#6aa84f"),
    ("attention_mem_efficient", "Mem-efficient attention", "#c0504d"),
    ("attention_math_softmax", "Attention math / softmax", "#e8a33d"),
    ("elementwise", "Elementwise / copy kernels", "#7fb3d5"),
    ("gdn_or_conv", "GDN / conv", "#b39ddb"),
    ("reduce", "Reduction", "#b0b0b0"),
    ("index_scatter_gather", "Index / gather / scatter", "#fdd0a2"),
    ("other", "Other kernels", "#d9b3d9"),
]


def figure(title, rows, cols, size, legend, foot):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=.98)
    fig.subplots_adjust(left=.07, right=.97, top=.90 if rows==5 else .87, bottom=.10 if rows==5 else .16,
                        hspace=.65 if rows==5 else .52, wspace=.33)
    fig.legend(handles=[Patch(color=color, label=label) for _,label,color in legend],
               loc="upper center", bbox_to_anchor=(.5,.956), ncol=4, fontsize=8, frameon=False)
    fig.text(.5,.02,foot,ha="center",fontsize=9,linespacing=1.5,
             bbox=dict(facecolor="#dcf0dd",edgecolor="#7bbf86",boxstyle="round,pad=.5"))
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",color="#e6e6e6",lw=.6);ax.set_axisbelow(True)
    return fig,axes


def load_all():
    records={m:read_json(record_path(m,STAMPS[m])) for m in MODELS}
    for m,rec in records.items():
        assert rec["measurement_protocol"]["version"]==2
        assert rec["gpu_selection"]["num_gpus"]==3
        rec["sequence_conditions"] = sorted([c for c in rec["conditions"] if c["sweep"]=="sequence"],key=lambda c:c["input_tokens"])
        assert [c["input_tokens"] for c in rec["sequence_conditions"]]==CONTEXTS
    return records


def memory_rows(records):
    rows=[];ar_rows=[]
    for m,rec in records.items():
        for c in rec["sequence_conditions"]:
            ar=c["ar"]["native"]["median"]
            weights=sum(ar["target_weight_bytes_per_device"].values())/1e9
            ar_rows.append(dict(model=m,context=c["input_tokens"],mode="AR native",phase_components_status="not_recorded",
                                target_weights_gb=weights,
                                allocated_sum_device_maxima_gb=ar["peak_memory_sum_device_maxima_bytes"]/1e9,
                                reserved_sum_device_maxima_gb=ar["peak_reserved_sum_device_maxima_bytes"]/1e9,
                                source=str(record_path(m,STAMPS[m]).relative_to(ROOT))))
            phases=c["passes"]["memory"]["median"]["phase_memory"]
            for name,label in PHASES:
                p=phases[name];peak=p["peak"];parts=peak["components"]
                r=dict(model=m,context=c["input_tokens"],mode="DFlash",phase=name,phase_label=label,count=p["count"],
                       peak_occurrence=peak["occurrence"],peak_interval_label=peak["peak_interval_label"],
                       allocated_before_gb=peak["allocated_before_bytes"]/1e9,
                       allocated_after_gb=peak["allocated_after_bytes"]/1e9,
                       interval_peak_gb=peak["interval_peak_bytes"]/1e9,
                       target_weights_gb=weights,
                       source=str(record_path(m,STAMPS[m]).relative_to(ROOT)))
                for field,source in [
                    ("target_attention_kv_gb","target_attention_kv_bytes"),
                    ("gdn_recurrent_gb","target_gdn_recurrent_bytes"),
                    ("gdn_conv_working_gb","target_gdn_conv_working_bytes"),
                    ("gdn_conv_recording_gb","target_gdn_conv_recording_bytes"),
                    ("draft_weights_gb","draft_weight_bytes"),
                    ("draft_kv_gb","draft_attention_kv_bytes"),
                    ("selected_hidden_gb","target_selected_hidden_bytes"),
                    ("context_feature_gb","context_feature_bytes")]:
                    r[field]=parts[source]/1e9
                r["unclassified_cache_gb"]=(parts["target_cache_unclassified_bytes"]+parts["draft_cache_other_bytes"])/1e9
                accounted=sum(r[k] for k,_,_ in MEMORY[:-1])
                r["peak_minus_probed_gb"]=r["interval_peak_gb"]-accounted
                assert r["peak_minus_probed_gb"]>=-1e-6,(m,c["input_tokens"],name,r["peak_minus_probed_gb"])
                r["peak_minus_probed_gb"]=max(0,r["peak_minus_probed_gb"])
                r["borrowed_gb"]=r["interval_peak_gb"]-max(r["allocated_before_gb"],r["allocated_after_gb"])
                rows.append(r)
    export("phase_memory_all_contexts.csv",rows)
    export("ar_memory_all_contexts.csv",ar_rows)
    return rows,ar_rows


def memory_figures(rows,ar_rows):
    for m in MODELS:
        foot=("DFlash: components probed at the close of each phase's largest occurrence; black line = interval peak (GB).\n"
              "Pale band = peak minus probed components, including temporary or already released storage; not pure activation.\n"
              "Phases are an envelope, not one chronological iteration. D1s is shown beside D1 for first/steady comparison.\n"
              "Native AR has no phase-memory probes; right column is whole-run sums of device maxima, not a phase split.")
        fig,axes=figure(f"{m}: what lives at every phase, at every context",5,3,(20,23),MEMORY,foot)
        # One common y range across contexts makes linear growth visible.
        maximum=max(r["interval_peak_gb"] for r in rows if r["model"]==m)
        weights=next(r["target_weights_gb"] for r in rows if r["model"]==m)
        draft_weights=next(r["draft_weights_gb"] for r in rows if r["model"]==m)
        ar_max=max(r["reserved_sum_device_maxima_gb"] for r in ar_rows if r["model"]==m)
        for ci,s in enumerate(CONTEXTS):
            selected=[r for r in rows if r["model"]==m and r["context"]==s]
            x=np.arange(len(selected))
            for col in (0,1):
                ax=axes[ci,col];bottom=np.zeros(len(selected))
                for key,label,color in MEMORY:
                    if col==1 and key in ("target_weights_gb","draft_weights_gb"):continue
                    vals=np.array([r[key] for r in selected])
                    ax.bar(x,vals,bottom=bottom,color=color,width=.85,
                           hatch="///" if key=="peak_minus_probed_gb" else None,lw=0)
                    bottom+=vals
                subtract=(weights+draft_weights) if col==1 else 0
                peaks=[r["interval_peak_gb"]-subtract for r in selected]
                ax.plot(x,peaks,color="#222",lw=1,marker="o",ms=2)
                for i in (0,5,6,9):
                    ax.annotate(f"{peaks[i]:.2f}",(i,peaks[i]),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7)
                ax.set_xticks(x,[label for _,label in PHASES],rotation=55,ha="right",fontsize=7)
                ax.set_ylim(0,(maximum-subtract)*1.2)
                ax.set_title(f"{s//1024}K{'†' if s==65536 else ''} · DFlash {'total live + peak gap' if col==0 else 'excluding target / draft weights'}",loc="left",fontsize=11,fontweight="bold")
                ax.set_ylabel("GB (decimal)")
            ax=axes[ci,2];ar=next(r for r in ar_rows if r["model"]==m and r["context"]==s)
            allocated=ar["allocated_sum_device_maxima_gb"];reserved=ar["reserved_sum_device_maxima_gb"]
            ax.bar(0,weights,color="#8c8c8c",width=.5)
            ax.bar(0,allocated-weights,bottom=weights,color="#d9b3d9",hatch="///",width=.5)
            ax.bar(1,reserved,color="#ddd",width=.5)
            for i,val in enumerate((allocated,reserved)):ax.text(i,val+ar_max*.02,f"{val:.2f} GB",ha="center",fontsize=10)
            ax.set_xticks([0,1],["Allocated\nΣ device maxima","Reserved\nΣ device maxima"],fontsize=9)
            ax.set_ylim(0,ar_max*1.28);ax.set_ylabel("GB (whole-run measurement)")
            ax.set_title(f"{s//1024}K · Native AR: phase split not recorded",loc="left",fontsize=11,fontweight="bold")
            ax.text(.04,.97,f"Measured target weights: {weights:.2f} GB\nHatched: peak minus weights, unclassified\nNo draft weights in native AR",transform=ax.transAxes,va="top",fontsize=9)
        save(fig,f"phase_memory_{m}")


def time_rows(records):
    rows=[];calls=[]
    for m,rec in records.items():
        for c in rec["sequence_conditions"]:
            p=c["passes"]["perf"]["median"];ar=c["ar"]["native"]["median"]
            assert not p["profiler_used_before"] and not ar["profiler_used_before"]
            for mode,stat in [("DFlash",p),("AR",ar)]:
                parts={k:0 for k,_,_ in TIME}
                if mode=="DFlash":
                    parts.update(prefill_feature_s=p["context_feature_prefill_s"],
                                 first_draft_s=p["draft_forward_first_s"],
                                 steady_draft_s=p["draft_forward_total_s"]-p["draft_forward_first_s"],
                                 draft_logits_s=p["draft_logits_total_s"],verify_s=p["target_verify_total_s"],
                                 decode_feature_s=p["context_feature_decode_total_s"])
                    parts["prefill_rest_s"]=p["time_to_first_token_s"]-parts["prefill_feature_s"]
                    known=sum(parts[k] for k,_,_ in TIME[2:7])
                    parts["decode_remainder_s"]=p["decode_latency_s"]-known
                else:
                    parts["prefill_rest_s"]=ar["time_to_first_token_s"]
                    parts["ar_decode_s"]=ar["decode_latency_s"]
                for value in parts.values():assert value>=0
                total=stat["time_to_first_token_s"]+stat["decode_latency_s"]
                assert np.isclose(sum(parts.values()),total)
                r=dict(model=m,context=c["input_tokens"],mode=mode,
                       e2e_interval_sum_s=total,record_total_median_s=stat["total_latency_s"],
                       ttft_s=stat["time_to_first_token_s"],decode_s=stat["decode_latency_s"],
                       source=str(record_path(m,STAMPS[m]).relative_to(ROOT)),**parts)
                for key,_,_ in TIME:
                    r[key.replace("_s","_e2e_pct")]=100*parts[key]/total
                    r[key.replace("_s","_decode_pct")]=(100*parts[key]/stat["decode_latency_s"] if key not in ("prefill_rest_s","prefill_feature_s") else 0)
                rows.append(r)
            calls.append(dict(model=m,context=c["input_tokens"],first_draft_ms=p["draft_forward_first_s"]*1000,
                              steady_draft_mean_ms=p["draft_forward_steady_mean_s"]*1000,
                              draft_logits_mean_ms=p["draft_logits_total_s"]/p["num_verify_steps"]*1000,
                              verify_mean_ms=p["target_verify_mean_s"]*1000,
                              ar_step_mean_ms=ar["time_per_output_token_s"]*1000,
                              verify_steps=p["num_verify_steps"],steady_draft_calls=p["num_verify_steps"]-1,
                              first_draft_stage_status="not_saved_in_arch_record",
                              source=str(record_path(m,STAMPS[m]).relative_to(ROOT))))
    export("phase_latency_all_contexts.csv",rows)
    export("phase_call_latency_all_contexts.csv",calls)
    return rows,calls


def latency_figures(rows):
    for m in MODELS:
        foot=("Perf-pass wall intervals; 256 output tokens. Every context has a DFlash row and its own native AR row.\n"
              "Total = median TTFT + median decode; stage medians are closed with a remainder. Tiny phases remain in the CSV.\n"
              "Remainder includes crops, synchronization and bookkeeping: their individual wall times were not recorded.\n"
              "Hybrid results use stock rollback. 64K† range limits apply as in the original figures.")
        fig,axes=figure(f"{m}: which phases dominate latency across context lengths?",2,2,(17,12),TIME,foot)
        selected=[r for s in CONTEXTS for mode in ("DFlash","AR") for r in rows if r["model"]==m and r["context"]==s and r["mode"]==mode]
        labels=[f"{r['context']//1024}K{'†' if r['context']==65536 else ''}  {r['mode']}" for r in selected]
        y=np.arange(10)
        for idx,ax in enumerate(axes.flat):
            decode=idx>=2;percent=idx%2==1;bottom=np.zeros(10)
            components=TIME[2:] if decode else TIME
            for key,label,color in components:
                vals=np.array([r[key]*(100/r["decode_s"] if decode else 100/r["e2e_interval_sum_s"]) if percent else r[key] for r in selected])
                ax.barh(y,vals,left=bottom,color=color,height=.75)
                for i,v in enumerate(vals):
                    frac=v if percent else v*100/(selected[i]["decode_s"] if decode else selected[i]["e2e_interval_sum_s"])
                    if frac>=12:
                        ax.text(bottom[i]+v/2,i,f"{v:.1f}%" if percent else f"{v:.2f}s",ha="center",va="center",fontsize=8,color="white" if color in ("#3b4d8f","#c0504d","#555555") else "#222")
                bottom+=vals
            ax.set_yticks(y,labels,fontsize=9);ax.invert_yaxis()
            ax.set_xlabel("% of decode latency" if decode and percent else "% of E2E latency" if percent else "seconds / request")
            ax.set_title(("Decode only" if decode else "Entire request (prefill + decode)")+(" · percentage" if percent else " · absolute latency"),loc="left",fontweight="bold",fontsize=12)
            if percent:ax.set_xlim(0,101)
            else:
                ax.set_xlim(0,max(bottom)*1.15)
                for i,v in enumerate(bottom):ax.text(v+max(bottom)*.015,i,f"Σ {v:.2f}s",va="center",fontsize=8)
        save(fig,f"phase_latency_{m}")


def trace_rows(records):
    modules=[];kernels=[]
    for m,rec in records.items():
        for c in rec["sequence_conditions"]:
            for mode,trace,phase in [
                ("DFlash verify",c["passes"]["trace"]["median"],"decode: target verify"),
                ("AR step",c["ar_trace"]["median"],"decode: ar step"),
                ("Steady draft",c["passes"]["trace"]["median"],"decode: draft forward")]:
                p=trace["analysis"]["phases"][phase];count=p["occurrences"]
                parts=p["module_kind_s"] if mode!="Steady draft" else p["kernel_category_s"]
                total=sum(parts.values())/count*1000
                for name,value in parts.items():
                    row=dict(model=m,context=c["input_tokens"],mode=mode,phase=phase,part=name,
                             profiled_occurrences=count,part_ms_per_occurrence=value/count*1000,
                             part_share_pct=value/count*1000/total*100,
                             total_profiled_ms_per_occurrence=total,
                             device_time_ms_per_occurrence=p["device_time_per_occurrence_s"]*1000,
                             kernels_per_occurrence=p["kernels_per_occurrence"],
                             activity_kind="kernels_only" if mode=="Steady draft" else "kernels_and_memcpy",
                             source=str(record_path(m,STAMPS[m]).relative_to(ROOT)))
                    (kernels if mode=="Steady draft" else modules).append(row)
    export("target_module_latency_all_contexts.csv",modules)
    export("steady_draft_kernel_latency_all_contexts.csv",kernels)
    return modules,kernels


def execution_figures(modules,kernels,calls):
    # Target execution: per model, both modes at all contexts, absolute + shares.
    for m in MODELS:
        foot=("Trace device activity / profiled occurrence, summed over devices; module stacks include kernels AND memcpy events.\n"
              "Shares are fractions of target device activity, not request wall latency. Black ticks = separate perf intervals.\n"
              "Trace samples the beginning of decode; ratios or gaps against perf are not utilization or measured CPU time.")
        fig,axes=figure(f"{m}: inside target verify and native AR, at every context",2,2,(17,12),MODULES,foot)
        for mode,col in [("DFlash verify",0),("AR step",1)]:
            selected=[r for r in modules if r["model"]==m and r["mode"]==mode]
            for percent,rowidx in [(False,0),(True,1)]:
                ax=axes[rowidx,col];bottom=np.zeros(5)
                totals=[next(r["total_profiled_ms_per_occurrence"] for r in selected if r["context"]==s) for s in CONTEXTS]
                for key,label,color in MODULES:
                    vals=np.array([sum(r["part_share_pct" if percent else "part_ms_per_occurrence"] for r in selected if r["context"]==s and r["part"]==key) for s in CONTEXTS])
                    ax.bar(range(5),vals,bottom=bottom,color=color,width=.68)
                    for i,v in enumerate(vals):
                        share=v if percent else v/totals[i]*100
                        if share>=12:ax.text(i,bottom[i]+v/2,f"{v:.0f}%" if percent else f"{v:.1f}",ha="center",va="center",fontsize=8,color="white" if color in ("#c0504d","#3b4d8f","#6a51a3","#756bb1") else "#222")
                    bottom+=vals
                ax.set_xticks(range(5),["4K","8K","16K","32K","64K†"])
                ax.set_ylabel("% of device activity (kernels + memcpy)" if percent else "Device activity ms / occurrence")
                ax.set_title(f"{mode} · {'module shares' if percent else 'module execution time'}",loc="left",fontsize=12,fontweight="bold")
                if percent:ax.set_ylim(0,104)
                else:
                    perf=[next(r["verify_mean_ms" if mode=="DFlash verify" else "ar_step_mean_ms"] for r in calls if r["model"]==m and r["context"]==s) for s in CONTEXTS]
                    ax.scatter(range(5),perf,marker="_",s=200,color="#222",label="Perf wall interval",zorder=4)
                    for i,y in enumerate(totals):ax.text(i,y+max(totals)*.025,f"Σ{y:.1f}",ha="center",fontsize=8)
                    ax.set_ylim(0,max(max(totals),max(perf))*1.22);ax.legend(frameon=False,fontsize=8)
        save(fig,f"target_execution_{m}")
    foot=("Steady draft forward: kernel categories from trace, all 5 contexts. First draft is excluded by the profiler window.\n"
          "Black ticks = separate perf steady-forward means; sums / gaps do not measure exact host overhead or utilization.\n"
          "GEMM includes multiple projections and MLPs. The five semantic draft-stage timers were not saved in these arch records.")
    fig,axes=figure("Steady draft forward: quantitative GPU work across models and contexts",3,2,(17,15),KERNELS,foot)
    for mi,m in enumerate(MODELS):
        selected=[r for r in kernels if r["model"]==m]
        totals=[next(r["total_profiled_ms_per_occurrence"] for r in selected if r["context"]==s) for s in CONTEXTS]
        for col in (0,1):
            ax=axes[mi,col];bottom=np.zeros(5)
            for key,label,color in KERNELS:
                vals=np.array([sum(r["part_share_pct" if col else "part_ms_per_occurrence"] for r in selected if r["context"]==s and r["part"]==key) for s in CONTEXTS])
                ax.bar(range(5),vals,bottom=bottom,color=color,width=.68)
                for i,v in enumerate(vals):
                    share=v if col else v/totals[i]*100
                    if share>=14:ax.text(i,bottom[i]+v/2,f"{v:.0f}%" if col else f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if color in ("#3b4d8f","#c0504d") else "#222")
                bottom+=vals
            ax.set_xticks(range(5),["4K","8K","16K","32K","64K†"]);ax.set_ylabel("% of GPU kernel time" if col else "GPU kernel ms / steady call")
            ax.set_title(f"{m} · {'kernel shares' if col else 'steady execution time'}",loc="left",fontsize=11,fontweight="bold")
            if col:ax.set_ylim(0,104)
            else:
                perf=[next(r["steady_draft_mean_ms"] for r in calls if r["model"]==m and r["context"]==s) for s in CONTEXTS]
                ax.scatter(range(5),perf,marker="_",s=200,color="#222",label="Perf steady mean")
                for i,y in enumerate(totals):ax.text(i,y+.2,f"Σ{y:.2f}",ha="center",fontsize=8)
                ax.set_ylim(0,max(max(totals),max(perf))*1.25);ax.legend(frameon=False,fontsize=8)
    save(fig,"steady_draft_all_contexts")


def call_figure(calls):
    foot=("Perf intervals in ms / call; first draft occurs once after TTFT. Steady draft repeats verify_steps − 1 times.\n"
          "Draft logits mean is total logits time / verify steps and includes target LM head plus transfers and argmax.\n"
          "Request totals / shares are in phase_latency figures: a large first call need not dominate a 256-token request.")
    legend=[("first_draft_ms","First draft", "#3b4d8f"),("steady_draft_mean_ms","Steady draft","#7fb3d5"),
            ("draft_logits_mean_ms","Draft logits","#e8a33d"),("verify_mean_ms","Target verify","#c0504d"),
            ("ar_step_mean_ms","Native AR step","#555555")]
    fig,axes=figure("Per-call phase costs: first draft, steady draft, verify and native AR",1,3,(18,7),legend,foot)
    for mi,m in enumerate(MODELS):
        ax=axes[0,mi];selected=[r for r in calls if r["model"]==m]
        for key,label,color in legend:ax.plot(range(5),[r[key] for r in selected],color=color,marker="o",label=label)
        ax.set_xticks(range(5),["4K","8K","16K","32K","64K†"]);ax.set_yscale("log");ax.set_ylabel("Wall interval ms / call (log scale)")
        ax.set_title(m,fontweight="bold",fontsize=12)
    save(fig,"phase_calls_all_contexts")


def verify_figures(memory,time,calls):
    foot=("D4 memory: phase interval peak with components probed at its close; hatched gap can include storage already released.\n"
          "All context lengths shown. Left: full footprint; right: remove resident target + draft weights to expose other costs.\n"
          "Native AR has no D4/phase-memory component measurement, so no synthetic AR component stack is drawn.")
    fig,axes=figure("Split at D4 target verify: memory components across models and context lengths",3,2,(18,16),MEMORY,foot)
    selected=[r for r in memory if r["phase"]=="decode: target verify"]
    export("verify_memory_all_contexts.csv",selected)
    for mi,m in enumerate(MODELS):
        rows=[r for r in selected if r["model"]==m]
        for col in (0,1):
            ax=axes[mi,col];bottom=np.zeros(5)
            for key,label,color in MEMORY:
                if col and key in ("target_weights_gb","draft_weights_gb"):continue
                vals=np.array([r[key] for r in rows])
                ax.bar(range(5),vals,bottom=bottom,color=color,width=.68,
                       hatch="///" if key=="peak_minus_probed_gb" else None)
                for i,v in enumerate(vals):
                    if v>=bottom[i]*.5 and v>=.15:
                        ax.text(i,bottom[i]+v/2,f"{v:.2f}",ha="center",va="center",fontsize=8,
                                color="white" if key in ("draft_weights_gb","target_attention_kv_gb") else "#222")
                bottom+=vals
            ax.plot(range(5),bottom,color="#222",marker="o",lw=1,ms=3)
            for i,v in enumerate(bottom):ax.text(i,v+max(bottom)*.025,f"Σ{v:.2f}",ha="center",fontsize=8)
            ax.set_ylim(0,max(bottom)*1.18);ax.set_xticks(range(5),["4K","8K","16K","32K","64K†"])
            ax.set_ylabel("GB at D4 phase / probed at phase close")
            ax.set_title(f"{m} · {'without target / draft weights' if col else 'full D4 split'}",loc="left",fontsize=11,fontweight="bold")
    save(fig,"verify_memory_split_all_contexts")
    foot=("Left compares one block verification to one AR token step; block includes anchor plus up to 15 proposals.\n"
          "Middle divides all verify interval time by 255 decode output tokens, so it includes acceptance and step counts.\n"
          "Right separates decode share from entire-request share (TTFT + decode). Every number comes from perf-pass timers.\n"
          "Hybrid speedups and acceptance use stock rollback; exact-state restoration cost has not been measured here.")
    legend=[("verify","Verify interval / token cost","#c0504d"),("ar","Native AR token step","#555555"),
            ("df","Entire DFlash decode / token","#3b4d8f"),("share","Verify share of E2E","#e8a33d")]
    fig,axes=figure("D4 verification overhead: absolute cost, output-token cost and latency share",3,3,(21,15),legend,foot)
    exports=[]
    for mi,m in enumerate(MODELS):
        call=[r for r in calls if r["model"]==m]
        rows=[r for r in time if r["model"]==m and r["mode"]=="DFlash"]
        ar=[r for r in time if r["model"]==m and r["mode"]=="AR"]
        for ci,(c,r,b) in enumerate(zip(call,rows,ar)):
            exports.append(dict(model=m,context=r["context"],verify_ms=c["verify_mean_ms"],ar_step_ms=c["ar_step_mean_ms"],
                                verify_over_ar=c["verify_mean_ms"]/c["ar_step_mean_ms"],
                                verify_minus_ar_ms=c["verify_mean_ms"]-c["ar_step_mean_ms"],
                                verify_total_s=r["verify_s"],verify_steps=c["verify_steps"],
                                verify_ms_per_decode_output_token=r["verify_s"]/255*1000,
                                df_tpot_ms=r["decode_s"]/255*1000,ar_tpot_ms=b["decode_s"]/255*1000,
                                verify_share_decode_pct=100*r["verify_s"]/r["decode_s"],
                                verify_share_e2e_pct=100*r["verify_s"]/r["e2e_interval_sum_s"]))
        e=exports[-5:];ax=axes[mi,0]
        ax.bar(np.arange(5)-.18,[r["verify_ms"] for r in e],.34,color="#c0504d")
        ax.bar(np.arange(5)+.18,[r["ar_step_ms"] for r in e],.34,color="#555555")
        for i,r in enumerate(e):
            ax.text(i,r["verify_ms"]+max(v["verify_ms"] for v in e)*.03,
                    f"{r['verify_over_ar']:.2f}×\n+{r['verify_minus_ar_ms']:.1f}ms",ha="center",fontsize=8)
        ax.set_ylim(0,max(r["verify_ms"] for r in e)*1.35);ax.set_ylabel("Wall interval ms / occurrence")
        ax.set_title(f"{SHORT[m]} · verify vs AR, per call",loc="left",fontsize=12,fontweight="bold")
        ax=axes[mi,1]
        for key,label,color in [("verify_ms_per_decode_output_token","Verify portion","#c0504d"),
                                ("df_tpot_ms","Entire DFlash decode","#3b4d8f"),("ar_tpot_ms","Native AR decode","#555555")]:
            ax.plot(range(5),[r[key] for r in e],color=color,marker="o",lw=1.8)
        ax.set_ylabel("ms / actual decode output token")
        ax.set_title(f"{SHORT[m]} · amortised cost per output token",loc="left",fontsize=12,fontweight="bold")
        ax=axes[mi,2]
        for offset,key,color in [(-.18,"verify_share_decode_pct","#c0504d"),(.18,"verify_share_e2e_pct","#e8a33d")]:
            vals=[r[key] for r in e];ax.bar(np.arange(5)+offset,vals,.34,color=color)
            for i,y in enumerate(vals):ax.text(i+offset,y+2,f"{y:.1f}",ha="center",fontsize=8)
        ax.set_ylim(0,110);ax.set_ylabel("Verify % of decode / E2E latency")
        ax.set_title(f"{SHORT[m]} · decode share vs E2E share",loc="left",fontsize=12,fontweight="bold")
        for ax in axes[mi,:]:ax.set_xticks(range(5),["4K","8K","16K","32K","64K†"])
    export("verify_overhead_all_contexts.csv",exports)
    save(fig,"verify_overhead_all_contexts")


def write_guide(time,calls):
    lines=[
        "# 전체 context의 phase 자원 및 latency 분석", "",
        "세 모델 각각의 4K·8K·16K·32K·64K를 모두 표시한다. 기본 block은 16, 출력은 256, B=1이며 같은 3-GPU protocol v2 기록을 사용한다.", "",
        "## 모델별 그림", "",
        "| 모델 | Phase 메모리 | DFlash / native AR latency | Verify / AR 내부 모듈 비용 |",
        "| --- | --- | --- | --- |",
    ]
    for m in MODELS:
        lines.append(f"| {m} | [PNG](phase_memory_{m}.png) / [PDF](phase_memory_{m}.pdf) | [PNG](phase_latency_{m}.png) / [PDF](phase_latency_{m}.pdf) | [PNG](target_execution_{m}.png) / [PDF](target_execution_{m}.pdf) |")
    lines += ["", "## D4 verify와 steady draft의 상세 그림", "",
              "- [D4 verify 메모리 split: 3모델 × 5context](verify_memory_split_all_contexts.png) / [PDF](verify_memory_split_all_contexts.pdf)",
              "- [D4 verify overhead: AR 대비 호출 비용, 출력 토큰당 비용, latency 기여율](verify_overhead_all_contexts.png) / [PDF](verify_overhead_all_contexts.pdf)",
              "- [Steady draft의 커널별 시간 및 비율](steady_draft_all_contexts.png) / [PDF](steady_draft_all_contexts.pdf)",
              "- [First / steady draft, draft logits, verify, AR의 호출당 시간](phase_calls_all_contexts.png) / [PDF](phase_calls_all_contexts.pdf)",
              "", "## 어떻게 읽는가", "",
              "Phase 메모리 그림은 각 context마다 전체 메모리, target/draft 가중치를 제외한 구성, native AR의 전체 실행 peak를 나란히 표시한다. 같은 모델에서는 context별 y축 범위가 같아서 자원 증가량을 비교할 수 있다. 각 phase의 반복 중 가장 큰 occurrence를 사용하므로 연결된 점은 단일 iteration의 시간 순서가 아니다. First draft 옆의 steady draft도 비교를 위한 배치다.", "",
              "D4 split은 같은 메모리 측정에서 verify phase만 추출한다. 보라색 사선 영역은 interval peak에서 phase 종료시 probe된 구성 요소를 뺀 나머지다. Verify 도중 해제된 prompt conv recording도 여기에 들어갈 수 있으므로 전부 attention activation이라고 해석하지 않는다. 가중치 제외 그림은 weights를 제거한 시각적 분해이며, AR 대비 메모리 overhead를 뜻하지 않는다.", "",
              "Latency 그림은 DFlash와 해당 모델의 native AR을 context별로 한 쌍씩 표시한다. 상단은 TTFT+decode 전체 request, 하단은 decode만의 절대 시간과 비율이다. First draft 1회와 steady draft 전체 호출을 구분한다. Prefill의 target forward/첫 토큰/기타는 개별 wall timer가 없어서 하나의 항으로 남긴다. Crop/동기화/기타 bookkeeping도 decode remainder에 남긴다.", "",
              "Target 모듈 그래프는 trace의 `module_kind_s`를 occurrence 수로 나눈 장치 activity다. **커널과 memcpy를 모두 포함**하며, attention mixer/FFN/GDN/LM head 등의 시간과 비율을 표시한다. 이 비율의 분모는 장치 activity이므로 request wall latency 기여율과 다르다. Black tick은 별도 perf 패스의 verify 또는 AR interval이다. 차이를 CPU 시간이나 GPU utilization으로 해석하지 않는다.", "",
              "Steady draft 내부 그래프는 `kernel_category_s`를 이용한 **커널만의 시간**이다. GEMM에는 여러 projection과 MLP가 포함되며 특정 projection 하나로 귀속하지 않는다. Profiler window는 first draft를 포함하지 않는다.", "",
              "Verify overhead 그림의 왼쪽은 block verify 1회와 AR 1토큰 step을 비교한다. `+ms`는 두 호출의 차이이고, 같은 토큰 수의 작업을 뺀 순수 overhead 측정이 아니다. 가운데는 verify 총 시간 / 실제 decode 출력 토큰 수(255)이므로 acceptance와 반복 횟수가 반영된다. 오른쪽은 verify의 decode 기여율과 E2E 기여율을 구분한다.", "",
              "## 미측정 항목", "",
              "이번 arch 기록에는 **native AR phase별 메모리 probe와 draft의 의미적 5-stage timer 값이 저장되어 있지 않다**. 따라서 AR phase별 KV/activation split, first/steady draft의 fc+norm/context-KV/cache append 단계별 wall time을 정확하게 재구성할 수 없다. 그림에는 AR의 실제 전체 실행 allocated/reserved peak와 가중치만 표시했고, draft 내부는 저장된 trace 커널 구성으로 표시했다. 이전 selective sweep은 모델·GPU 배치·측정 조건이 달라 이번 결과와 혼합하지 않았다.", "",
              "Hybrid 측정은 stock rollback 경로다. 정확한 state restoration 비용을 포함한 lossless speedup은 미확정이며, 64K의 8B RoPE 범위 및 35B draft 학습 길이 제한도 유지된다.", "",
              "## 작은 phase까지 확인하는 정량 표", "",
              "아래는 막대 안에 글자가 들어가지 않는 first/steady draft도 정확히 읽을 수 있도록 제공하는 표다. `steady 전체`는 한 호출의 시간이 아니라 해당 request에서 반복된 모든 steady forward의 합이다. 모든 phase의 E2E/decode 비율은 CSV에 별도로 저장한다.", "",
    ]
    for m in MODELS:
        lines += [f"### {m}", "",
                  "| Context | DF TTFT / AR TTFT (s) | First draft (ms) | Steady 전체 (s) / 호출 평균 (ms) | Draft logits 전체 (s) | Verify 전체 (s) / 호출 평균 (ms) | Verify % decode / E2E | DF decode / AR decode (s) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in CONTEXTS:
            d=next(r for r in time if r["model"]==m and r["context"]==s and r["mode"]=="DFlash")
            a=next(r for r in time if r["model"]==m and r["context"]==s and r["mode"]=="AR")
            c=next(r for r in calls if r["model"]==m and r["context"]==s)
            lines.append(f"| {s//1024}K | {d['ttft_s']:.3f} / {a['ttft_s']:.3f} | {c['first_draft_ms']:.3f} | {d['steady_draft_s']:.3f} / {c['steady_draft_mean_ms']:.3f} | {d['draft_logits_s']:.3f} | {d['verify_s']:.3f} / {c['verify_mean_ms']:.3f} | {d['verify_decode_pct']:.2f}% / {d['verify_e2e_pct']:.2f}% | {d['decode_s']:.3f} / {a['decode_s']:.3f} |")
        lines += [""]
    lines += ["## 데이터와 재현", "",
              "| CSV | 범위 |", "| --- | --- |",
              "| [phase_memory_all_contexts.csv](phase_memory_all_contexts.csv) | 180행: 3모델 × 5context × 12phase. GB 구성, before/after/peak, occurrence, borrowed |",
              "| [ar_memory_all_contexts.csv](ar_memory_all_contexts.csv) | 15행: native AR weights 및 전체 실행 allocated/reserved sums of device maxima |",
              "| [phase_latency_all_contexts.csv](phase_latency_all_contexts.csv) | 30행: 3모델 × 5context × DFlash/AR. 각 phase seconds, E2E 및 decode 비율 |",
              "| [phase_call_latency_all_contexts.csv](phase_call_latency_all_contexts.csv) | 15행: 호출당 ms 및 호출 수 |",
              "| [target_module_latency_all_contexts.csv](target_module_latency_all_contexts.csv) | Verify/AR 모듈별 device activity ms 및 비율; kernels + memcpy |",
              "| [steady_draft_kernel_latency_all_contexts.csv](steady_draft_kernel_latency_all_contexts.csv) | Steady draft kernel별 ms 및 비율; kernels only |",
              "| [verify_memory_all_contexts.csv](verify_memory_all_contexts.csv) | 15행: D4 memory split |",
              "| [verify_overhead_all_contexts.csv](verify_overhead_all_contexts.csv) | 15행: verify/AR 비용 차, 비율, 확정 출력당 비용, decode/E2E 기여율 |",
              "", "Trace CSV의 `part_ms_per_occurrence`, `part_share_pct`, `total_profiled_ms_per_occurrence`는 `activity_kind`에 따라 kernels-only 또는 kernels+memcpy를 뜻한다.", "",
              "```bash", "/home/seoyounglee/venvs/dflash-fla/bin/python visualization_long_context/plot_phases.py", "```", "",
              "[phase_sources.json](phase_sources.json)에 입력 기록을 고정했다. 새 GPU 실험 없이 기존 데이터로 재현한다.", ""]
    (OUT/"PHASE_VISUALIZATION.md").write_text("\n".join(lines))


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    records=load_all()
    memory,ar=memory_rows(records)
    memory_figures(memory,ar)
    time,calls=time_rows(records)
    latency_figures(time)
    modules,kernels=trace_rows(records)
    execution_figures(modules,kernels,calls)
    call_figure(calls)
    verify_figures(memory,time,calls)
    write_guide(time,calls)
    sources={"records":[str(record_path(m,STAMPS[m]).relative_to(ROOT)) for m in MODELS],
             "missing_measurements":["native AR phase memory / component split","semantic first / steady draft stage timers"],
             "reference":"visualization_selective/plot_phase_profile_v2.py"}
    (OUT/"phase_sources.json").write_text(__import__("json").dumps(sources,indent=2)+"\n")


if __name__=="__main__":
    main()
