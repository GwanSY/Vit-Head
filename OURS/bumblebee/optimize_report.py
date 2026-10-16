"""Publication figures and a concise evidence-based optimization report."""
import argparse
import csv
import html
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from runtime import ROOT, write
from optimize_analysis import uncertainty

NAMES = {"baseline":"Baseline", "organized":"Public slices", "reciprocal_only":"Slices + joint reciprocal",
         "batched":"Slices + joint normalization"}
COLORS = {"baseline":"#456a8d", "organized":"#dc924a", "reciprocal_only":"#458568", "batched":"#9b6089"}
CASES = ("S10R2","S6R6","S6R3L3")


def rows(out,name):
    path=out / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def value(x):
    try: return float(x)
    except (ValueError,TypeError): return float("nan")


def fmt(x,digits=3):
    x=value(x)
    return f"{x:.{digits}f}" if math.isfinite(x) else "—"


def table(data,columns):
    cells="".join("<th>"+html.escape(label)+"</th>" for key,label in columns)
    body="".join("<tr>"+"".join("<td>"+html.escape(str(r.get(key,"")))+"</td>" for key,label in columns)+"</tr>" for r in data)
    return "<div class='table'><table><thead><tr>"+cells+"</tr></thead><tbody>"+body+"</tbody></table></div>"


def generate(out):
    os.environ.setdefault("MPLCONFIGDIR",str(out / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    results=json.loads((out / "analysis.json").read_text())
    summaries=results["summary"]
    samples=rows(out,"raw_samples.csv");pairs=rows(out,"paired_differences.csv")
    primary=[r for r in summaries if r["phase"]=="confirm"]
    variants=[v for v in ("organized","reciprocal_only") if any(r["variant"]==v for r in primary)]
    figures=[]
    def save(fig,name,caption):
        fig.tight_layout()
        fig.savefig(out / (name+".pdf"),bbox_inches="tight")
        fig.savefig(out / (name+".png"),dpi=250,bbox_inches="tight")
        plt.close(fig);figures.append(dict(name=name,caption=caption))
    if primary:
        fig,axes=plt.subplots(1,2,figsize=(11,4.1))
        for index,variant in enumerate(variants):
            series=[next((r for r in primary if r["case"]==case and r["variant"]==variant),None) for case in CASES]
            for metric,axis in (("saved_mean_s",axes[0]),("saved_payload_mean_mib",axes[1])):
                xs=np.arange(3)+(index-.5)*.24
                for x,row in zip(xs,series):
                    if row is None: continue
                    prefix="saved_ci" if metric=="saved_mean_s" else "saved_payload_ci"
                    suffix="_s" if metric=="saved_mean_s" else "_mib"
                    lo,hi=row[prefix+"_low"+suffix],row[prefix+"_high"+suffix]
                    axis.errorbar(x,row[metric],yerr=[[row[metric]-lo],[hi-row[metric]]],
                        fmt="o",capsize=4,color=COLORS[variant],label=NAMES[variant] if x==xs[0] else None)
                    runtime_means=defaultdict(list)
                    for pair in pairs:
                        if pair["phase"]=="confirm" and pair["case"]==row["case"] and pair["variant"]==variant:
                            runtime_means[pair["session"]].append(value(pair["saved_s" if metric=="saved_mean_s" else "saved_payload_mib"]))
                    jitter=np.linspace(-.045,.045,len(runtime_means))
                    axis.scatter(x+jitter,[st.mean(v) for v in runtime_means.values()],s=14,color=COLORS[variant],alpha=.45)
        for axis in axes:
            axis.axhline(0,color="#888888",lw=.8);axis.set_xticks(range(3),CASES)
        axes[0].set_ylabel("Paired native time saved (s)")
        axes[1].set_ylabel("Paired payload saved (MiB)")
        axes[0].legend(fontsize=9);axes[0].set_title("Fresh-runtime means; cluster bootstrap 95% CI")
        axes[1].set_title("Actual two-party sent bytes")
        save(fig,"figure_A_paired_benefit","A：配对净收益；散点为每个独立运行时内两轮配对均值，误差线按运行时重采样。正值表示节约，通信来自双方真实发送字节。")
        decomposition=rows(out,"cost_decomposition.csv")
        categories=("matrix","gelu","exponential","comparison","normalization","organization","other")
        palette=("#456a8d","#86648f","#e1ab45","#c26964","#458568","#687078","#c8cbd0")
        fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
        for case,axis in zip(CASES,axes):
            chosen=[r for r in decomposition if r["phase"]=="confirm" and r["case"]==case and r["party"]=="0"]
            names=[v for v in ("baseline",*variants) if any(r["variant"]==v for r in chosen)]
            base=np.zeros(len(names))
            for category,color in zip(categories,palette):
                ys=[value(next(r for r in chosen if r["variant"]==v)[category+"_s_mean"]) for v in names]
                axis.bar(range(len(names)),ys,bottom=base,color=color,label=category)
                base+=ys
            axis.set_xticks(range(len(names)),["Base","Slices","Joint recip."][:len(names)],rotation=20)
            axis.set_title(case)
        axes[0].set_ylabel("Fixed P0 inclusive HLO wall time (s)")
        axes[-1].legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.02,1))
        save(fig,"figure_B_block_decomposition","B：固定 P0 的闭合 HLO 分解，包含通信等待。HAL/MPC 子阶段与等待区间不再叠加；矩阵与公共计算的漂移保留在实际总时间中。")
        key=rows(out,"key_operator_decomposition.csv")
        fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
        for case,axis in zip(CASES,axes):
            for j,variant in enumerate(("baseline",*variants)):
                ys=[]
                for component in ("public_gather","reciprocal","normalization_multiply"):
                    row=next((r for r in key if r["phase"]=="confirm" and r["case"]==case and r["variant"]==variant
                              and r["party"]=="0" and r["component"]==component),None)
                    ys.append(value(row["wall_s_mean"]) if row else 0.)
                axis.bar(np.arange(3)+(j-1)*.23,ys,width=.23,color=COLORS[variant],label=NAMES[variant])
            axis.set_title(case);axis.set_xticks(range(3),("Gather","Reciprocal","Norm. multiply"),rotation=22)
        axes[0].set_ylabel("Fixed P0 inclusive operator time (s)")
        axes[-1].legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.02,1))
        save(fig,"figure_C_key_operators","C：真实关键算子分解；这里只展示同一 HLO 层级的实际时间。倒数合批保持处理元素总数和每元素定点算法，调用减少不代表时间按调用数成比例减少。")
        roles=rows(out,"role_costs.csv")
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for party in (0,1):
            for column,metric in enumerate(("process_cpu_s","recv_blocking_union_s")):
                axis=axes[party,column]
                for j,variant in enumerate(("baseline",*variants)):
                    means=[]; deviations=[]
                    for case in CASES:
                        clusters=defaultdict(list)
                        for r in roles:
                            if r["phase"]=="confirm" and r["party"]==str(party) and r["case"]==case and r["variant"]==variant:
                                clusters[r["session"]].append(value(r[metric]))
                        vals=[st.mean(v) for v in clusters.values()]
                        means.append(st.mean(vals) if vals else float("nan"));deviations.append(st.stdev(vals) if len(vals)>1 else 0.)
                    axis.bar(np.arange(3)+(j-1)*.23,means,width=.23,yerr=deviations,capsize=3,
                             color=COLORS[variant],label=NAMES[variant])
                axis.set_xticks(range(3),CASES);axis.set_title(f"Fixed P{party}")
                axis.set_ylabel("Process CPU-seconds (all threads)" if column==0 else "Recv blocking union (wall s)")
        axes[0,0].legend(fontsize=8)
        save(fig,"figure_D_cpu_and_wait","D：进程 CPU-seconds 是实际各工作线程累计 CPU 用量，Recv 阻塞是墙钟区间并集，包含等待对方计算；二者不能相加。误差线为独立运行时均值的标准差。")
    full=[r for r in summaries if r["phase"]=="full"]
    if full:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for row in full:
            selected=[r for r in samples if r["phase"]=="full" and r["case"]==row["case"]]
            for variant in ("baseline",row["variant"]):
                data=sorted((r for r in selected if r["variant"]==variant),key=lambda r:int(r["session"]))
                axes[0].plot([int(r["session"]) for r in data],[value(r["native_s"]) for r in data],"o-",
                             label=NAMES[variant],color=COLORS[variant])
            paired=[r for r in pairs if r["phase"]=="full" and r["case"]==row["case"]]
            axes[1].scatter([int(r["session"]) for r in paired],[value(r["saved_s"]) for r in paired],color=COLORS[row["variant"]])
        axes[0].set_ylabel("12-layer real secure native time (s)");axes[0].set_xlabel("Randomized paired round")
        axes[1].axhline(0,color="#888888",lw=.8);axes[1].set_ylabel("Paired native time saved (s)");axes[1].set_xlabel("Round")
        axes[0].legend(fontsize=9)
        save(fig,"figure_E_full_inference","E：完整 12 层安全推理，双方每个对照分别使用独立冷运行时；轮内 AB/BA 顺序平衡随机化。纵轴来自真实执行，未乘以 Block 预测收益。")
    before_after=[]
    for r in primary+full:
        before_after.append(dict(case=r["case"],variant=NAMES[r["variant"]],
            n=f"{r['independent_runtimes']} / {r['paired_calls']}",
            seconds=f"{r['baseline_mean_s']:.3f} → {r['optimized_mean_s']:.3f}",
            gain=f"{r['saved_mean_s']:+.3f} [{r['saved_ci_low_s']:+.3f}, {r['saved_ci_high_s']:+.3f}]",
            percent=f"{r['saved_mean_percent']:+.2f}%",p=f"{r['exact_two_sided_p']:.4f} / {r['p_holm']:.4f}",
            payload=f"{r['saved_payload_mean_mib']:+.3f} [{r['saved_payload_ci_low_mib']:+.3f}, {r['saved_payload_ci_high_mib']:+.3f}]"))
    numerical=rows(out,"numerical_validation.csv")
    protocol=rows(out,"operator_summary.csv")
    operation_table=[]
    for case in CASES:
        for op in ("f_mmul","f_batch_mmul","f_gelu","f_reciprocal","f_mul","a2b","b2a","trunc_a","pphlo.gather"):
            r=dict(case=case,operation=op)
            for variant in ("baseline",*variants):
                entry=next((x for x in protocol if x["phase"]=="confirm" and x["case"]==case
                            and x["variant"]==variant and x["party"]=="0" and x["operation"]==op),None)
                r[variant]=fmt(entry["calls_mean"],1) if entry else "0"
            if primary: operation_table.append(r)
    # Conditional static budgets, never substitute these for measured savings.
    key_rows=rows(out,"key_operator_decomposition.csv")
    role_rows=rows(out,"cost_decomposition.csv")
    op_rows=rows(out,"protocol_summary.csv")
    limits=[]
    for case in CASES:
        def key_time(variant,component):
            r=next((x for x in key_rows if x["phase"]=="confirm" and x["case"]==case
                    and x["party"]=="0" and x["variant"]==variant and x["component"]==component),None)
            return value(r["wall_s_mean"]) if r else 0.
        role=next((x for x in role_rows if x["phase"]=="confirm" and x["case"]==case
                   and x["party"]=="0" and x["variant"]=="baseline"),None)
        if role is None: continue
        native=value(role["native_s_mean"]);gather=key_time("baseline","public_gather")
        b2a=next((x for x in op_rows if x["phase"]=="confirm" and x["case"]==case
                  and x["party"]=="0" and x["variant"]=="baseline" and x["name"]=="b2a"),None)
        limits.append(dict(case=case,party=0,baseline_native_mean_s=native,
            removable_public_gather_mean_s=gather,ideal_gather_only_fraction_percent=100*gather/native,
            matrix_and_gelu_mean_s=value(role["matrix_s_mean"])+value(role["gelu_s_mean"]),
            unchanged_matrix_and_gelu_fraction_percent=100*(value(role["matrix_s_mean"])+value(role["gelu_s_mean"]))/native,
            reciprocal_baseline_mean_s=key_time("baseline","reciprocal"),
            reciprocal_joint_mean_s=key_time("reciprocal_only","reciprocal"),
            reciprocal_stage_saved_mean_s=key_time("baseline","reciprocal")-key_time("reciprocal_only","reciprocal"),
            all_b2a_inclusive_union_mean_s=value(b2a["interval_union_s_mean"]) if b2a else 0.,
            assumption="Static local budget only: all other operator costs, scheduling and blocking unchanged; not a measured acceleration bound across implementations"))
    if limits:
        from experiment import save_rows
        save_rows(out / "optimization_limits.csv",limits)
    stable=[r for r in primary if r["saved_ci_low_s"]>0 and r["p_holm"]<.05]
    lines=["异构头执行开销的可优化性验证", "",
           f"真实配对测量：{results['measured_calls']} 次；含预热的独立数值复核：{results['numerical_checks_including_warmups']} 次；双方时间/字节/编译图闭合检查：{results['closure_checks']} 项，全部通过。",
           "正式 Block：每种配置 8 个独立运行时 × 2 轮正反顺序配对；头类型、私有权重、共享激活、FM64/f18（GELU FM32/f12）、joint QK/AV、25 线程及 1 Gbps/0.25 ms 网络保持一致。",
           "公开连续头选择/identity 恢复可消除本地 gather；S/R 小分母向量可共同调用原有安全倒数，再恢复原大小的归一化乘法批次。数学归约、ReLU 零分母保护及各元素精度不变。原生库和预训练权重未改动。",
           "仅合并倒数减少的是既有归一化内部 A2B/B2A/截断调用批次，不是新增或消除类型之间独立 HE/MPC 转换；每元素算术量并未按调用次数同比减少。"]
    for r in primary:
        status="通过预设净收益门槛" if r in stable else "尚未通过预设净收益门槛"
        lines.append(f"{r['case']} / {NAMES[r['variant']]}：{r['baseline_mean_s']:.3f} → {r['optimized_mean_s']:.3f} s；"
            f"配对平均节约 {r['saved_mean_s']:+.3f} s，95% CI [{r['saved_ci_low_s']:+.3f},{r['saved_ci_high_s']:+.3f}]，"
            f"{r['saved_mean_percent']:+.2f}%；通信节约 {r['saved_payload_mean_mib']:+.3f} MiB；{status}。")
    if full:
        for r in full:
            conclusion=("观察到所测完整模型的配对时间收益" if r['saved_ci_low_s']>0 and r['exact_two_sided_p']<.05
                        else "未证明完整模型稳定加速")
            lines.append(f"完整 {r['case']}，{r['independent_runtimes']} 轮冷执行配对："
                f"{r['baseline_mean_s']:.3f} → {r['optimized_mean_s']:.3f} s；节约 {r['saved_mean_s']:+.3f} s "
                f"[{r['saved_ci_low_s']:+.3f},{r['saved_ci_high_s']:+.3f}]；通信 {r['saved_payload_mean_mib']:+.3f} MiB；{conclusion}。")
    else:
        lines.append("未开展新增完整 ViT 性能推断：完整执行须先通过 Block 净收益门槛；Block 节约 × 12 不能当作完整模型实测收益。")
    lines+= ["计时：主指标取同一执行双方 native 总时间的最大值；HLO 采用固定角色、同一层级闭合分解。进程 CPU-seconds 与链路阻塞墙钟单列，不能与 HLO 时间叠加，也不能把墙钟减 Recv 并集称为纯 CPU。",
             "不确定性：按独立运行时/冷执行配对轮次重采样 30,000 次；精确双侧符号翻转检验，正式 Block 六项比较 Holm 校正。一个预实验运行时不提供跨运行时置信区间。t 区间作为敏感性分析另列。",
             "局限：单图片、第一层真实激活和固定同机两方网络；明文非连续头已做边界验证，其安全性能未验证；没有准确率预算搜索、新密文打包协议或跨网络验证。归一化的每元素安全操作仍需执行，公共矩阵/FFN/GELU 等限制整体收益。",
             "后续成本表应对可选执行图分别建模：固定逻辑头类型与组大小、公开选择路径、倒数合批大小、归一化乘法大小、公共矩阵和双方实际等待。避免用单头加性收益或调用数简单缩放；最后仍以同头配置整模型测量验收。"]
    if limits:
        lines.append("静态局部预算（固定 P0，假设其他成本不变）：gather 约 "+
            "/".join(f"{r['removable_public_gather_mean_s']:.3f}" for r in limits)+" s/Block，仅占原 Block 的 "+
            "/".join(f"{r['ideal_gather_only_fraction_percent']:.2f}%" for r in limits)+
            "；矩阵与 GELU 占约 "+"/".join(f"{r['unchanged_matrix_and_gelu_fraction_percent']:.1f}%" for r in limits)+
            "。这些是计时预算，不是实测加速或跨 Block 硬上限。")
        lines.append("倒数阶段本身的 P0 均值节约为 "+
            "/".join(f"{r['reciprocal_stage_saved_mean_s']:.3f}" for r in limits)+
            " s；实际总时间还包含归一化乘法、比较/归约、矩阵和 GELU 的变化。矩阵/GELU 的图与调用数保持一致，其计时差没有单独因果验证，不能全部归为优化引入的新开销。")
    if results["partial"]:
        lines.insert(0,"进行中：当前是报告生成检查，全部测量完成前不作最终性能结论。")
    (out / "summary.txt").write_text("\n".join(lines)+"\n")
    provenance=json.loads((out / "confirmation_provenance.json").read_text())
    history=("历史 out/12 已观察到纯公开 gather 和组大小相关的比较/归一化开销，本轮组织优化编译图与旧 slices 控制在去除模块名称后相同。本轮新增的是更严格的配对验证及既有倒数的联合批处理，未开发新密码协议。"
             "扩大整个归一化乘法批次的方案仅有探索性预实验，不能据此断言稳定加速或稳定退化。")
    notes=("显式倒数乘法与直接 FP32 除法可能有数个 ULP 的明文舍入差；采用 abs≤5e−6+1e−6|reference| 复核，安全数值门槛 relativeL2≤0.05、maxAbs≤0.5 保持原值。"
           "首次过严的绝对容差中止发生在安全执行前，原始记录保存在 aborted/；没有挑选或删除不利性能样本。"
           "reciprocal_only 使用经过公开常数 1 和匹配类型验证的局部 IR 降低，避免调用不同的通用定点除法。")
    page="""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>异构头开销可优化性验证</title>
<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 system-ui;color:#26313e;padding:0 18px}h1{font-size:27px}h2{font-size:21px;margin-top:32px}p{margin:13px 0}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border:1px solid #dae0e5;text-align:left}th{background:#f1f4f6}figure{margin:25px 0}img{width:100%}figcaption{font-size:14px;color:#536171}code{font-size:13px;word-break:break-all}.note{padding:13px;background:#f3f5f6}</style><body>
<h1>异构注意力头执行开销的可优化性验证</h1>"""
    page+="<p>"+html.escape(lines[2])+"</p><p>"+html.escape(lines[3])+"</p>"
    page+="<p>"+html.escape(history)+"</p><h2>同头配置的真实净收益</h2>"
    page+=table(before_after,(("case","头配置"),("variant","执行图"),("n","运行时 / 配对数"),
        ("seconds","平均时间 / s"),("gain","配对节约及 95% CI / s"),("percent","配对节约比例"),
        ("p","精确 p / Holm p"),("payload","通信节约及 95% CI / MiB")))
    page+="<p>"+html.escape(next(line for line in lines if line.startswith("不确定性：")))+"</p>"
    page+="<p>"+html.escape(next(line for line in lines if line.startswith("计时：")))+"</p>"
    page+="<p>统计中的节约比例是逐配对相对差的平均值，可能与两列总时间均值之比不同；主结论采用配对秒数及预设精确检验。</p>"
    page+="<h2>实际消除的操作</h2><p>"+html.escape(lines[4])+"</p><p>"+html.escape(lines[5])+"</p>"
    page+=table(operation_table,(("case","头配置"),("operation","实际算子"),("baseline","基线 P0 调用"),
        ("organized","组织优化"),("reciprocal_only","组织 + 合并倒数")))
    page+="<p class='note'>"+html.escape(notes)+"</p>"
    for f in figures:
        page+=f"<figure><a href='{f['name']}.pdf'><img src='{f['name']}.png'></a><figcaption>{html.escape(f['caption'])} PDF 为可直接使用的矢量图。</figcaption></figure>"
    page+="<h2>论文可用结论与适用范围</h2>"
    page+="".join("<p>"+html.escape(line)+"</p>" for line in lines[6:])
    page+="<h2>原始证据与复现</h2><p>"
    for file in ("raw_samples.csv","before_after.csv","paired_differences.csv","cost_decomposition.csv",
                 "key_operator_decomposition.csv","protocol_summary.csv","operation_volumes.csv",
                 "fixed_party_component_contrasts.csv","normalization_ablation.csv","numerical_validation.csv",
                 "secure_output_differences.csv","identity_checks.csv","closure_checks.csv","optimization_limits.csv","REPRODUCE.txt"):
        page+=f"<a href='{file}'>{file}</a>　"
    page+="</p><p>原生观察库 SHA256：<code>"+provenance["native_sha256"]+"</code><br>预训练权重 SHA256：<code>"+provenance["checkpoint_sha256"]+"</code></p></body></html>"
    (out / "report.html").write_text(page)
    write(out / "figure_manifest.json",figures)
    (out / "REPRODUCE.txt").write_text("""BumbleBee-ViT 异构头开销验证复现

工作目录：Vit-Head/OURS/bumblebee；使用原项目模型、图片与 Docker 依赖。
宿主分析 Python：/home/dm/anaconda3/envs/ss_ssm/bin/python
安全执行依赖：Docker ours-bb:base / openbumblebee，CPU SEAL 原生观察库 build/stage2.so。
模型、原始库及观察库的 SHA256 见 confirmation_provenance.json；禁止更换权重或精度。
按下列顺序串行运行，以冻结设计记录为准：

python optimize_study.py preflight --out out/reproduce-opt
OPENBLAS_NUM_THREADS=1 taskset -c 25-26 python optimize_driver.py --out out/reproduce-opt
OPENBLAS_NUM_THREADS=1 taskset -c 25-26 python optimize_analysis.py --out out/reproduce-opt
OPENBLAS_NUM_THREADS=1 taskset -c 25-26 python optimize_report.py --out out/reproduce-opt
python optimize_package.py --out out/reproduce-opt

driver：三种配置，各 8 个独立运行时，各 2 轮正反顺序配对。
所有变体预热后才测量；原始图、公开组织优化、仅合并倒数三者的逻辑头类型固定。
只有 Block 的配对 CI > 0 且六比较 Holm 精确 p < .05 才进行一组完整 12 层 ViT 的 6 轮冷执行对照。
原预实验为：python optimize_study.py blocks --out <new path> --phase pilot --sessions 1 --pairs 2 --variants baseline organized batched
预实验需要先在该路径运行 preflight；它不是正式复测的独立统计重复。

native 主计时：max(P0 native total, P1 native total)，不含加载、编译、输入上传及输出揭示。
CPU：节点 api.Runtime.run 外包层 time.process_time_ns，包含所有节点线程；其范围比 native 核心略宽。
等待：实际 YACL Send/Recv 记录区间并集，Recv 包含等待对方计算，不等于纯网络耗时。
HLO 为同层级闭合归类。HAL/MPC/解密/打包区间可能嵌套，只用于诊断，不与 HLO 叠加。
两轮配对先在运行时内平均，再按 8 个独立运行时报告不确定性；不把 16 次调用当成 16 个独立运行时。
full 每个 arm 使用独立冷运行时，以避免将首次密码初始化与后续热调用直接比较。

所有输出张量均保存在对应执行目录 outputs.npz；数值、双方字节、时间边界和编译操作序列重新核验。
explicit reciprocal 的局部 IR lowering 只处理已验证的公开常数 1 / 同型分母，调用基线同一 ReciprocalOp。
未降低定点位数、改变 Goldschmidt 迭代数、移除零分母保护或更换 S/R/L 定义。
按工程原门槛 relativeL2 <= .05、maxAbs <= .5，full 还要求与同配置明文参考 top1 相同。
FP32 参考舍入复核允许 abs <= 5e-6 + 1e-6*abs(reference)，不放宽安全执行门槛。

原始数据包：raw_executions.tar.gz；精简图表代码包：results.zip；成员 SHA256 见 raw_manifest.csv。
历史成本依据复用 out/12、out/11；旧 slices 与 organized 编译图仅模块名称不同。
本实验没有成本感知分配器、新密文打包、准确率预算搜索或模拟安全执行。
""")
    print("Report and vector figures generated:",out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/13")
    generate(parser.parse_args().out.resolve())
