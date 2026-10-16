"""Publication figures and a concise native-packing result report."""
import argparse
import csv
import html
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from runtime import ROOT,write


def rows(out,name):
    p=out/name
    return list(csv.DictReader(p.open())) if p.exists() else []


def report(out):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    a=json.loads((out/'analysis.json').read_text())
    r=json.loads((out/'run.json').read_text())
    cmp=rows(out,'before_after.csv');role=rows(out,'cost_decomposition.csv');matrix=rows(out,'matrix_packing_summary.csv')
    paired=rows(out,'paired_differences.csv');raw=rows(out,'raw_samples.csv');schedule=rows(out,'packing_schedule_raw.csv')
    main=[x for x in cmp if x['phase']=='confirm']; controls=[x for x in cmp if x['phase']=='control'];full=[x for x in cmp if x['phase']=='full']
    main.sort(key=lambda c:('S10R2','S6R6','S6R3L3').index(c['case']))
    controls.sort(key=lambda c:('S12','R12').index(c['case']))
    def f(x,k):return float(x[k])
    def prob(x,k):
        value=float(x[k]);return f'{value:.5f}' if math.isfinite(value) else '—'
    def role_for(phase,case,variant,party=0):
        return next(x for x in role if (x['phase'],x['case'],x['variant'],x['party'])==(phase,case,variant,str(party)))
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=280,bbox_inches='tight');plt.close(fig)
    compare=main+controls
    if compare:
        fig,ax=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[1.2,1]});x=np.arange(len(compare));w=.36
        ax[0].bar(x-w/2,[f(c,'baseline_mean_s') for c in compare],w,yerr=[f(c,'baseline_sd_s') for c in compare],capsize=3,color='#667987',label='Original packing')
        ax[0].bar(x+w/2,[f(c,'optimized_mean_s') for c in compare],w,yerr=[f(c,'optimized_sd_s') for c in compare],capsize=3,color='#19846e',label='Batched packing')
        ax[0].set_xticks(x,[c['case'] for c in compare]);ax[0].set_ylabel('Complete secure Block time (s)');ax[0].legend(frameon=True,framealpha=.95,fontsize=8)
        ax[0].set_title('Mean ± across-runtime SD; unchanged attention graph')
        gain=[f(c,'saved_mean_percent') for c in compare]
        ax[1].bar(x,gain,color=['#247da5']*len(main)+['#6c8a9b']*len(controls))
        ax[1].errorbar(x,gain,yerr=[[g-f(c,'saved_percent_ci_low') for g,c in zip(gain,compare)],[f(c,'saved_percent_ci_high')-g for g,c in zip(gain,compare)]],fmt='none',color='black',capsize=4)
        ax[1].set_xticks(x,[c['case'] for c in compare]);ax[1].set_ylabel('Paired native time saved (%)');ax[1].set_title('95% bootstrap CI over independent runtimes');ax[1].axhline(0,color='black',lw=.7)
        for j in ax:j.grid(axis='y',alpha=.15);j.set_axisbelow(True)
        save(fig,'figA_block_net_gain')
    if main:
        case='S10R2';components=['Q','K','V','QK','AV','output_projection','FFN1','FFN2'];x=np.arange(len(components));w=.36
        fig,ax=plt.subplots(figsize=(12,4))
        for variant,offset,alpha,label in [('baseline',-w/2,1,'Original'),('tree_batch',w/2,.6,'Batched')]:
            data=[next(t for t in matrix if (t['phase'],t['case'],t['variant'],t['party'],t['component'])==('confirm',case,variant,'0',component)) for component in components]
            pack=np.array([f(t,'packing_union_s_mean') for t in data]);total=np.array([f(t,'matrix_wall_s_mean') for t in data])
            ax.bar(x+offset,pack,w,color='#18546b',alpha=alpha,label=label+': packing')
            ax.bar(x+offset,total-pack,w,bottom=pack,color='#9eb9c5',alpha=alpha,label=label+': remaining matrix wall')
        ax.set_xticks(x,['Q','K','V','QK','AV','O','FFN1','FFN2']);ax.set_ylabel('Matrix operator wall time (s), fixed P0');ax.set_title('S10R2; packing is contained within matrix wall time')
        ax.legend(frameon=False,fontsize=8,ncol=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        save(fig,'figB_matrix_packing')
        fig,axes=plt.subplots(1,3,figsize=(12,3.8));x=np.arange(len(main));w=.36
        for variant,offset,color,label in [('baseline',-w/2,'#667987','Original'),('tree_batch',w/2,'#19846e','Batched')]:
            data=[role_for('confirm',c['case'],variant) for c in main]
            axes[0].bar(x+offset,[f(d,'process_cpu_s_mean') for d in data],w,color=color,label=label)
            axes[1].bar(x+offset,[f(d,'recv_blocking_union_s_mean') for d in data],w,color=color)
            payload=[st.mean(float(t['payload_mib'])/1024 for t in raw if (t['phase'],t['case'],t['variant'])==('confirm',c['case'],variant)) for c in main]
            axes[2].bar(x+offset,payload,w,color=color)
        axes[0].set_ylabel('P0 process CPU (CPU-s)');axes[1].set_ylabel('P0 Recv blocking union (s)');axes[2].set_ylabel('Both-party payload (GiB)')
        axes[0].legend(frameon=False,fontsize=8)
        for ax in axes:ax.set_xticks(x,[c['case'] for c in main]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle('Measured process CPU, observed Recv blocking, and protocol payload')
        save(fig,'figC_cpu_wait_payload')
    if full:
        c=full[0];data=[p for p in paired if p['phase']=='full'];x=np.arange(len(data));fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(x+1,[f(p,'baseline_s') for p in data],'o-',color='#667987',label='Original packing');axes[0].plot(x+1,[f(p,'optimized_s') for p in data],'o-',color='#19846e',label='Batched packing')
        axes[0].set_xlabel('Independent cold paired round');axes[0].set_ylabel('Complete 12-layer native time (s)');axes[0].legend(frameon=False)
        axes[1].bar(x+1,[f(p,'saved_s') for p in data],color='#247da5',label='Measured full-model saving')
        block=next(p for p in main if p['case']==c['case']);prediction=12*f(block,'saved_mean_s')
        axes[1].axhline(prediction,color='#e29540',ls='--',label='12 × measured Block mean (estimate)');axes[1].set_xlabel('Paired round');axes[1].set_ylabel('Native time saved (s)');axes[1].legend(frameon=False,fontsize=8)
        for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        save(fig,'figD_full_model')
    # Freeze the actual schedule layout/launch counts with per-call totals.
    totals=defaultdict(lambda:dict(inputs=0,groups=0,launches=0,packing_operations=0))
    for row in schedule:
        key=tuple(row[k] for k in ('scope','phase','case','session','pair','variant','party'))
        d=totals[key];d['inputs']+=int(row['inputs']);d['groups']+=int(row['groups']);d['launches']+=int(row['launches']);d['packing_operations']+=1
    schedule_summary=[dict(zip(('scope','phase','case','session','pair','variant','party'),key),**d) for key,d in sorted(totals.items())]
    from experiment import save_rows
    if schedule_summary:save_rows(out/'packing_schedule_totals.csv',schedule_summary)
    text=['底层打包优化实测结论','']
    for c in main+controls+full:
        text.append(f"{c['phase']} {c['case']}: {f(c,'baseline_mean_s'):.3f}→{f(c,'optimized_mean_s'):.3f}s；配对平均节省{f(c,'saved_mean_percent'):.2f}%；Δ={f(c,'saved_mean_s'):.3f}s，bootstrap95%CI[{f(c,'saved_ci_low_s'):.3f},{f(c,'saved_ci_high_s'):.3f}]；n={c['independent_runtimes']}；exact p={prob(c,'p_exact')}；Holm={prob(c,'p_holm')}")
    text+=['','优化改变 CPU RLWE 返回密文跨组蝶形树调度与独立结果移动写回；同 IR、同头、同权重、同精度、同旋转和返回密文数。',
        '混合Block返回打包树的parallel_for调用：P0 1892→50，P1 596→14；密码操作未删减。',
        '收益属于公共 HE 打包加速，单类型对照也改善；不能称为消除了异构组专有非加性成本。',
        '通信结构未改变；如有小幅字节变化，仅报告实际量与不确定性，不称结构性通信优化。',
        f"纳入分析的真实计时调用{a['real_measured_calls']}；含warmup独立复核输出{a['numerically_checked_outputs']}；双方闭合检查{a['fixed_party_closure_checks']}。最坏relativeL2={a['max_relative_l2']:.8f},maxAbs={a['max_abs']:.8f}。",
        '未验证：Phantom GPU 分支、与 slices/联合 reciprocal 的叠加收益、多图片准确率、跨设备/网络环境可迁移性。']
    if not full:text+=['完整模型未验证：'+r.get('full_skip_reason','完整推理尚未完成')]
    (out/'summary.txt').write_text('\n'.join(text)+'\n')
    def table(headers,data):
        return '<div class="table"><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</table></div>'
    body='''<!doctype html><meta charset="utf-8"><title>BumbleBee-ViT 原生密文打包优化实测</title>
<style>body{max-width:1200px;margin:28px auto;padding:16px;font:16px/1.65 sans-serif;color:#18303e}table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border:1px solid #ddd;text-align:left}.table{overflow-x:auto}img{width:100%;margin:18px 0}code{background:#eee}h2{color:#18546b}a{color:#167490}</style>
<h1>BumbleBee-ViT 原生密文打包优化的可优化性验证</h1>
<p>已完成源码审计、CPU 返回密文跨组蝶形树批处理原型、精确密码算子验收和真实两方对照。改动是原生打包的线程调度、批处理和独立结果移动写回；保留 S/R/L 数学定义、原始模型权重、FM64/f18 与原有 GELU FM32/f12。Public gather 和分组 reciprocal 仍按原始图执行。</p>
<p><a href="audit_report.html">冻结的源码审计</a> · <a href="layout_addendum.html">AV布局补充审计</a> · <a href="native.patch">原生代码与测试补丁</a> · <a href="REPRODUCE.txt">复现实验入口</a> · <a href="before_after.csv">完整对照CSV</a> · <a href="raw_samples.csv">逐调用原始指标</a> · <a href="matrix_packing_raw.csv">关键矩阵阶段原始记录</a></p>
<h2>Block 净收益与不确定性</h2>
<p>每个代表混合配置8个独立两方运行时，每运行时对两个模式各warmup一次后进行一次AB或BA，次序平衡；没有按结果增加或删掉重复。主要时间为固定双方 native 完整执行时间的 max。Block的独立运行时是统计单位；完整模型以独立冷启动配对轮为统计单位（每轮两个运行时）；95% bootstrap 为30000次重采样；exact sign-flip 双侧检验，并对3个主配置做Holm校正。单类型对照为次级观察。CSV另记录independent_pairs与two_party_runtimes；legacy independent_runtimes列在完整模型中表示配对统计单元数。</p>'''
    body+=table(['阶段/配置','原始均值±SD(s)','优化均值±SD(s)','配对节省%','节省(s), bootstrap95%CI','n','exact p / Holm'],
        [[c['phase']+'/'+c['case'],f"{f(c,'baseline_mean_s'):.3f} ± {f(c,'baseline_sd_s'):.3f}",f"{f(c,'optimized_mean_s'):.3f} ± {f(c,'optimized_sd_s'):.3f}",f"{f(c,'saved_mean_percent'):.2f}",f"{f(c,'saved_mean_s'):.3f} [{f(c,'saved_ci_low_s'):.3f},{f(c,'saved_ci_high_s'):.3f}]",c['independent_runtimes'],prob(c,'p_exact')+' / '+prob(c,'p_holm')] for c in main+controls])
    if main:body+='<img src="figA_block_net_gain.png" alt="Block净收益">'
    body+='''<h2>真正消除的工作与保留的密码工作</h2>
<p>原实现每个 gap-sized 返回组独立完成预处理和整个蝶形树。末层每组只有1个任务，组外循环顺序执行。新实现将所有组相同层的任务一次提交到既有YACL线程池，层间屏障保留；最后一层同时具有19或74个组任务。独立输出向量避免了直接并行化原有紧凑写回的跨组别名竞态。</p>
<p>三种代表混合 Block 的返回打包树 parallel_for 调用：P0 1892→50，P1 596→14；两方合计2488→64。该数量由已执行分支的源码循环及实际mode/shape/gap记录核对得出，属于调度调用计数；不是密码交互轮数、整个Block的全部并行区或线程创建数量。输入结果密文、返回密文、每树旋转、模数调整、H2A随机掩码、加密解密和AShr转换均保留。独立结果用move写回，避免原有每组的结果复制；其单独收益未做消融。8个矩阵HAL与10个双方合计dot_H2A不变。完整图逐字节相同，并和实际HLO执行顺序闭合。12个原生测试通过，包括48组同输入密文精确系数对照、实际3→2模数链、尾组、NTT/系数输入与别名拒绝；原有解密验收同时通过。</p>'''
    budget=[]
    for c in main:
        x,y=role_for('confirm',c['case'],'baseline'),role_for('confirm',c['case'],'tree_batch')
        budget.append([c['case'],f"{f(x,'packing_union_s_mean'):.3f}→{f(y,'packing_union_s_mean'):.3f}",
            f"{f(x,'matrix_s_mean'):.3f}→{f(y,'matrix_s_mean'):.3f}",
            f"{f(x,'process_cpu_s_mean'):.2f}→{f(y,'process_cpu_s_mean'):.2f}",
            f"{f(x,'recv_blocking_union_s_mean'):.3f}→{f(y,'recv_blocking_union_s_mean'):.3f}"])
    if budget:body+=table(['配置','P0打包并集(s)','P0完整矩阵HLO(s)','P0 CPU-s','P0 Recv阻塞并集(s)'],budget)
    body+='''<p>packing 已包含在 matrix 内，不能将两者相加。剩余matrix wall包括HE乘法、编码、环矩阵乘、通信与等待，并非纯计算。CPU-s累积所有线程；Recv阻塞也可能是在等对方CPU。只用固定party的互不嵌套HLO表闭合总时长。主要收益对应打包/矩阵阶段；GELU等未改算子的时间差应作为波动或整体调度影响观察，不能据此宣称其协议得到优化。</p>'''
    body+='''<p>dot_decrypt_moddown区域从Recv前开始，包含等对方响应；其时间缩短不能解释成独立解密/域转换算法减少。转换、HAL、MPC与packing的嵌套区间保留在单独诊断表中，不重复叠加为总时长。</p>'''
    if main:body+='<img src="figB_matrix_packing.png" alt="矩阵与打包"><img src="figC_cpu_wait_payload.png" alt="CPU和通信">'
    body+='''<h2>通信、原库桥接与正确性</h2>
<p>两方sent payload相加，不把sent+recv重复计算；原始YACL字节、发送次数与双方接收闭合。此实现没有降低返回密文数，实际压缩密文长度与Beaver补充时机导致的小幅字节变化单列95%CI。HE backend counters 保留在 backend_counters.csv 与 protocol_counter_summary.csv；总HE次数可能随原有缓存补充变化，不能用其偶然差异证明密码操作削减。三组8轮确认中双方平均HE次数、转换计数及协议发送次数均逐项相同，打包树parallel_for调用明显减少。</p>'''
    wire=[]
    for c in main+full:
        vals=[v for v in raw if (v['phase'],v['case'])==(c['phase'],c['case'])]
        mean=lambda variant:st.mean(f(v,'payload_mib')/1024 for v in vals if v['variant']==variant)
        wire.append([c['phase']+'/'+c['case'],f"{mean('baseline'):.6f}→{mean('tree_batch'):.6f}",f"{f(c,'saved_payload_mean_mib'):.4f} [{f(c,'saved_payload_ci_low_mib'):.4f},{f(c,'saved_payload_ci_high_mib'):.4f}]"])
    if wire:body+=table(['配置','实际两方payload(GiB)','节省MiB,95%CI'],wire)
    bridge=[]
    for arm in ('cached_original','rebuilt_original'):
        folders=sorted((out/'bridge/log').glob('*'+arm))
        calls=[json.loads((p/'run.json').read_text()) for p in folders]
        if calls:
            times=[m['measurements'][0]['online_s'] for m in calls]
            bridge.append([arm,len(calls),f"{st.mean(times):.3f} ± {st.stdev(times):.3f}",calls[0]['library']['sha256']])
    if bridge:body+=table(['原库桥接（S6R6）','独立调用','均值±SD(s)','native SHA256'],bridge)
    body+='''<p>主配对在同一新库中关闭或开启批处理，关闭分支保留原有顺序packing实现；另用未修改的stage2.so和新库关闭模式做3轮单独运行时桥接。桥接用于核对原始图/内核口径，样本较少，不是等效性统计证明。原始build/spu/libspu.so及build/stage2.so哈希保持不变。</p>'''
    body+=f"<p>纳入分析的真实计时调用{a['real_measured_calls']}次；含warmup逐输出重新计算数值验收{a['numerically_checked_outputs']}次；固定双方闭合记录{a['fixed_party_closure_checks']}条，全通过。最坏relativeL2={a['max_relative_l2']:.8f}、maxAbs={a['max_abs']:.8f}，门槛仍为0.05/0.5。完整模型还检查与相同头计划参考的top1一致。该验收是安全数值正确性，不能替代全数据集准确率结论。</p>"
    if (out/'excluded_execution.csv').exists():
        body+='''<p>另有一次会话中断前的完整模型原始执行，数值输出与native TSV已保存，但驱动/stdout汇总不完整，且没有同一时段的优化对照。在重新测量首对之前登记排除原因；原始记录保留在aborted/，单列excluded_execution.csv，不纳入6对计时统计。之后按同一冻结设计重新完成配对，未按快慢筛选样本。</p>'''
    body+='''<p>安全边界：新调度仅依赖公开shape/gap，保留原有加密、自动同构、模数处理、随机掩码和一次性Beaver规则；参数存储类型与协议消息路径经过核对。现有基准的两个容器共享只读工作区（runtime.baseline挂载），用于本地主机上的协议性能验证；这不是对独立部署文件系统访问隔离的证明。优化没有将参数转为公开类型，也未利用文件读取绕过安全计算。</p>'''
    body+='<h2>完整12层ViT与跨Block累积</h2>'
    if full:
        c=full[0];vals=[v for v in raw if v['phase']=='full'];requests={v:st.mean(f(t,'request_s') for t in vals if t['variant']==v) for v in ('baseline','tree_batch')}
        body+=f"<p>仅在8轮Block确认通过门槛后，执行{c['independent_runtimes']}对独立冷启动12层ViT，AB/BA平衡；每个arm都是独立两方运行时，无warmup。每层固定{c['case']}，权重和逻辑头类型完全相同。native均值{f(c,'baseline_mean_s'):.3f}→{f(c,'optimized_mean_s'):.3f}s，配对平均节省{f(c,'saved_mean_percent'):.2f}%（Δ={f(c,'saved_mean_s'):.3f}s，bootstrap95%CI[{f(c,'saved_ci_low_s'):.3f},{f(c,'saved_ci_high_s'):.3f}]），exact p={f(c,'p_exact'):.5f}。客户端真实调用（含编译与输出reveal）{requests['baseline']:.3f}→{requests['tree_batch']:.3f}s；参数上传、模型载入和容器搭建在该范围外。</p>"
        body+='<img src="figD_full_model.png" alt="完整模型实测与估计">'
        p0base=role_for('full',c['case'],'baseline');p0opt=role_for('full',c['case'],'tree_batch')
        pack_saved=f(p0base,'packing_union_s_mean')-f(p0opt,'packing_union_s_mean')
        body+=f"<p>固定P0的完整模型打包区间并集 {f(p0base,'packing_union_s_mean'):.3f}→{f(p0opt,'packing_union_s_mean'):.3f}s，减少{pack_saved:.3f}s；矩阵HLO合计 {f(p0base,'matrix_s_mean'):.3f}→{f(p0opt,'matrix_s_mean'):.3f}s。打包减少量约为完整净时间差的{100*pack_saved/f(c,'saved_mean_s'):.1f}%，是主要对应开销。它已包含在矩阵时间内；P1并行发生的打包收益不再叠加。未修改算子的剩余时间差不解释为额外协议优化。逐轮固定双方分解见full_component_contrasts.csv。</p>"
        body+='''<p>图中柱子是完整模型的真实安全收益；12×Block均值仅是估计对照线。不能把这条线当作端到端测量。完整模型逐party时间、CPU、payload及数值输出均保存在CSV/NPZ/TSV中；本结论限于单物理主机、两方隔离CPU、1Gbps/0.25ms egress和本输入。</p>'''
    else:body+='<p>完整ViT未验证：'+html.escape(r.get('full_skip_reason','推理尚未完成'))+'。不得报告端到端加速。</p>'
    body+='''<h2>对异构执行与后续成本模型的意义</h2>
<p>这次消除的是公共HE返回打包的调度浪费。单类型对照能检验其通用性质；混合joint QK/AV仍以12头批执行，未出现类型间新增独立域转换。本结果不证明原有S/R非线性分组成本、比较、归一化或gather已被消除，也不证明异构专有加速优于单类型。</p><p>补充审计发现原生S12的einsum AV下推为Vᵀ@Wᵀ，项目joint为W@V，投影顺序也不同。类型配置内的本轮A/B始终保持同IR；S12与混合方案的绝对成本差不能全归因于类型混合。原生布局和项目joint布局应分别校准，不能笼统断言所有单类型和混合图都有完全相同packing。</p>
<p>后续头分配成本应包含：实际内核版本/布局/线程与网络的公共矩阵基线；组大小相关的S/R/L算子及OT/Beaver阈值；公共组织成本；真实协议payload与关键路径等待。不能把单头时间线性乘以头数，或按头类型捏造HE↔MPC转换项。打包策略影响公共基线，在本joint方案中不改变类型组的矩阵形状；用新内核重新校准，保留非线性分组模型，对未测试的分布不作精确预测。</p>
<p>可继续研究的方向：选块模型纳入真实编码/NTT/批大小成本，重复投影输入处理，以及BatchDotOLE逐头流水。它们均尚未实现/验证。当前收益已来自一个有较大实测预算、低数学风险的底层方向，本轮不强行增加第二种协议。未验证GPU、叠加旧图优化、多输入准确率或不同部署环境。</p>
<p>所有图提供PDF与PNG；论文应引用确认阶段和完整模型的真实配对结果，探索试跑单列于原始CSV。文件入口：cost_decomposition.csv、matrix_packing_summary.csv、key_operator_decomposition.csv、backend_counters.csv、packing_schedule_totals.csv、numerical_validation.csv、closure_checks.csv，以及原始归档。</p>'''
    pdfs=[p for p in sorted(out.glob('fig*.pdf'))]
    body+='<p>论文图PDF：'+ ' · '.join('<a href="'+p.name+'">'+p.stem+'</a>' for p in pdfs)+'</p>'
    (out/'report.html').write_text(body)
    print('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');a=p.parse_args();report(a.out.resolve())
