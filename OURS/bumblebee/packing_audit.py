"""Freeze a source/IR/historical execution audit before native modifications."""
import csv
import hashlib
import html
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from runtime import ROOT, digest, write
from mixed import load_events, union_seconds


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    src = ROOT / 'build/src'
    manifest = json.loads((ROOT / 'build/stage2.json').read_text())
    sources = {}
    names = ['libspu/mpc/bumblebee/protocol.cc', 'libspu/mpc/bumblebee/type.h',
        'libspu/mpc/bumblebee/state.cc', 'libspu/mpc/bumblebee/kernels/matmul.cc',
        'libspu/mpc/bumblebee/kernels/mul.cc', 'libspu/mpc/bumblebee/kernels/compare.cc',
        'libspu/mpc/bumblebee/kernels/truncate.cc',
        'libspu/mpc/bumblebee/arith/bumblebee_dot.cc',
        'libspu/mpc/bumblebee/arith/matmat_prot.cc',
        'libspu/mpc/bumblebee/arith/vector_encoder.cc',
        'libspu/mpc/bumblebee/arith/bumblebee_mul_impl.cc',
        'libspu/mpc/bumblebee/rlwe/packlwes.cc', 'libspu/mpc/bumblebee/rlwe/packlwes.h',
        'libspu/mpc/bumblebee/rlwe/packlwes_test.cc',
        'libspu/mpc/bumblebee/rlwe/phantom_evaluator.cc',
        'libspu/kernel/hal/fxp_base.cc', 'libspu/kernel/hal/fxp_approx.cc',
        'libspu/mpc/utils/linalg.cc']
    for name in names:
        p = src / name
        sha = digest(p)
        if sha != manifest['sources'][name]:
            raise RuntimeError(f'Baseline source differs from actual build manifest: {name}')
        dest = out / 'baseline_source' / name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        sources[name] = sha
    for name in ('runtime.py', 'model.py', 'heads.py', 'optimized_heads.py', 'optimize_study.py'):
        sources[name] = digest(ROOT / name)
    raw, tiles = [], {}
    for folder in sorted((ROOT / 'out/13/confirm/log').glob('confirm-r*-*/')):
        meta = json.loads((folder / 'run.json').read_text())
        if meta['status'] != 'complete': continue
        for record, profile in zip(meta['measurements'], meta['profiles']):
            if record['variant'] != 'baseline': continue
            for party in (0, 1):
                path = folder / 'log' / f"native-p{party}-e{record['native_execution']}.tsv"
                events = load_events(path)
                pack = [e for e in events if e['kind'] == 'packing']
                mats = sorted([e for e in events if e['kind'] == 'HAL' and e['name'] in ('f_mmul','f_batch_mmul')], key=lambda x:x['start_ns'])
                labels = ['Q','K','QK','V','AV','output_projection','FFN1','FFN2']
                if len(mats) != len(labels): raise RuntimeError('Unexpected baseline matrix graph')
                row = dict(case=meta['config']['plan_name'], session=meta['config']['session'],
                    pair=record['pair'], party=party, native_s=profile[party]['seconds'],
                    packing_union_s=union_seconds(pack),
                    matrix_inclusive_s=sum((e['end_ns']-e['start_ns'])/1e9 for e in mats),
                    evidence=str(path.relative_to(ROOT)))
                for label,m in zip(labels,mats):
                    inside = [e for e in pack if m['start_ns'] <= e['start_ns'] and e['end_ns'] <= m['end_ns']]
                    row[label+'_pack_s'] = union_seconds(inside)
                raw.append(row)
    import re
    pat = r'(\d+)@(\d+)x(\d+)x(\d+) split as (\d+)x(\d+)x(\d+).*CtPack \(rlwe\)'
    for m in re.finditer(pat, (ROOT/'out/13/confirm/log/confirm-r08-S10R2/log/b1-p0.log').read_text()):
        b,m0,k,n,d0,d1,d2 = map(int,m.groups())
        key = (b,m0,k,n)
        blocks = [(m0+d0-1)//d0,(k+d1-1)//d1,(n+d2-1)//d2]
        result_polys = b*blocks[0]*blocks[2]
        packed = (result_polys+d1-1)//d1
        tiles[key] = dict(batch=b,m=m0,k=k,n=n,tile_m=d0,tile_k=d1,tile_n=d2,
            lhs_polys_per_batch=blocks[0]*blocks[1],rhs_polys_per_batch=blocks[1]*blocks[2],
            result_polys=result_polys,response_ciphertexts=packed,gap=d1,
            useful_output_coefficients=b*m0*n,allocated_response_coefficients=packed*8192,
            output_coefficient_utilization=b*m0*n/(packed*8192),
            old_parallel_launches_upper_bound=packed*(1+d1.bit_length()-1),
            proposed_parallel_launches=1+d1.bit_length()-1,
            evidence='out/13/confirm/log/confirm-r08-S10R2/log/b1-p0.log')
    def save(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('historical_packing_samples.csv',raw)
    groups=defaultdict(list)
    for row in raw:groups[(row['case'],row['party'])].append(row)
    summary=[]
    for (case,party),rows in sorted(groups.items()):
        s=dict(case=case,party=party,calls=len(rows),independent_runtimes=len(set(r['session'] for r in rows)))
        for name in rows[0]:
            if name.endswith('_s'):s[name+'_mean']=statistics.mean(r[name] for r in rows)
        s['packing_fraction']=s['packing_union_s_mean']/s['native_s_mean'];summary.append(s)
    save('historical_packing_summary.csv',summary)
    save('actual_matrix_packing.csv',list(tiles.values()))
    candidates=[
        dict(priority=1,direction='跨返回密文组的蝶形树批处理',class_='通用 HE 打包；可加速异构配置公共阶段',
            evidence='packlwes.cc:144,174 + bumblebee_dot.cc:482；实测 packing_union_s 见表',
            expected_effect='减少并行区启动与末层闲置；密文数、模数、旋转数量、数学结果保持相同；通信无结构性下降',
            risk='必须消除输出覆盖其他输入组的别名；同层屏障、尾组零填充；CPU 分支测试精确密文等价',decision='本轮首先实现；Block 净收益待实测'),
        dict(priority=2,direction='QKV 横向合并投影',class_='通用私有权重投影',
            evidence='同一归一化输入的三次 197x768@768x768；IR 查询缩放位于投影之后',
            expected_effect='潜在减少重复输入加密与交互；仍须全部三组输出与原有截断',
            risk='拼接会改变选块与加法/截断调度；公平验证必需；不属于异构专有',decision='候选；先验证更低风险打包方向'),
        dict(priority=3,direction='BatchDotOLE 多头接收/编码/计算流水批处理',class_='通用多头矩阵阶段',
            evidence='bumblebee_dot.cc:812,837 逐头循环；内部已有并行编码与 HE 乘法',
            expected_effect='可能减少小任务启动，但单头内部已占满部分线程；跨头可能抢占双向 OLE',
            risk='链路消息顺序、密钥初始化、双向任务线程共享；未量化分阶段预算',decision='保留，不能仅凭循环声称瓶颈'),
        dict(priority=4,direction='异构 S/R 非线性 OT 分块与乘法阈值',class_='异构组大小相关；协议也可通用',
            evidence='state.cc InitOTState min2048,maxinstances；MulAA <4096 使用 Beaver，否则 HE OLE',
            expected_effect='小组归一化可能跨越协议阈值；比较组同时有不同任务量和交互成本',
            risk='不得改变位宽、近似比较、随机数或重复使用 Beaver；目前仅相关性，缺少因果验证',decision='审计定位；本轮不修改密码协议'),
        dict(priority=5,direction='reciprocal 因子的重复 B2A 缓存',class_='通用固定点非线性',
            evidence='fxp_base.cc:165 factor 被两次 f_mul 消费；out/13 全部 B2A union 约0.09–0.10秒',
            expected_effect='实际可删转换，但全部 B2A 预算不足 Block 的0.4%；局部调用削减不保证净收益',
            risk='不能把类型内部 B2A 当作新增 HE/MPC 头类型转换',decision='低优先级，不用其宣称端到端收益')]
    save('candidates.csv',candidates)
    identity=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='audit_frozen_before_native_changes',
        original_cpu_sha256=digest(ROOT/'build/spu/libspu.so'),observer_sha256=digest(ROOT/'build/stage2.so'),
        observer_manifest_sha256=digest(ROOT/'build/stage2.json'),sources=sources,
        historical_real_baseline_calls=len(raw)//2,selected='CPU cross-group RLWE butterfly batching')
    write(out/'audit_identity.json',identity)
    def table(rows):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in rows[0])+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{v:.5f}' if isinstance(v,float) else str(v))+'</td>' for v in r.values())+'</tr>' for r in rows)+'</table>'
    report='''<!doctype html><meta charset="utf-8"><title>BumbleBee-ViT 底层源码审计</title>
<style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}code{background:#eee}h2{color:#18546b}</style>
<h1>BumbleBee-ViT 底层计算与打包源码审计</h1>
<p>此文件在核心代码变更前冻结。性能依据为 out/13 已完成的 48 次原始图真实两方 Block 执行，8 个独立运行时/配置、每运行时2次；不能视作48个独立样本。源码逐文件 SHA 与实际 stage2.so 构建清单核对，原始 CPU 库另行保留。选择优化不等于证明收益。</p>
<h2>实际执行与来源</h2>
<p>run.py → runtime.baseline/node/worker → optimize_study.paired_worker → 原始 Flax ViT Layer → heads.bind → JAX/PPHLO → HAL → protocol.cc 注册的 BumbleBee 内核。参数来自同一 safetensors；P0 私有权重 Priv2k，Block 激活 AShr 两方加法分享；全模型输入为 P1 私有。实际 FM64/f18，GELU 原有 FM32/f12，Exp 6次迭代。实验为 CPU SEAL，Phantom GPU 桥接未启用；GPU 源码存在不能证明本实验使用 GPU。</p>
<p>原有 BumbleBee 内核包括双向 OLE、packing、模数调整/随机掩码、OT 比较、截断、Beaver 缓存。当前仓库已有 Phantom GPU 适配与观察器；S/R/L 头计划、R 分母零值保护、L alpha、公有 gather/slices、联合 reciprocal 是此 ViT 项目层实现。当前私有构建源码对照的是本地 OpenBumbleBee 工作副本，不能用该副本的相同哈希证明最早论文发行版来源。无远程历史版本的未核查差异不作结论。</p>
<h2>计算协议、内核与数据布局</h2>
<p>Q/K/V、输出投影和两层 FFN：共享激活×P0私有权重，MatMulAV 做一个 HE OLE 交叉项，加本方 ring_mmul。QK/AV：BatchMatMulAA 对 (x0+x1)(y0+y1) 做两条链路并发 OLE，补本地环矩阵乘。实际每 Block 6次普通矩阵乘、2次12头批矩阵乘、10次双方合计 dot_H2A（P0 8、P1 2）。环矩阵乘使用 linalg::matmul；HE 明文×密文乘和打包通过 PhantomEvaluator 的 CPU seal::Evaluator 分支。</p>
<p>矩阵使用 CKKS scheme 的 RLWE 容器，但 VectorEncoder 用环整数的 ModulusUp/Centralize 系数编码、scale=1；不是把普通 CKKS 浮点 SIMD 8192/2 slots 当成这里的布局。FM64 N=8192、Q primes bits={59,55,49}，编码3个模数、解码2个模数。子矩阵按 d0×d1×d2 放入系数，输出间隔 gap=d1，先计算结果 RLWE，再蝶形自动同构合并 gap 个结果密文；不同头输出可以横跨同一返回密文。之后原有 H2A 使用随机掩码与公钥零加密，解密/模数还原为 AShr。安全参数、噪声处理、SEALContext 原有 sec_level 设置不在本轮改动范围。</p>
<p>秘密张量是 NdArrayRef 的环元素、shape/strides；转置常为视图，编码按真实 strides 读值，部分 reshape/clone 会复制。Head tensor [1,12,197,64] 在 QK/AV 为 [12,197,64]、[12,64,197] 等。公共索引 gather 不调用密码协议但逐元素组织开销可见。joint 一次恢复逻辑顺序再做12头 AV；split 按类型多次 AV，可增加密文尾组与调用成本。</p>
<p>S=安全 max 比较归约→exp→行和→reciprocal→乘法；R=安全比较/OT multiplexer→行和→安全 den&gt;0 guard→reciprocal→乘法；L=原有 alpha*z，仍执行197² QK 与 AV。比较在 compare.cc/MsbA2B + CompareProtocol/Millionaire OT 中完成；InitOTState 以2048任务为最小块，最多实例数为编译常量。大元素乘用 BFV RLWE OLE，N=4096、多 CRT 明文域与随机掩码；小于4096元素时用安全生成并消耗一次的 Beaver。平方另有原有对称优化。A2B/B2A 与截断属于上述协议内部，未出现 S/R/L 类型之间新增独立 HE↔MPC 域转换。</p>
<h2>实际打包与填充证据</h2>
<p>下表子形状来自执行日志，不是估计。utilization 是有效返回系数/分配返回系数，不是完整协议计算利用率；稀疏编码间隔有正确性用途，不能简单删零。GetSubMatShape 静态代价 bandwidth=200,cpu=1，没有纳入真实网络/CPU或 BatchDotOLE 的全局 batch_size。已有全局返回打包已经将所有头连续合并，尾组浪费比每头独立返回小。</p>'''
    report+=table(list(tiles.values()))
    report+='''<h2>实测瓶颈与混合影响边界</h2><p>下面 packing_union 是实际 native packing 区间并集，已包含在 matrix 中，不能与 matrix 再相加。同一 Block P0/P1 固定观察；双方时间用 max，两方发送字节之和才是链路 payload。CPU-seconds（线程累计）与 wall time 不可相加；Recv 阻塞可能是等对方计算，不等于纯网络。</p>'''+table(summary)
    report+='''<p>三种 joint 头配置 QK 与 AV 的形状、两向 OLE、选块和返回密文数一致。异构主要改变非线性组大小、比较/归一化/组织和小算子协议阈值，没有证据证明 joint 密文矩阵布局因类型混合退化。此前 gather/reciprocal 对照没有通过预先设定净收益标准。历史矩阵+GELU约占 Block 53–61%；候选打包耗时是较大的公共计算预算。</p>
<p>已定位的实际低效：bumblebee_dot.cc 对每一个返回组顺序调用 PackingWithModulusDrop；每组独立 preprocessing 及 log2(gap) 次 parallel_for，最后一层只有1个任务。QK gap32 有74个返回组；AV gap128 有19组。YACL parallel_for 默认 grain=1，嵌套区执行串行。候选跨组同层批处理保留每个蝶形完全相同操作顺序/同层屏障，增加最后几层可并行任务，减少调度启动。原实现输出写到输入数组前部，若直接把外循环改 parallel_for 会发生跨组覆盖竞态；必须独立输出后再移动。</p>
<h2>候选与实施门槛</h2>'''+table(candidates)
    report+='''<p>本轮首先实现返回密文跨组批处理；它属于通用打包加速，在异构配置上的作用是降低公共成本，不能声称消除了所有混合非加性成本。新实现先做同一输入密文逐系数精确相等测试（含尾组），再做同 IR/同逻辑头的真实两方 AB/BA。只有 Block 重复测量净收益清晰，才启动12层模型；否则报告局部上限与失败。其结构不改变传输密文数，所以通信下降不是预期结果。QKV融合与多头流水等继续保留为未验证候选。</p>
<p>证据入口：baseline_source/（实际构建源码快照）；audit_identity.json（库与源码SHA）；historical_packing_samples.csv（逐调用区间）；historical_packing_summary.csv；actual_matrix_packing.csv；candidates.csv；out/13 编译图与原始 TSV。后续实现和真实实验另写结果报告，此冻结审计不改成事后预测。</p>'''
    (out/'audit_report.html').write_text(report)
    print(json.dumps(identity,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');args=p.parse_args()
    audit(args.out)
