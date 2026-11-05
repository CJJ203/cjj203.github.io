"""Validate all per-request SLO/resource evidence and build capacity figures.

Usage: python summarize_capacity.py /path/to/blog-capacity-results
Requires numpy and matplotlib. Writes capacity-summary.json and capacity-runs.json
beside this script; figures to ../assets/capacity*.svg/png.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np


def summarize(raw,destination):
    cases=json.loads((raw/'cases.json').read_text())
    decision=json.loads((raw/'decision.json').read_text())
    slo=decision['slo']
    summaries={};runs=[]
    for name,case in cases.items():
        assert len(case['runs'])==3
        assert {r['repeat'] for r in case['runs']}=={1,2,3}
        for r in case['runs']:
            details=json.loads((raw/(r['tag']+'-requests.json')).read_text())
            official=json.loads((raw/(r['tag']+'.jsonl')).read_text())
            # Published archives omit this duplicate array; the untouched official
            # JSONL stores the same ITL values in seconds and request order.
            for i,v in enumerate(details):
                if 'itl_ms' not in v:
                    v['itl_ms']=[t*1000 for t in official['itls'][i]]
            m=r['metrics'];n=case['num_prompts']
            assert len(details)==n and all(v['success'] and not v['error'] for v in details)
            assert m['completed']==n
            assert set(official['input_lens'])=={case['input_len']}
            assert set(official['output_lens'])=={case['output_len']}
            assert m['total_input_tokens']==n*case['input_len']
            assert m['total_output_tokens']==n*case['output_len']
            assert math.isclose(m['output_throughput'],m['total_output_tokens']/m['duration'],rel_tol=1e-10)
            for key,field in [('p99_e2e_latency_ms','e2e_ms'),('p99_ttft_ms','ttft_ms'),('p99_tpot_ms','tpot_ms')]:
                assert np.isclose(m[key],np.percentile([v[field] for v in details],99))
            assert np.isclose(m['p99_itl_ms'],np.percentile([t for v in details for t in v['itl_ms']],99))
            for v in details:
                assert np.isclose(v['decode_tps'],(v['output_tokens']-1)/((v['e2e_ms']-v['ttft_ms'])/1000))
            minimum=min(v['decode_tps'] for v in details)
            assert np.isclose(minimum,r['min_request_decode_tps'])
            slow=sum(v['decode_tps']<slo['min_request_decode_tps'] for v in details)
            assert slow==r['slow_requests']
            expected=(slow==0 and m['p99_ttft_ms']<=slo['p99_ttft_ms']
                      and m['p99_e2e_latency_ms']<=slo['p99_e2e_ms'] and m['p99_itl_ms']<=slo['p99_itl_ms'])
            assert r['passed']==expected
            resource=json.loads((raw/(r['tag']+'-resources.json')).read_text())
            start=min(v['start'] for v in details)
            end=max(v['start']+v['e2e_ms']/1000 for v in details)
            valid=[v for v in resource['samples'] if start<=v['time']<=end and 'error' not in v]
            assert len(valid)==resource['summary']['samples'] and len(valid)>=3
            assert resource['summary']['sample_errors']==0
            for key in ['gpu_util','gpu_used_mib','gpu_power_w','server_cpu_percent','client_cpu_percent','server_rss_gib']:
                assert np.isclose(max(v[key] for v in valid),r['resources'][key]['max'])
                assert np.isclose(np.mean([v[key] for v in valid]),r['resources'][key]['mean'])
            runs.append(r)
        rr=case['runs']
        summary={k:v for k,v in case.items() if k!='runs'}
        summary.update(requests=sum(r['num_prompts'] for r in rr),
            min_request_decode_tps=min(r['min_request_decode_tps'] for r in rr),
            slow_requests=sum(r['slow_requests'] for r in rr),
            output_tps=statistics.median(r['metrics']['output_throughput'] for r in rr),
            request_tps=statistics.median(r['metrics']['request_throughput'] for r in rr),
            mean_ttft_ms=statistics.median(r['metrics']['mean_ttft_ms'] for r in rr),
            p99_ttft_ms=max(r['metrics']['p99_ttft_ms'] for r in rr),
            p99_e2e_ms=max(r['metrics']['p99_e2e_latency_ms'] for r in rr),
            p99_itl_ms=max(r['metrics']['p99_itl_ms'] for r in rr),
            min_run_sec=min(r['metrics']['duration'] for r in rr),
            max_run_sec=max(r['metrics']['duration'] for r in rr))
        for key in ['gpu_util','gpu_used_mib','gpu_power_w','server_cpu_percent','client_cpu_percent',
                    'server_rss_gib','client_rss_gib','host_cpu_percent','host_available_gib']:
            summary[key+'_mean']=statistics.median(r['resources'][key]['mean'] for r in rr)
            summary[key+'_max']=max(r['resources'][key]['max'] for r in rr)
            summary[key+'_min']=min(r['resources'][key]['min'] for r in rr)
        summary['gpu_free_mib_min']=min(r['resources']['gpu_total_mib']['min']-r['resources']['gpu_used_mib']['max'] for r in rr)
        assert summary['passed']==all(r['passed'] for r in rr)
        summaries[name]=summary
    assert len(runs)==decision['runs']
    assert sum(r['num_prompts'] for r in runs)==decision['requests']
    assert decision['peak_concurrency']==max(c['concurrency'] for c in summaries.values() if c['stage']=='concurrency' and c['passed'])
    final=summaries[decision['validation_case']]
    assert final['passed'] and final['input_len']==decision['recommended_input_tokens'] and final['output_len']==decision['recommended_output_tokens']
    result=dict(decision=decision,cases=summaries)
    (destination/'capacity-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (destination/'capacity-runs.json').write_text(json.dumps(runs,indent=2,allow_nan=False)+'\n')
    print('Validated',len(runs),'runs;',decision['requests'],'requests;',len(summaries),'cases')
    print(json.dumps(decision,indent=2))
    return result


def plots(result,assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    cases=sorted([v for v in result['cases'].values() if v['stage']=='concurrency'],key=lambda v:v['concurrency'])
    x=[v['concurrency'] for v in cases]
    slo=result['decision']['slo']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,label,limit in zip(axes,['output_tps','min_request_decode_tps','p99_ttft_ms'],
            ['Output throughput (tok/s)','Slowest request decode rate (tok/s)','Worst-run P99 TTFT (ms)'],[None,slo['min_request_decode_tps'],slo['p99_ttft_ms']]):
        y=[v[key] for v in cases]
        ax.plot(x,y,color='#64748b',alpha=.6)
        for passed,color,label2 in [(True,'#147d92','Pass'),(False,'#c45e3b','Fail')]:
            ax.scatter([a for a,c in zip(x,cases) if c['passed']==passed],
                       [v[key] for v in cases if v['passed']==passed],color=color,s=30,label=label2)
        if limit:ax.axhline(limit,color='#b45309',ls='--',lw=1,label=f'SLO {limit}')
        ax.set_xlabel('Client concurrency');ax.set_ylabel(label)
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
    axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=slo['min_request_decode_tps']*.6)
    axes[-1].legend(fontsize=8)
    fig.suptitle('Qwen3-0.6B / RTX 5090 / Chat Completions\n1024 input + 128 output; all 3 runs must pass all SLO gates')
    for ext in ['svg','png']:fig.savefig(assets/f'capacity-concurrency.{ext}',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].plot(x,[v['gpu_util_mean'] for v in cases],'-o',color='#147d92',label='Mean GPU util')
    axes[0].plot(x,[v['gpu_util_max'] for v in cases],'--',color='#147d92',alpha=.6,label='Sampled peak')
    axes[0].set_ylabel('GPU utilization (%)');axes[0].set_ylim(0,105);axes[0].legend(fontsize=8)
    axes[1].plot(x,[v['gpu_used_mib_max']/1024 for v in cases],'-o',color='#7251a1')
    axes[1].axhline(32607/1024,ls='--',color='#999',lw=1,label='Device total')
    axes[1].set_ylabel('Peak occupied VRAM (GiB)');axes[1].set_ylim(0,34);axes[1].legend(fontsize=8)
    axes[2].plot(x,[v['server_cpu_percent_mean'] for v in cases],'-o',label='Serving process tree',color='#147d92')
    axes[2].plot(x,[v['client_cpu_percent_mean'] for v in cases],'-o',label='Benchmark client',color='#bf6336')
    axes[2].set_ylabel('CPU utilization (100% = 1 logical core)');axes[2].set_ylim(bottom=0);axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Client concurrency');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Resource samples inside formal request windows\n0.5 s polling; means = median of run means; peaks = maximum sampled value')
    for ext in ['svg','png']:fig.savefig(assets/f'capacity-resources.{ext}',dpi=150)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw_dir',type=Path);a=p.parse_args()
    destination=Path(__file__).resolve().parent
    result=summarize(a.raw_dir,destination)
    assets=destination.parent/'assets';assets.mkdir(exist_ok=True)
    plots(result,assets)


if __name__=='__main__':main()
