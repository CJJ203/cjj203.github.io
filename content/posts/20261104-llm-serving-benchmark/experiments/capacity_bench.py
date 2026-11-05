"""SLO-constrained concurrency/length search using official sglang.bench_serving.

Runs a persistent CPU HTTP benchmark client, capturing exact per-request E2E
before official aggregation and sampling resources. Does not change the server.
Requires psutil plus the same dependencies as SGLang's serving benchmark.
"""
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import numpy as np
from resource_monitor import ResourceMonitor,summarize_resources


SLO={'min_request_decode_tps':50.,'p99_ttft_ms':500.,'p99_e2e_ms':10000.,'p99_itl_ms':100.}


def get_json(path):
    with urllib.request.urlopen('http://127.0.0.1:30000'+path,timeout=30) as r:return json.load(r)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--dataset-dir',required=True,type=Path)
    parser.add_argument('--tokenizer',required=True)
    parser.add_argument('--gpu-uuid',default='GPU-df8fc299-cd4b-e00f-ac95-1f279395bb0d')
    parser.add_argument('--server-pid',type=int,default=1785345)
    parser.add_argument('--scheduler-pid',default='1785577')
    parser.add_argument('--one-case',action='store_true')
    parser.add_argument('--concurrency',type=int,default=24)
    parser.add_argument('--input-len',type=int,default=1024)
    parser.add_argument('--output-len',type=int,default=128)
    parser.add_argument('--num-prompts',type=int)
    options=parser.parse_args()
    output=options.output_dir
    output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    from transformers import PreTrainedTokenizerBase
    original_template=PreTrainedTokenizerBase.apply_chat_template
    def template(self,*args,**kwargs):
        kwargs.setdefault('return_dict',False)
        return original_template(self,*args,**kwargs)
    PreTrainedTokenizerBase.apply_chat_template=template
    # bench_serving is the supported legacy alias for this exact implementation.
    import sglang.bench_serving as official
    import sglang.benchmark.serving as impl
    original_metrics=impl.calculate_metrics
    captured=[]
    def metrics(*args,**kwargs):
        records=kwargs.get('outputs',args[1] if len(args)>1 else [])
        captured[:] = [dict(success=o.success,error=o.error,start=o.start_time,
            e2e_ms=o.latency*1000,ttft_ms=o.ttft*1000,output_tokens=o.output_len,
            tpot_ms=(o.latency-o.ttft)*1000/(o.output_len-1) if o.output_len>1 else None,
            decode_tps=(o.output_len-1)/(o.latency-o.ttft) if o.latency>o.ttft and o.output_len>1 else None,
            itl_ms=[v*1000 for v in o.itl]) for o in records]
        return original_metrics(*args,**kwargs)
    impl.calculate_metrics=metrics
    def timed_out(signum,frame):raise TimeoutError('Case exceeded 900 seconds; stop the plan')
    signal.signal(signal.SIGALRM,timed_out)
    lock_dir=Path.home()/'.cache/gpu-locks';lock_dir.mkdir(parents=True,exist_ok=True)
    cases={}
    manifest=[]
    with (lock_dir/(options.gpu_uuid+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert get_json('/v1/models')['data'][0]['id']=='Qwen/Qwen3-0.6B'
        (output/'server-info.json').write_text(json.dumps(get_json('/get_server_info'),indent=2))
        (output/'slo.json').write_text(json.dumps(SLO,indent=2))

        def run_case(name,c,ilen=1024,olen=128,count=None,stage='concurrency'):
            if name in cases:return cases[name]
            n=count or max(128,6*c)
            repetitions=[]
            for repeat in [1,2,3]:
                processes=subprocess.check_output(['nvidia-smi','-i',options.gpu_uuid,
                    '--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
                if {r.split(',')[0].strip() for r in processes.splitlines() if r.strip()}!={options.scheduler_pid}:
                    raise RuntimeError('GPU processes changed: '+processes)
                tag=f'{name}-rep{repeat}'
                args=['--backend','sglang-oai-chat','--base-url','http://127.0.0.1:30000',
                    '--model',options.tokenizer,'--tokenizer',options.tokenizer,
                    '--served-model-name','Qwen/Qwen3-0.6B','--dataset-name','openai',
                    '--dataset-path',str(options.dataset_dir/f'chat-{ilen}.jsonl'),
                    '--sharegpt-output-len',str(olen),'--num-prompts',str(n),
                    '--request-rate','inf','--max-concurrency',str(c),
                    '--extra-request-body','{"stream_options":{"include_usage":true},"ignore_eos":true}',
                    '--seed','42','--temperature','0','--warmup-requests','8','--flush-cache',
                    '--output-details','--disable-tqdm','--output-file',str(output/(tag+'.jsonl'))]
                entry=dict(tag=tag,case=name,stage=stage,repeat=repeat,concurrency=c,
                    input_len=ilen,output_len=olen,num_prompts=n,args=args,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_processes=processes)
                manifest.append(entry)
                with (output/'manifest.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                print('START',tag,'C',c,'L',ilen,olen,'N',n,flush=True)
                monitor=ResourceMonitor(options.gpu_uuid,options.server_pid)
                captured.clear();monitor.start()
                sys.argv=['sglang.bench_serving']+args
                try:
                    signal.alarm(900)
                    with (output/(tag+'.log')).open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        official.cli_main()
                finally:
                    signal.alarm(0);monitor.stop()
                r=json.loads((output/(tag+'.jsonl')).read_text())
                assert len(captured)==n
                valid=[v for v in captured if v['success']]
                start=min(v['start'] for v in valid)
                end=max(v['start']+v['e2e_ms']/1000 for v in valid)
                resources=summarize_resources(monitor.samples,start,end)
                (output/(tag+'-resources.json')).write_text(json.dumps(dict(summary=resources,samples=monitor.samples),indent=2))
                (output/(tag+'-requests.json')).write_text(json.dumps(captured))
                min_rate=min(v['decode_tps'] or 0 for v in valid)
                slow=sum((v['decode_tps'] or 0)<SLO['min_request_decode_tps'] for v in valid)
                passed=(r['completed']==n and slow==0 and r['p99_ttft_ms']<=SLO['p99_ttft_ms']
                    and r['p99_e2e_latency_ms']<=SLO['p99_e2e_ms'] and r['p99_itl_ms']<=SLO['p99_itl_ms'])
                record=dict(**entry,passed=passed,slow_requests=slow,min_request_decode_tps=min_rate,
                    metrics={k:v for k,v in r.items() if not isinstance(v,(list,dict)) and k!='request_rate'},resources=resources)
                repetitions.append(record)
                with (output/'runs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('DONE',tag,'pass',passed,'min_tps',round(min_rate,2),
                    'p99_ttft',round(r['p99_ttft_ms'],2),'p99_e2e',round(r['p99_e2e_latency_ms'],2),flush=True)
            group=dict(case=name,stage=stage,concurrency=c,input_len=ilen,output_len=olen,
                num_prompts=n,passed=all(r['passed'] for r in repetitions),runs=repetitions)
            cases[name]=group
            (output/'cases.json').write_text(json.dumps(cases,indent=2))
            return group

        if options.one_case:
            run_case('manual',options.concurrency,options.input_len,options.output_len,options.num_prompts,'manual')
            return
        coarse=[1,8,16,24,32,48,64,96,128]
        for c in coarse:run_case(f'c{c}',c)
        def best():
            return max(g['concurrency'] for g in cases.values() if g['stage']=='concurrency' and g['passed'])
        peak=best()
        failing=sorted(g['concurrency'] for g in cases.values() if g['stage']=='concurrency' and not g['passed'] and g['concurrency']>peak)
        if failing:
            upper=failing[0]
            for c in range(peak+4,upper,4):run_case(f'c{c}',c)
            peak=best()
            upper=min(g['concurrency'] for g in cases.values() if g['stage']=='concurrency' and not g['passed'] and g['concurrency']>peak)
            for c in range(peak+1,upper):run_case(f'c{c}',c)
            peak=best()
        safe=max(1,math.floor(peak*.8))
        while not run_case(f'safe-c{safe}',safe,stage='safe-baseline')['passed']:
            safe=max(1,safe-4)
        input_passed=[1024]
        for ilen in [128,2048,4096,8192]:
            g=run_case(f'in{ilen}-c{safe}',safe,ilen,stage='input-length')
            if g['passed']:input_passed.append(ilen)
            elif ilen>1024:break
        output_passed=[128]
        for olen in [256,512,1024,2048]:
            g=run_case(f'out{olen}-c{safe}',safe,1024,olen,stage='output-length')
            if g['passed']:output_passed.append(olen)
            else:break
        ilen=max(input_passed);olen=max(output_passed)
        while not run_case(f'joint-{safe}-{ilen}-{olen}',safe,ilen,olen,stage='joint')['passed']:
            lower=sorted(v for v in input_passed if v<ilen)
            if lower:ilen=lower[-1]
            else:
                lower=sorted(v for v in output_passed if v<olen)
                if not lower:raise RuntimeError('Joint profile unexpectedly fails baseline')
                olen=lower[-1]
        validation=None
        for attempt in range(3):
            validation=run_case(f'validation-{safe}-{ilen}-{olen}',safe,ilen,olen,
                count=max(512,12*safe),stage='validation')
            if validation['passed']:break
            safe=max(1,math.floor(safe*.8))
        decision=dict(peak_concurrency=peak,recommended_concurrency=safe,
            recommended_input_tokens=ilen,recommended_output_tokens=olen,
            validation_passed=validation['passed'],validation_case=validation['case'],slo=SLO,
            scope='maximum verified sampled concurrency; not exhaustive all-integer/global optimum',
            cases=len(cases),runs=sum(len(g['runs']) for g in cases.values()),
            requests=sum(g['num_prompts']*3 for g in cases.values()))
        (output/'decision.json').write_text(json.dumps(decision,indent=2))
        print('DECISION',json.dumps(decision),flush=True)


if __name__=='__main__':main()
