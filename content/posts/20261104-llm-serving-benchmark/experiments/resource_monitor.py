"""Sample an existing serving GPU and process tree without allocating a GPU."""
import os
import subprocess
import threading
import time
import psutil


class ResourceMonitor:
    def __init__(self,gpu_uuid,server_pid,interval=0.5):
        self.uuid=gpu_uuid
        self.server=psutil.Process(server_pid)
        self.client=psutil.Process(os.getpid())
        self.interval=interval
        self.samples=[]
        self.stop_event=threading.Event()
        self.previous={}
        self.thread=threading.Thread(target=self.run,daemon=True)

    def cpu(self,name,processes,now):
        total=0.;rss=0
        for process in processes:
            try:
                times=process.cpu_times();total+=times.user+times.system
                rss+=process.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
        previous=self.previous.get(name)
        self.previous[name]=(now,total)
        percent=0. if previous is None else max(0.,100*(total-previous[1])/(now-previous[0]))
        return percent,rss/1024**3

    def run(self):
        psutil.cpu_percent()
        while not self.stop_event.is_set():
            start=time.perf_counter()
            try:
                line=subprocess.check_output(['nvidia-smi','-i',self.uuid,
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
                    '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                gpu,memutil,used,total,power,temp=[float(s.strip()) for s in line.split(',')]
                now=time.perf_counter()
                server_cpu,server_rss=self.cpu('server',[self.server]+self.server.children(recursive=True),now)
                client_cpu,client_rss=self.cpu('client',[self.client],now)
                vm=psutil.virtual_memory()
                self.samples.append(dict(time=now,gpu_util=gpu,gpu_mem_util=memutil,
                    gpu_used_mib=used,gpu_total_mib=total,gpu_power_w=power,gpu_temp_c=temp,
                    server_cpu_percent=server_cpu,server_rss_gib=server_rss,
                    client_cpu_percent=client_cpu,client_rss_gib=client_rss,
                    host_cpu_percent=psutil.cpu_percent(),host_available_gib=vm.available/1024**3))
            except Exception as e:
                self.samples.append(dict(time=time.perf_counter(),error=str(e)))
            self.stop_event.wait(max(0.,self.interval-(time.perf_counter()-start)))

    def start(self):self.thread.start()

    def stop(self):
        self.stop_event.set();self.thread.join(timeout=8)


def summarize_resources(samples,start,end):
    rows=[s for s in samples if start<=s['time']<=end and 'error' not in s]
    if not rows:
        raise RuntimeError('No valid resource sample inside the formal request window')
    result={'samples':len(rows),'sample_interval_sec':0.5,'formal_window_sec':end-start}
    for key in rows[0]:
        if key!='time':
            values=[r[key] for r in rows]
            result[key]={'mean':sum(values)/len(values),'min':min(values),'max':max(values)}
    result['sample_errors']=sum('error' in s and start<=s['time']<=end for s in samples)
    return result
