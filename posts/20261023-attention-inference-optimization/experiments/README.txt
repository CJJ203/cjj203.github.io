Attention 推理优化：复现实验
==========================

文章：../index.md
日期：2026-10-23（Asia/Shanghai）

GPU 实测环境
------------
NVIDIA H800，compute capability 9.0，单卡
NVIDIA driver 580.159.04
Python 3.12.14
PyTorch 2.13.0+cu130 / CUDA runtime 13.0
Triton 3.7.1
Nsight Compute 2025.3.1.0

源码阅读使用 vLLM v0.11.0、SGLang v0.5.4 的发布标签；实验没有启动这两个
版本的服务，也没有测模型的端到端吞吐。GPU 代码使用实验机已有环境，
不要求安装 vLLM 或 SGLang。

运行
----
先确认 GPU 空闲并设置 CUDA_VISIBLE_DEVICES。共享机器上请遵循机器的
资源分配及合作锁约定。以下命令在本目录执行：

  python bench_attention.py > attention-results.json
  python check_softmax.py > softmax-results.json
  python -m unittest -v test_prefix_cache.py
  python prefix_cache.py > prefix-results.json
  python check_vllm_blocks.py > vllm-block-results.json

前缀缓存只需 Python >= 3.10 标准库。整数 slot 模拟 KV 索引，不执行模型。
每条请求 acquire 后视为已经填入新 KV，随即 release；不是并发服务实验。
管理器能够返回完整 KV 前缀；真实生成仍可能需要重算末 token 获取 logits。
本次统计负载没有重复的完整请求，每条都有新的后缀。

vLLM 块管理实验
--------------
check_vllm_blocks.py 只需 Python >= 3.10 标准库。默认从 vLLM v0.11.0
下载 kv_cache_utils.py、block_pool.py，并校验脚本中固定的 SHA-256。
也可使用已有源码，避免下载：

  python check_vllm_blocks.py --source-dir /path/to/vllm/v1/core > vllm-block-results.json

脚本通过 AST 提取并执行原文件中的块类型、块 hash 类型、组 ID 打包函数、
空闲队列、hash 索引与 BlockPool 定义；不执行其引擎导入或日志初始化，
不修改被提取定义的方法实现。KV 事件关闭。实验只运行 CPU 元数据管理，
不分配 GPU KV 张量、不执行 attention 或测量服务吞吐。

本次在本机 Python 3.10.17 上，使用此前下载的原版本源码离线运行。
池中 9 个块，其中编号 0 为 null block，8 个可分配。A 长 37、B 长 40，
block size=16，前 32 token 共享。按分配、共享、释放、再次 touch、重新
分配全部块的顺序核对引用计数、空闲队列与缓存 hash。
容量对照中的固定预留 128 是人为设定的基线，槽位数按长度和块数计算，
不是某个连续分配器的性能测量。

Attention 范围：FP16 contiguous [B,H,N,D]，causal MHA，forward only。
无 GQA、dropout、padding mask、变长 batch 或任意 stride。
在线 Python 版本使用 FP32，主要用于解释和数值验证。
check_softmax.py 单独验证在线归一化的最大值递增、递减、大跨度与尾块。
它可在 CPU 上计算，但导入 attention.py 时仍需安装 PyTorch 与 Triton。
Triton 累加器为 FP32，P 在乘 V 时转 FP16。

计时：warmup 10 次；CUDA Graph 捕获 20 次调用，重复 7 组 CUDA Event 计时，
除以 20 后取中位数。反复使用同一批输入，属于热数据的重复执行测试。
内存：图捕获前，warmup 后清理缓存，记录一次调用的增量 peak allocated；
包含输出，不包含已有输入，不是 nvidia-smi 占用，也不是 reserved。
基准不手工锁频；随机种子 20261023。JSON 包含每组原始样本和误差。

Nsight Compute
--------------
确保 ncu 在 PATH 中。以下为实际采集使用的参数（换成你自己的 python）：

  ncu --profile-from-start off --target-processes all \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv python bench_attention.py --profile flash128 > ncu-flash128.csv

将 flash128 替换为 flash64 或 naive，分别采集。未显式设置 replay、
cache-control 或 clock-control，采用 NCU 默认值。普通计时与 profiler
的时钟/缓存条件不同，分别解读，不跨表计算加速比。

本次普通账户访问计数器返回 ERR_NVGPUCTRPERM；确认已有 sudo -n 权限后，
通过 sudo -n env CUDA_VISIBLE_DEVICES=... TRITON_CACHE_DIR=... TMPDIR=...
运行上述 ncu 和指定 Python 完成采集，没有修改驱动配置或 sudoers。
其他机器若没有计数器权限，需要由管理员提供访问权限。

上述 ncu-*.csv 和 ncu-summary.json 是早期采样的中间文件，仅保留在本地，
不随文章提交。重新采集 CSV 后，可用
python summarize_ncu.py > ncu-summary.json 生成汇总。脚本只对耗时及 DRAM
字节求和，不将 occupancy、SM throughput 等百分比跨 kernel 求和。
DRAM bytes 是到达显存的事务，不等同于逻辑张量大小；写入可能停留在 L2。

GUI 报告与截图（补充采样）
------------------------
3.3 节的截图及表格使用 nsight-reports/ 中这次补充采集的报告。
本地的 ncu-*.csv 和 ncu-summary.json 来自最初的指标采样，两批数据不要混用。
GPU、驱动、框架版本同上；采集器为 NCU 2025.3.1.0，本机查看器为
macOS arm64 Nsight Compute 2026.3.1.0。

实际采集参数如下，分别将 method 设为 flash64、flash128、naive：

  method=flash128
  ncu --profile-from-start off --target-processes all \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart \
    --section ComputeWorkloadAnalysis \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --import-source yes --export ncu-gui-$method \
    python bench_attention.py --profile $method

本次同样使用普通 SSH 账户已有的 sudo -n 权限访问 performance counter，
通过 env 保留 CUDA_VISIBLE_DEVICES（设备 UUID）、TRITON_CACHE_DIR、TMPDIR，
并指定环境中 Python 的绝对路径。没有更改驱动配置。Triton 报告各执行
24 个 replay pass。--profile 分支只采集 warmup 后一次普通调用，不经过
measure() 中的 CUDA Graph。各报告均采用 NCU 默认 cache/clock control。

原始报告：
  nsight-reports/ncu-gui-flash64.ncu-rep
  nsight-reports/ncu-gui-flash128.ncu-rep
  nsight-reports/ncu-gui-naive.ncu-rep.gz

Naive 原报告约 37 MiB，压缩存放；打开前解压：

  gzip -dk nsight-reports/ncu-gui-naive.ncu-rep.gz
  ncu-ui nsight-reports/ncu-gui-naive.ncu-rep

CSV 可通过下列命令导出，保留原始数值与单位；这些可从报告重新生成的
*-raw.csv 仅保留在本地，不随文章提交：

  ncu --import ncu-gui-flash128.ncu-rep --page raw --csv > ncu-gui-flash128-raw.csv

这些 *-raw.csv 是宽表：首行为指标名，第二行为单位，后续 ID 为数字的行
才是 kernel 数据。Mbyte/Kbyte 为十进制，MiB/KiB 换算用 2**20/2**10。
launch__shared_mem_per_block 包含 driver 部分；flash128 的 kernel 动态
共享内存为 73728 bytes，driver 为 1024 bytes，合计 74752 bytes = 73 KiB。

本机 GUI 实际打开 flash128 报告后，截图到 ../assets/nsight/：
  flash128-throughput.png：Details / Speed Of Light / GPU Throughput Chart
  flash128-memory.png：Memory Chart，Transfer Size，截取 L1/TEX 到 Device Memory
  flash128-occupancy.png：Occupancy / % Occupancy Graphs，截取寄存器曲线

截图均来自实际 GUI，后两张仅裁剪，未改动指标或图表。Memory Chart 中的
12.61 MB 为 dram__bytes_read.sum 的 12609536 bytes（12.025390625 MiB）。
Naive 7 个 kernel 的 DRAM 读取合计 1186480640 bytes（1131.51611328125 MiB）。
GPU Throughput 图中的 Memory 16.94% 不能与 DRAM Throughput 2.73% 混淆。

缓存工作负载
------------
每组独立冷缓存开始；100 条，每条 256 token；不含生成 token。
unique：每条完全不同。
shared192：共同 192 + 每条独有 64。
four_groups：共同 48 + 四组之一的共同 144 + 每条独有 64；四组轮转。
容量为 32768 或 512 个 token slot；按整个叶子淘汰，因此可能超额释放。
token_reuse_rate = reused_tokens / input_tokens。
requests_with_hit / requests 是另一口径，表示至少命中一个 token 的请求比例。

来源
----
https://arxiv.org/abs/1805.02867
https://arxiv.org/abs/2205.14135
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
https://arxiv.org/abs/2309.06180
https://github.com/vllm-project/vllm/tree/v0.11.0/vllm/v1/core
https://docs.vllm.ai/en/v0.11.0/design/paged_attention.html
https://docs.vllm.ai/en/v0.11.0/design/prefix_caching.html
https://arxiv.org/abs/2312.07104
https://github.com/sgl-project/sglang/blob/v0.5.4/python/sglang/srt/mem_cache/radix_cache.py
