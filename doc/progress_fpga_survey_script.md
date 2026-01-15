# FPGA Offloading Survey - Oral Presentation Script

**Duration:** ~8-10 minutes
**Date:** January 14, 2026

---

## Slide 1: Title Slide

*[20 seconds]*

This is my weekly progress report on the FPGA offloading survey for TRACCC.

---

## Slide 2: Completed Work Overview

*[30 seconds]*

Here's an overview of what I've completed this week. I'll go through each item and explain what it means for our FPGA offloading decision.

The main output is a comprehensive survey document - over 3000 lines of analysis in survey-fpga.md.

---

## Slide 3: FP32 vs FP64 Precision Validation

*[1 minute]*

**What I did:**
I built TRACCC with both single and double precision, ran Kalman fitter tests on 10,000 muon tracks, and compared the pull distributions and chi-squared values.

**What it means:**
The result is that FP32 and FP64 produce identical physics results. The pull sigma differs by less than 0.003, which is within statistical noise. Chi-squared per NDF is 1.0042 for both precisions.

The implication is that DSP58 native FP32 is safe for FPGA offloading - we won't lose any physics precision.

---

## Slide 4: NCU Profiling Analysis

*[1 minute]*

**What I did:**
I profiled the propagate-to-next-surface kernel with NVIDIA Nsight Compute, analyzing register pressure, cache hit rates, and warp stall reasons.

**What it means:**
93% of cycles are warp stalls - the kernel is latency-bound, not compute-bound. Register pressure is 96-128 registers per thread, which limits occupancy. L1 cache hit rate is only 46-54%, indicating unpredictable memory access.

The implication is that GPU's SIMT model is fundamentally inefficient for this workload. FPGA's pipelined execution can eliminate these stalls entirely.

---

## Slide 5: Nsys Profiling Analysis

*[1 minute]*

**What I did:**
I ran the full pipeline with Nsight Systems to measure how GPU time is distributed across kernels.

**What it means:**
propagate-to-next-surface takes 63% of GPU time - this is our primary FPGA target. build_tracks takes 14.6% but must stay on GPU due to pointer chasing. The remaining 22% is mostly FPGA-suitable.

In total, about 80% of GPU work could potentially move to FPGA.

---

## Slide 6: FPGA Suitability Assessment

*[1 minute]*

**What I did:**
I analyzed each kernel's computational pattern - MAC chains, memory access patterns, branching behavior - and mapped them to DSP58 capabilities.

**What it means:**
Good for FPGA: RK4 propagation with its MAC chains, B-field polynomial evaluation using Horner's method, matrix-vector operations that map well to systolic arrays, and CCL clustering which is a well-known FPGA pattern.

Keep on GPU: Track deduplication with irregular branching, build_tracks with pointer chasing, and 6x6 matrix inversion with complex control flow.

This gives us a clear partitioning strategy.

---

## Slide 7: Alveo V80 Resource Estimation

*[45 seconds]*

**What I did:**
I estimated DSP58 usage per track pipeline at about 110 DSP58, then calculated how many parallel pipelines we can fit on the V80.

**What it means:**
With 10,848 DSP58 slices on the V80, we can theoretically support about 98 parallel track pipelines. The 32GB HBM is sufficient for geometry and B-field storage.

This is a theoretical estimate - actual numbers require Vitis HLS synthesis to confirm.

---

## Slide 8: PCIe Latency Analysis

*[1 minute]*

**What I did:**
I measured GPU-to-Host transfer latency as a proxy for GPU-to-FPGA communication, testing different data sizes.

Important caveat: this is only a proxy measurement. The actual GPU-to-FPGA path will have different characteristics due to XRT runtime overhead and different PCIe topology. This gives us a rough estimate, but we'll need to validate on real hardware.

**What it means:**
If we transfer only the 24-byte parameter vector per track, overhead is 2.7% - acceptable. If we transfer full track state at 176 bytes, overhead jumps to 13% - concerning. Adding Jacobians pushes it to 18% - prohibitive.

The implication is we have a transfer strategy defined - params-only - but the actual overhead needs validation on real V80 hardware.

---

## Slide 9: Next Step - Validate Per-Step Sync Barrier

*[1.5 minutes]*

This is the critical blocker we need to address next.

The CKF algorithm requires GPU-to-FPGA synchronization at each of 15 surfaces. If this sync overhead exceeds 200 microseconds per step, FPGA offloading is simply not viable.

**What needs to be done:**
1. Instrument the CKF loop to measure actual per-step synchronization time
2. Test with GPU-to-Host sync as a proxy since we don't have FPGA hardware yet
3. Calculate total overhead compared to the 23ms baseline

**Decision criteria:**
- If overhead is less than 5%, we proceed with FPGA development
- If overhead is 5-15%, we need to consider async overlap strategies
- If overhead exceeds 15%, FPGA offloading is not viable for CKF

This is the go/no-go decision point before we invest in any FPGA development.

---

## Slide 10: Summary

*[45 seconds]*

To summarize what's completed:
1. FP32 equals FP64 for physics - DSP58 native FP32 is safe
2. 63% of GPU time is in one kernel with 93% stalls - ideal FPGA candidate
3. 80% of work is FPGA-suitable - clear partitioning strategy
4. V80 can support about 98 parallel pipelines - sufficient resources
5. PCIe overhead is 2.7% for params-only transfer - strategy defined

Next step is to measure the per-step sync barrier overhead. This is the critical go/no-go decision. If it's viable, we'll prototype the RK4 kernel in Vitis HLS.

The full documentation is in doc/survey-fpga.md.

---

## Q&A Notes

**Anticipated Questions:**

1. **Why is sync barrier so critical?**
   - CKF is iterative: propagate -> match -> update -> repeat
   - Each step needs results from previous step
   - Can't pipeline across steps without sync

2. **Can we batch multiple events to hide latency?**
   - Already doing multi-event batching on GPU (+93% throughput)
   - FPGA would need same strategy
   - Doesn't eliminate per-step sync within an event

3. **What if sync overhead is too high?**
   - Could offload only non-CKF kernels (seeding, clustering)
   - Or explore coarser-grained offloading (full propagation batch)
   - Worst case: FPGA not viable, continue GPU-only optimization

---

*End of English script*

---

# 中文口述稿

**時長：** 約 8-10 分鐘
**日期：** 2026 年 1 月 14 日

---

## 投影片 1：封面

*[20 秒]*

老師好，這是我這週的進度報告，主要是關於 TRACCC 做 FPGA offload 的可行性調查。

---

## 投影片 2：完成工作總覽

*[30 秒]*

這邊先列出我完成的工作項目，等一下會一個一個說明我做了什麼、還有這些結果代表什麼意義。

主要產出是一份蠻完整的調查文件，大概三千多行，放在 survey-fpga.md 裡面。

---

## 投影片 3：FP32 與 FP64 精度驗證

*[1 分鐘]*

**我做了什麼：**
我把 TRACCC 分別用 FP32 跟 FP64 編譯，然後跑 Kalman fitter 的測試，總共測了一萬條 muon track，去比較 pull distribution 跟 chi-square。

**這代表什麼：**
結果發現 FP32 跟 FP64 的物理結果基本上一樣。Pull 的 sigma 差不到 0.003，在統計誤差內。Chi-square 兩邊都是 1.0042。

所以結論是 DSP58 原生的 FP32 拿來做 FPGA offload 是沒問題的，不會掉精度。

---

## 投影片 4：NCU Profiling 分析

*[1 分鐘]*

**我做了什麼：**
我用 Nsight Compute 去 profile propagate-to-next-surface 這個 kernel，看它的 register pressure、cache hit rate、還有 warp stall 的狀況。

**這代表什麼：**
結果發現 93% 的 cycle 都卡在 warp stall，所以這個 kernel 是 latency-bound，不是 compute-bound。每個 thread 用了 96 到 128 個 register，occupancy 很低。L1 cache hit rate 只有五成左右，表示 memory access pattern 很亂。

簡單來說，GPU 的 SIMT model 跑這種東西本來就不太適合。但 FPGA 用 pipeline 的方式就可以避掉這些 stall。

---

## 投影片 5：Nsys Profiling 分析

*[1 分鐘]*

**我做了什麼：**
我用 Nsight Systems 跑整個 pipeline，看 GPU time 花在哪些 kernel 上面。

**這代表什麼：**
propagate-to-next-surface 吃掉 63% 的時間，這是我們主要想 offload 的目標。build_tracks 佔 14.6%，但它有 pointer chasing 的問題，不太適合搬到 FPGA。剩下大概 22% 大部分都可以考慮放 FPGA。

整體來看，大概 80% 的 GPU 工作有機會搬到 FPGA 上。

---

## 投影片 6：FPGA 適合度評估

*[1 分鐘]*

**我做了什麼：**
我去分析每個 kernel 的運算特性，像是有沒有 MAC chain、memory access pattern 規不規則、branching 多不多，然後對應到 DSP58 能做什麼。

**這代表什麼：**
適合放 FPGA 的像是：RK4 propagation 因為就是一直做乘加、B-field 的多項式用 Horner's method 很適合、matrix-vector 可以用 systolic array、CCL clustering 本來就是 FPGA 很成熟的東西。

要留在 GPU 的像是：track deduplication 因為 branching 很亂、build_tracks 有 pointer chasing、還有 6x6 matrix inversion 控制流程太複雜。

所以我們有一個蠻清楚的切分策略。

---

## 投影片 7：Alveo V80 資源估算

*[45 秒]*

**我做了什麼：**
我估了一下每條 track 的 pipeline 大概要 110 個 DSP58，然後算 V80 可以塞幾條。

**這代表什麼：**
V80 有一萬多個 DSP58，理論上可以跑大概 98 條平行的 track pipeline。HBM 有 32GB，放 geometry 跟 B-field 絕對夠。

不過這是紙上估算啦，實際要等 Vitis HLS 合成才知道。

---

## 投影片 8：PCIe 延遲分析

*[1 分鐘]*

**我做了什麼：**
我量了 GPU 到 Host 的傳輸延遲，拿這個來估 GPU 到 FPGA 的通訊成本，測了不同的資料大小。

不過要先說，這只是一個 proxy，實際 GPU 到 FPGA 會因為 XRT 的 overhead 跟 PCIe topology 不一樣而有差異。這只能給一個大概，真正的數字要在硬體上驗證。

**這代表什麼：**
如果只傳 24 bytes 的 parameter，overhead 大概 2.7%，OK。如果傳整個 track state 176 bytes，就跳到 13%，有點多。再加 Jacobian 的話到 18%，太高了。

結論是傳輸策略確定了，就是只傳 parameter，但實際 overhead 還是要在 V80 上驗證。

---

## 投影片 9：下一步：驗證每步同步屏障

*[1.5 分鐘]*

接下來要處理一個比較關鍵的問題。

CKF 演算法每過一個偵測面就要做一次 GPU 跟 FPGA 的同步，總共 15 個面。如果每步的 sync overhead 超過 200 微秒，那 FPGA offload 基本上就不可行了。

前面的 survey 其實是為了這個測試做準備，要先知道傳多少資料、有幾個 sync point、FP32 可不可以用，才能設計這個測試。

**要做的事：**
1. 在 CKF loop 裡面加 timing，量每一步的 sync 時間
2. 先用 GPU 到 Host 的 sync 來測，因為 FPGA 還在裝
3. 算總 overhead，跟 23ms 的 baseline 比

**決策標準：**
- 小於 5%：繼續做 FPGA
- 5% 到 15%：要想辦法用 async 或 overlap
- 超過 15%：FPGA offload 對 CKF 來說就不太行了

這是我們決定要不要繼續投入 FPGA 開發的關鍵點。

---

## 投影片 10：總結

*[45 秒]*

總結一下：
1. FP32 跟 FP64 物理結果一樣，DSP58 的 FP32 可以用
2. 63% 的 GPU time 集中在一個 kernel，而且 93% 都在 stall，很適合 FPGA
3. 80% 的工作適合 FPGA，切分策略很清楚
4. V80 理論上可以跑 98 條平行 pipeline，資源夠
5. 只傳 parameter 的話 PCIe overhead 2.7%，策略確定了

下一步就是量 sync barrier 的 overhead，這是關鍵決策點。如果可行的話，就開始用 Vitis HLS 做 RK4 kernel 的 prototype。

完整文件在 survey-fpga.md。

---

## 預期問答

**可能會被問的：**

1. **為什麼 sync barrier 這麼重要？**
   - CKF 是迭代的：propagate → match → update → 重複
   - 每一步都要等前一步的結果
   - 沒辦法跨 step 做 pipeline

2. **可以用 multi-event batching 來藏延遲嗎？**
   - GPU 上已經在做了，throughput 提升 93%
   - FPGA 也要用一樣的策略
   - 但這不能消除單一 event 裡面的 per-step sync

3. **如果 sync overhead 太高怎麼辦？**
   - 可以只 offload 非 CKF 的 kernel，像 seeding、clustering
   - 或是用更粗粒度的 offload
   - 最壞就是 FPGA 不可行，繼續做 GPU 優化

---

*口述稿結束*
