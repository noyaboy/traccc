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

**What it means:**
If we transfer only the 24-byte parameter vector per track, overhead is 2.7% - acceptable. If we transfer full track state at 176 bytes, overhead jumps to 13% - concerning. Adding Jacobians pushes it to 18% - prohibitive.

The implication is we must minimize data transfer. We should cache covariance matrices on FPGA HBM and only transfer the small parameter vectors.

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

各位好，這是我本週關於 TRACCC FPGA 卸載可行性調查的進度報告。

---

## 投影片 2：完成工作總覽

*[30 秒]*

這張投影片列出了我這週完成的所有工作項目。接下來我會逐一說明每項工作的內容，以及它對我們 FPGA 卸載決策的意義。

主要的產出是一份完整的調查文件，總共超過三千行的分析內容，放在 survey-fpga.md 裡面。

---

## 投影片 3：FP32 與 FP64 精度驗證

*[1 分鐘]*

**我做了什麼：**
我分別用單精度和雙精度編譯了 TRACCC，然後對一萬條 muon 軌跡執行 Kalman fitter 測試，比較兩者的 pull 分布和卡方值。

**這代表什麼意義：**
結果顯示 FP32 和 FP64 產生的物理結果完全相同。Pull 的標準差差異小於 0.003，這在統計誤差範圍內。兩種精度的卡方值都是 1.0042。

這個結果的意義是：DSP58 原生的 FP32 運算對 FPGA 卸載來說是安全的，不會損失任何物理精度。

---

## 投影片 4：NCU Profiling 分析

*[1 分鐘]*

**我做了什麼：**
我用 NVIDIA Nsight Compute 對 propagate-to-next-surface 這個 kernel 做了詳細的效能分析，檢查暫存器壓力、快取命中率，還有 warp stall 的原因。

**這代表什麼意義：**
93% 的 cycle 都在 warp stall，這表示這個 kernel 是延遲受限的，不是運算受限的。每個 thread 用了 96 到 128 個暫存器，這限制了 occupancy。L1 快取命中率只有 46% 到 54%，顯示記憶體存取模式很不規則。

這代表 GPU 的 SIMT 執行模型對這種工作負載根本上就是沒效率的。FPGA 的 pipeline 執行方式可以完全消除這些 stall。

---

## 投影片 5：Nsys Profiling 分析

*[1 分鐘]*

**我做了什麼：**
我用 Nsight Systems 跑了完整的 pipeline，量測 GPU 時間在各個 kernel 之間的分布。

**這代表什麼意義：**
propagate-to-next-surface 佔了 63% 的 GPU 時間，這是我們主要的 FPGA 卸載目標。build_tracks 佔 14.6%，但因為有 pointer chasing 的問題，必須留在 GPU 上。剩下的 22% 大部分都適合放到 FPGA。

總結來說，大約 80% 的 GPU 工作有機會移到 FPGA 上執行。

---

## 投影片 6：FPGA 適合度評估

*[1 分鐘]*

**我做了什麼：**
我分析了每個 kernel 的運算特性，包括乘加鏈、記憶體存取模式、分支行為，然後對應到 DSP58 的能力。

**這代表什麼意義：**
適合 FPGA 的有：RK4 傳播運算因為是乘加鏈、磁場多項式用 Horner's method 很適合、矩陣向量運算可以用 systolic array、還有 CCL clustering 本來就是 FPGA 上很成熟的設計。

需要留在 GPU 的有：軌跡去重因為分支很不規則、build_tracks 有 pointer chasing、還有 6x6 矩陣反運算控制流程太複雜。

這給了我們一個清楚的切分策略。

---

## 投影片 7：Alveo V80 資源估算

*[45 秒]*

**我做了什麼：**
我估算了每條軌跡 pipeline 大約需要 110 個 DSP58，然後計算 V80 上可以放多少條平行的 pipeline。

**這代表什麼意義：**
V80 有 10,848 個 DSP58，理論上可以支援大約 98 條平行的軌跡 pipeline。32GB 的 HBM 足夠存放幾何資料和磁場。

不過這是理論估算，實際數字要等 Vitis HLS 合成之後才能確認。

---

## 投影片 8：PCIe 延遲分析

*[1 分鐘]*

**我做了什麼：**
我量測了 GPU 到 Host 的傳輸延遲，用這個來估算未來 GPU 到 FPGA 的通訊成本，測試了不同的資料大小。

**這代表什麼意義：**
如果只傳 24 bytes 的參數向量，overhead 是 2.7%，這是可以接受的。如果傳完整的軌跡狀態 176 bytes，overhead 就跳到 13%，這就有點令人擔心。如果再加上 Jacobian，就到 18%，這太高了。

結論是我們必須最小化資料傳輸量。應該把共變異數矩陣快取在 FPGA 的 HBM 上，只傳小的參數向量。

---

## 投影片 9：下一步：驗證每步同步屏障

*[1.5 分鐘]*

這是我們接下來要處理的關鍵阻礙。

CKF 演算法在每一個偵測面都需要 GPU 和 FPGA 之間做同步，總共有 15 個面。如果每步的同步 overhead 超過 200 微秒，FPGA 卸載就根本不可行。

**需要做的事：**
1. 在 CKF 迴圈中加入計時，量測實際的每步同步時間
2. 先用 GPU 到 Host 的同步來測試，因為我們還沒有 FPGA 硬體
3. 計算總 overhead，跟 23 毫秒的 baseline 比較

**決策標準：**
- 如果 overhead 小於 5%，我們就繼續進行 FPGA 開發
- 如果在 5% 到 15% 之間，要考慮非同步重疊的策略
- 如果超過 15%，FPGA 卸載對 CKF 來說就不可行

這是我們投入任何 FPGA 開發之前的關鍵決策點。

---

## 投影片 10：總結

*[45 秒]*

總結一下已完成的工作：
1. FP32 等於 FP64 的物理結果，DSP58 原生 FP32 是安全的
2. 63% 的 GPU 時間集中在一個 kernel，而且有 93% 的 stall，非常適合 FPGA
3. 80% 的工作適合 FPGA，切分策略很清楚
4. V80 可以支援大約 98 條平行 pipeline，資源足夠
5. 只傳參數的話 PCIe overhead 是 2.7%，傳輸策略已經確定

下一步是量測每步同步屏障的 overhead，這是關鍵的決策點。如果可行的話，我們就開始用 Vitis HLS 做 RK4 kernel 的原型。

完整的文件在 doc/survey-fpga.md。

---

## 預期問答

**可能被問的問題：**

1. **為什麼同步屏障這麼關鍵？**
   - CKF 是迭代式的：傳播 → 匹配 → 更新 → 重複
   - 每一步都需要前一步的結果
   - 沒辦法跨步驟做 pipeline

2. **可以用多事件批次來隱藏延遲嗎？**
   - GPU 上已經在做多事件批次了，throughput 提升了 93%
   - FPGA 也需要同樣的策略
   - 但這不能消除單一事件內的每步同步

3. **如果同步 overhead 太高怎麼辦？**
   - 可以只卸載非 CKF 的 kernel，像是 seeding 和 clustering
   - 或者探索更粗粒度的卸載方式
   - 最壞的情況就是 FPGA 不可行，繼續做 GPU 優化

---

*口述稿結束*
