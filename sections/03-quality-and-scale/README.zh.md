<!--
LOCALIZED-SOURCE
id: 03-quality-and-scale
language: zh-CN
-->

# 控制误差，并扩展到多卡

Diffusion 的每一步输出都会成为下一步输入。FP8 或稀疏 attention 引入的局部误差可能沿去噪过程累积，因此评测同时覆盖 kernel 吞吐、tensor 误差、最终视频和音频。

## FP8 attention 的误差控制

MiniMax-H3 实际层的 profile 显示，部分 K/V 的均值明显偏离零点，直接使用对称 FP8 量化会损失有效动态范围。TeleFuser 在计算前对 K/V 进行中心化，并在 attention 输出中恢复等价偏移。这是一种针对 attention 数据分布设计的非对称量化处理。

中心化和输出修正已经融合进 FP8 Sol-Attn。最早的非融合版本让去噪时间增加 11.7%，融合后开销降到 2.2%。在捕获的真实 H3 层上，K 的量化 MSE 降低 21.65%，attention 输出 MSE 降低 8.18%；KV smoothing 和 V correction 在优化配置中默认开启。

| 模型 | 分辨率与帧数 | 采样 | GPU | 场景 / seed |
|---|---|---|---:|---|
| MiniMax-H3 Base，T2VA | 1344 × 768，107 帧，4 秒，24 fps | 50 points / 49 DiT updates | 1 × H100 | 雪地电车 / 17 |

![MiniMax-H3 FP8 smoothing 单卡性能](assets/smoothing-performance.svg)

平滑 FP8 的去噪吞吐比 BF16 Linear + FlashAttention 4 提高 37.2%，峰值分配显存降低 42.6%；相对未平滑 FP8，融合修正增加 2.1% 去噪时间。

| 配置（BF16 为 reference） | 视频 PSNR ↑ | 视频 SSIM ↑ | 音频 cosine ↑ | 频谱收敛误差 ↓ |
|---|---:|---:|---:|---:|
| FP8，不使用 smoothing | **19.63 dB** | **0.6712** | 0.8504 | 0.5055 |
| FP8，开启 smoothing | 19.27 dB | 0.6700 | **0.8891** | **0.4537** |

**Prompt:** `Locked-off cinematic wide shot of a red vintage tram gliding slowly through a snowy alpine village at sunrise. The tram remains rigid and geometrically consistent, its windows and wheels stay aligned. Light snow falls; soft rail sounds and distant church bells are synchronized with the scene. No people, no cuts, no camera movement.`

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 参考</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8，不使用 smoothing</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8，开启 smoothing</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

在视频后半段，未平滑输出的车顶标识、受电弓连线和窗框对齐相较平滑输出更不稳定。

## 开箱即用、可继续调优的稀疏策略

不同模型和生成任务对稀疏 attention 的敏感位置并不相同。TeleFuser 提供 dense window、dense layer、阈值模式和稀疏强度等配置接口，便于针对新的模型或质量目标继续调优；MiniMax-H3 的默认配置已经过性能与生成质量验证，无需用户手动选择稀疏参数。

## 多卡执行

TeleFuser 对 SP kernel 做了面向 FP8、Sol-Attn 和 world-model 视频生成的专门优化：Ulysses All-to-All 完成后，各 GPU 再按本地 attention 布局执行 FP8 quantization 和 Sol-Attn；3D 视频 token 在序列切分前完成 reorder；scalar timestep 保持完整，per-token timestep 则随视频 token 一起切分。TeleFuser 还将 Ulysses 通信与 attention 计算重叠，以降低多卡通信开销。
