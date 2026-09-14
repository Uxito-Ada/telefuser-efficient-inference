<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# 性能与生成效果 {#results}

## 四卡 Base H3

主测试在四张 GPU 上运行 MiniMax-H3 Base。LightX2V 和 TeleFuser 均启用 `TP2 x Ulysses SP2`，并采用相同的 prompt、seed、分辨率、帧数、帧率和 50-point schedule，同时关闭 feature cache。两套框架均采用各自经过验证的优化配置。

![四 GPU MiniMax-H3 Base 性能](assets/lightx2v-base-h3.svg)

<div class="result-summary">
  <div><strong>快 2.64 倍</strong><span>完整生成</span></div>
  <div><strong>快 2.62 倍</strong><span>去噪，step 吞吐提升 162.2%</span></div>
  <div><strong>降低 40.3%</strong><span>代表性单卡峰值显存</span></div>
</div>

完整生成时间从 137.76 秒降至 52.27 秒，去噪时间从 129.22 秒降至 49.28 秒。两套框架使用相同的四卡并行配置，因此可以直接比较加速比。

<div class="video-pair" data-sync-group="base-h3">
  <figure>
    <figcaption>LightX2V</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video>
  </figure>
</div>

两段输出都是 124 帧拉面场景，并带有同步双声道音频。每个框架先 warm-up 一次，再记录一次正式请求。GPU 0 当时有一份固定的无关显存占用，因此图中的显存取其余三张空闲卡的峰值中位数。

## Adapter 工作负载

Adapter 评测覆盖 MiniMax-H3 Turbo LoRA 和 FastH3 dense hybrid adapter。每组测试均选择支持对应模型的外部框架作为对照，并对齐输入、输出规格和 DiT 调用次数。

### MiniMax-H3 Turbo LoRA

Turbo 测试使用单张 GPU 和 8-step v1.0 768p Adapter，共执行八次 DiT。两套框架均将 DiT 常驻 GPU；TeleFuser 在低精度执行前合并 LoRA。图中不包含 CPU block offload 结果。

![MiniMax-H3 Turbo Adapter 性能](assets/turbo-performance.svg)

<div class="result-summary">
  <div><strong>降低 26.7%</strong><span>相比 LightX2V 的去噪时间</span></div>
  <div><strong>提升 36.5%</strong><span>8-step 去噪吞吐</span></div>
  <div><strong>降低 12.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

<div class="video-pair" data-sync-group="turbo">
  <figure>
    <figcaption>LightX2V：MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video>
  </figure>
</div>

性能测试仍使用两端一致的 prompt。页面中的 TeleFuser Turbo 视频采用稳定镜头展示 prompt：保持水平三脚架机位，只做轻微推近，并明确抑制环绕、滚转和旋转，使画面运动集中在人物动作上。

### FastH3 dense adapter

FastH3 测试同样使用单张 GPU。两套框架采用相同的 dense adapter、prompt、seed、1344 x 768 输出和 124 帧配置，实际执行四次 DiT，并采用各自经过验证的优化配置；去噪期间均不进行 DiT CPU offload。每套框架先完成一次 warm-up，正式结果取三次生成的中位数。

![FastH3 Adapter 对齐性能](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>降低 25.7%</strong><span>相比 FastVideo 的去噪时间</span></div>
  <div><strong>提升 34.6%</strong><span>实际 DiT forward 吞吐</span></div>
  <div><strong>降低 14.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

<div class="video-pair" data-sync-group="fasth3">
  <figure>
    <figcaption>FastVideo：FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

FastH3 只比较去噪阶段，因为两套框架对 prompt-conditioning cache 的处理不同，完整请求时间不能直接相除。上面的四段 Adapter 输出均为 1344 x 768、124 帧的 H.264 视频，并带有 32kHz 双声道 AAC 音频。

## 通信计算重叠

四卡执行支持 Ulysses 通信与 attention 计算重叠。在另一组包含五个 case 的回归测试中，平均请求时间从 78.546 秒降至 75.766 秒，改善 **3.539%**；五个 MP4 均保持 byte-identical。该优化用于四卡 Base H3 的分布式执行。
