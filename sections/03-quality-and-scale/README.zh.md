<!--
LOCALIZED-SOURCE
id: 03-quality-and-scale
language: zh-CN
-->

# 面向生成质量的 FP8

Diffusion 的每一步输出都会成为下一步输入，局部 FP8 误差可能沿去噪过程累积。因此，TeleFuser 将 attention smoothing 直接纳入 FP8 执行路径，而不是作为独立的后处理步骤。

MiniMax-H3 的层级 profile 显示，部分 K/V tensor 的均值明显偏离零点。TeleFuser 在 FP8 attention 前中心化 K/V，并在输出中恢复等价偏移。这种针对 attention 分布设计的非对称量化处理，可以更充分地利用 FP8 动态范围，同时保持模型接口不变。

中心化与输出修正已经融合进 FP8 Sol-Attn，并在优化配置中默认开启。相关端到端质量与性能数据放在主体性能评测之后。
