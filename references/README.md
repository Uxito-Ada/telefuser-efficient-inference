# Reference Work and Writing Pattern

The article follows the evidence-first structure used by recent systems and
efficient-AI releases. It borrows the presentation pattern, not their wording.

## Primary technical references

| Work | Relevance |
|---|---|
| [MiniMax H3](https://www.minimax.io/blog/minimax-h3) | Model context: unified video and stereo-audio generation |
| [FastH3 Preview v1](https://haoailab.com/blogs/fasth3-preview/) | Strong release pattern: TL;DR, quality samples, matched performance protocol, ablations, and recipes |
| [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) | Training-free on-the-fly block routing and approximate correction |
| [SageAttention2](https://arxiv.org/abs/2411.10958) | Quantized attention and outlier smoothing |
| [xDiT](https://arxiv.org/abs/2411.01738) | Diffusion inference parallelism and Ulysses/ring composition |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) | Algorithm-kernel co-design under asymmetric hardware scaling |
| [NVIDIA: TensorRT video diffusion](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/) | Profile-first FP8 deployment and end-to-end accounting |
| [NVIDIA: NVFP4 FLUX.2](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/) | Composing quantization, caching, compilation, and sequence parallelism |
| [VSA](https://arxiv.org/abs/2505.13389) | Trainable block-sparse video attention and hardware-aware sparsity |

## Presentation decisions derived from the study

- Start with a measurable deployment problem, not an API changelog.
- Present a system diagram before individual kernels.
- Separate algorithm, numerical format, kernel, and framework lifecycle.
- Keep a matched workload adjacent to every chart.
- Show generated media next to quantitative evidence.
- Include ablations and failed paths; optimized systems are defined by their
  rejected alternatives as much as by their final configuration.
- End with reproduction boundaries and architecture support.

## Scope boundaries

TeleFuser does not claim to invent FP8, NF4, Sol-Attn, Ulysses, LoRA, or
attention smoothing. The contribution studied here is their integration into a
coherent video-DiT inference stack, together with SM90 kernels, lifecycle
management, distributed placement, quality controls, and measured deployment
trade-offs.
