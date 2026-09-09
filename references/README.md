# Related Work and Writing References

The article follows the causal, evidence-first structure of strong systems
papers and technical releases. It borrows their method of argument, never their
wording or unmatched benchmark values.

## Model and competing systems

| Work | Role in this article |
|---|---|
| [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | The only model studied; the official Diffusers route is a quality reference. |
| [FastH3](https://haoailab.com/blogs/fasth3-preview/) | Current H3 acceleration work and presentation reference. |
| [FastVideo MiniMax-H3 recipes](https://haoailab.com/FastVideo/cookbook/minimax-h3/) | Maintained external CUDA baseline and FastH3 adapter contract. |
| [LightX2V MiniMax-H3](https://github.com/ModelTC/LightX2V) | Candidate external Sol-Attn comparison, admitted only after valid matched output. |

## Algorithms and kernels

| Work | Connection |
|---|---|
| [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) | Training-free on-the-fly sparse routing with summary correction. |
| [SageAttention2](https://arxiv.org/abs/2411.10958) | Low-precision attention and smoothing as part of numerical kernel design. |
| [xDiT](https://arxiv.org/abs/2411.01738) | Sequence-parallel layouts for diffusion transformers. |
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) | Hardware-aware algorithm/kernel co-design and the external dense baseline. |
| [VSA](https://arxiv.org/abs/2505.13389) | Learned video sparse attention; related but not a matched adapter-free comparison. |
| [TorchAO](https://pytorch.org/ao/stable/) | General quantization infrastructure; coverage is not a custom FP8 Sol contract. |

The implementation discussion also cites
[TeleFuser PR #37](https://github.com/Tele-AI/TeleFuser/pull/37), which follows
the original Ulysses integration with communication/attention overlap and
lossless data-movement fusion.

## Presentation references

- [NVIDIA TensorRT video-diffusion optimization](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/):
  profile the full pipeline, then connect kernel work to end-to-end impact.
- [FastH3 release](https://haoailab.com/blogs/fasth3-preview/): keep model
  quality, latency protocol, generated media, and deployment recipe adjacent.
- [NVIDIA multi-device inference](https://developer.nvidia.com/blog/scaling-ai-inference-across-multiple-gpus-using-nvidia-tensorrt-with-multi-device-inference-support/):
  distinguish local kernel throughput from distributed system throughput.

## Writing rules derived from the study

1. Start from a user-visible model requirement and its computational cost.
2. Introduce a method only after the previous design leaves a measured or
   formally explained gap.
3. Use one running workload to make abstract tensor and lifecycle problems
   concrete.
4. Separate hardware availability, algorithmic approximation, kernel
   implementation, and framework lifecycle.
5. End with one new matched experiment, not a collage of historical wins.
