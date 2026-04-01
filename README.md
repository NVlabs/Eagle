<div align="center">

#  🦅  Eagle: Frontier Vision-Language Models with Data-Centric Strategies

<p>
    <img src="Eagle/assets/Eagle.png" alt="Eagle" width="500" height="auto">
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-NVIDIA%20License-red.svg)](LICENSE_MODEL)

[[📘Eagle 2.5 Report](Eagle2_5/Eagle2.5.pdf)] [[📘Eagle 2 Report](Eagle2_5/Eagle2.pdf)] [[📘Eagle Report](Eagle/Eagle.pdf)] [[🤗HF Models](https://huggingface.co/collections/nvidia/eagle)] [[🤗HF Demo](https://huggingface.co/spaces/nvidia/Eagle-2.5-8B-demo)] [[🌐Project Page](https://nvlabs.github.io/Eagle/)]

</div>


## Updates
- [2025/10] 🔥 Release Eagle 2.5 [source code](https://github.com/NVlabs/EAGLE/tree/main/Eagle2_5).
- [2025/09] 🔥 Eagle 2.5 is accepted to [NeurIPS 2025](https://neurips.cc/Conferences/2025).
- [2025/09] 🎉 Eagle 2 is supported in [Torch-TRT](https://github.com/pytorch/TensorRT/tree/main/tools/llm).
- [2025/07] 🎉 Release Eagle 2.5 [model](https://huggingface.co/nvidia/Eagle2.5-8B).
- [2025/06] 🔥 Eagle 2.5 is adopted as the VLM backbone of [GR00T-N1.5](https://huggingface.co/nvidia/GR00T-N1.5-3B). Check out the [tech blog](https://research.nvidia.com/labs/gear/gr00t-n1_5/) for more details.
- [2025/04] 🎉 Release Eagle 2.5 [tech report](https://arxiv.org/abs/2504.15271).
- [2025/03] 🔥 Eagle 2 is adopted as the VLM backbone (System-2) of [GR00T-N1](https://youtu.be/m1CH-mgpdYg?si=WSDroL32HHE9xYSN). Check out the [GTC launch](https://www.youtube.com/watch?v=_waPvOwL9Z8&t=6935s) and [white paper](https://arxiv.org/abs/2503.14734) for more details.
- [2025/01] 🎉 Release Eagle 2 [tech report](http://arxiv.org/abs/2501.14818) and [models](https://huggingface.co/collections/nvidia/eagle).
- [2025/01] 🎉 [Eagle](./Eagle/README.md) is accepted as [ICLR 2025](https://iclr.cc) Spotlight.
- [2024/08] 🎉 Release [Eagle](./Eagle/README.md).

## Resources

### 🌟 Start Here: Set Up Environment, Train the Model, and Run Evaluations
- 📚 [Getting Started with Eagle-2.5](./Eagle2_5/document/0.onboarding.md)

### 🌐 Playground
- 🤗 Demo on Huggingface Space
  - https://huggingface.co/spaces/nvidia/Eagle-2.5-8B-demo

## Introduction

Eagle 2.5 is a family of frontier vision-language models (VLMs) designed for long-context multimodal learning. While most existing VLMs focus on short-context tasks, Eagle 2.5 addresses the challenges of long video comprehension and high-resolution image understanding, providing a generalist framework for both. Eagle 2.5 supports up to 512 video frames and is trained jointly on image + video data.

We also introduce Eagle-Video-110K, a novel dataset with both story-level and clip-level annotations, specifically curated for long video understanding. The dataset contains over 110K annotated samples, including QA, localization, and summarization. The videos range from a few minutes to 3 hours - pushing the limits of long-form visual reasoning.

### 🚀 Strong Results Across The Board:

- SOTA on 6 out of 10 long video benchmarks
- Outperforms GPT-4o (0806) on 3/5 video tasks
- Outperforms Gemini 1.5 Pro on 4/6 video tasks
- Matches or outperforms Qwen2.5-VL-72B on multiple key datasets
- 72.4% on Video-MME with 512 input frames
- Strong image understanding with consistent improvement over Eagle 2, matching Qwen2.5-VL.

### 🎯 Key Innovations

- **Information-First Sampling**:
  - *Image Area Preservation (IAP)*: Optimizes image tiling to retain most of the original image area and aspect ratio, preserving fine-grained details.
  - *Automatic Degrade Sampling (ADS)*: Dynamically balances visual and textual input, ensuring complete text retention while maximizing visual content within context length constraints.
- **Progressive Mixed Post-Training**:
  - Gradually increases context length during training, enhancing the model's ability to process varying input sizes and improving information density over static sampling.
- **Diversity-Driven Data Recipe**:
  - Combines open-source data (human-annotated and synthetic) with the self-curated Eagle-Video-110K dataset, collected via a diversity-driven strategy and annotated with both story-level and clip-level QA pairs.

### ⚡ Efficiency & Framework Optimization

- **GPU Memory Optimization**:
  - Integrate Triton-based fused operators replacing PyTorch’s MLP, RMSNorm, and RoPE implementations.
  - Reduced GPU memory with fused linear layers + cross-entropy loss (removes intermediate logit storage) and CPU-offloading of hidden states.
  - Sufficient to fit up to 32K context length with an 8B model on a single GPU. 
- **Distributed Context Parallelism**:
  - Adopts a two-layer communication group based on Ulysses and Ring/Context Parallelism building on USP.
  - Implements ZigZag Llama3-style Context Parallelism with all-gather KV to reduce communication latency.
- **Video Decoding Acceleration**:
  - Optimized sparse video frame sampling with rapid video metadata parsing, improved long video decoding and reduced memory consumption.
- **Inference Acceleration**:
  - Supports vLLM deployment with reduced memory and accelerated inference.
  

## Model Details

- **Model Type**: Long-context vision-language model
- **Architecture**: 
  - Vision encoder: Siglip2-So400m-Patch16-512
  - Language model: Qwen2.5-7B-Instruct
  - Multimodal base architecture: LLaVA with tiling-based vision input
- **Supported Inputs**: 
  - Long video sequences (up to 512 frames)
  - High-resolution images (up to 4K HD input size)
  - Multi-page documents
  - Long text
- **Training Strategy**: 
  - Progressive mixed post-training, expanding from 32K to 128K context length
  - Information-first sampling for optimal visual and textual information retention
- **Training Data**: 
  - Open-source video and document datasets
  - Eagle-Video-110K (110K long videos with dual-level annotation)


## Model Zoo

### 📦 Eagle 2.5 Models
| Model Name  | Date       |   LLM Backbone   |  Vision Encoder  | Max Length | Download |
| ----------- |------------| ---------------- | ---------------- | ---------- | ------- |
| Eagle2.5-8B | 2025.04.16 | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-512) | 128K | 🤗 [HF Link](https://huggingface.co/nvidia/Eagle2-1B) |

### 📦 Eagle 2 Models
| Model Name  | Date       |   LLM Backbone   |  Vision Encoder  | Max Length | Download |
| ----------- |------------| ---------------- | ---------------- | ---------- | ------- |
| Eagle2-1B | 2025.01.11 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448)  | 16K | 🤗 [HF Link](https://huggingface.co/nvidia/Eagle2-1B) |
| Eagle2-2B | 2025.01.11 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448)  | 16K | 🤗 [HF Link](https://huggingface.co/nvidia/Eagle2-2B) |
| Eagle2-9B | 2025.01.11 | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)     |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448) + [ConvNext](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)  | 16K | 🤗 [HF Link](https://huggingface.co/nvidia/Eagle2-9B) |
| Eagle2-34B | 2025.01.11 | [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)  |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448) + [ConvNext](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)  | 16K | 🤗 [HF Link](https://huggingface.co/nvidia/Eagle2-34B) |

## Benchmarks Results
### 🎥 Video Benchmarks

| Benchmark                                  | GPT-4o             | Gemini-1.5 Pro    | InternVL2.5-8B      | Qwen2.5-VL-8B       | **Eagle2.5-8B**     |
|--------------------------------------------|--------------------|-------------------|---------------------|---------------------|---------------------|
| MVBench<sub>test</sub>                     | -                  | -                 | 72.0                | 69.6                | 74.8                |
| Perception_test<sub>val</sub>              | -                  | -                 | -                   | 70.5                | 82.0                |
| EgoSchema<sub>fullset</sub>                | -                  | 72.2              | -                   | 65.0                | 72.2                |
| MMB-Video                                  | 1.63               | 1.30              | 1.68                | 1.79                | 1.94                |
| MLVU<sub>val</sub>                         | -                  | -                 | 68.9                | 70.2                | 77.6                |
| LVBench<sub>val</sub>                      | 66.7               | 64.0              | 60.0                | 56.0                | 66.4                |
| Video-MME<sub>w/o subtitle</sub>           | 71.9               | 75.0              | 64.2                | 65.1                | 72.4                |
| Video-MME<sub>w subtitle</sub>             | 77.2               | 81.3              | 66.9                | 71.6                | 75.7                |
| CG-Bench<sub>Clue</sub>                    | 58.6               | 50.9              | -                   | 44.5                | 55.8                |
| CG-Bench<sub>Long</sub>                    | 44.9               | 37.8              | -                   | 35.5                | 46.6                |
| CG-Bench<sub>mIoU</sub>                    | 5.73               | 3.85              | -                   | 2.48                | 13.4                |
| HourVideo<sub>Dev</sub>                    | -                  | 37.2              | -                   | -                   | 44.5                |
| HourVideo<sub>Test</sub>                   | -                  | 37.4              | -                   | -                   | 41.8                |
| Charade-STA<sub>mIoU</sub>                 | 35.7               | -                 | -                   | 43.6                | 65.9                |
| HD-EPIC                                    | -                  | 37.6              | -                   | -                   | 42.9                |
| HRVideoBench                               | -                  | -                 | -                   | -                   | 68.5                |
| EgoPlan<sub>val</sub>                      | -                  | -                 | -                   | -                   | 45.3                |

### 🦾 Embodied Benchmarks
| Benchmark                                  | GPT-4o             | Gemini-1.5 Pro    | InternVL2.5-8B      | Qwen2.5-VL-8B       | **Eagle2.5-8B**     |
|--------------------------------------------|--------------------|-------------------|---------------------|---------------------|---------------------|
| OpenEQA                                    | -                  | -                 | -                   | -                   | 63.5                |
| ERQA                                       | 47.0               | 41.8              | -                   | -                   | 38.3                |
| EgoPlan<sub>val</sub>                      | -                  | -                 | -                   | -                   | 45.3                |

### 🖼️ Image Benchmarks

| Benchmark                                  | GPT-4o             | Gemini-1.5 Pro    | InternVL2.5-8B      | Qwen2.5-VL-8B       | **Eagle2.5-8B**     |
|--------------------------------------------|--------------------|-------------------|---------------------|---------------------|---------------------|
| DocVQA<sub>test</sub>                      | 92.8               | 93.1              | 93.0                | 95.7                | 94.1                |
| ChartQA<sub>test</sub>                     | 85.7               | 87.2              | 84.8                | 87.3                | 87.5                |
| InfoVQA<sub>test</sub>                     | 79.2               | 81.0              | 77.6                | 82.6                | 80.4                |
| TextVQA<sub>val</sub>                      | 77.4               | 78.8              | 79.1                | 84.9                | 83.7                |
| OCRBench<sub>test</sub>                    | 736                | 754               | 822                 | 864                 | 869                 |
| MMstar<sub>test</sub>                      | 64.7               | 59.1              | 62.8                | 63.9                | 66.2                |
| RWQA<sub>test</sub>                        | 75.4               | 67.5              | 70.1                | 68.5                | 76.7                |
| AI2D<sub>test</sub>                        | 84.6               | 79.1              | 84.5                | 83.9                | 84.5                |
| MMMU<sub>val</sub>                         | 69.1               | 62.2              | 56.0                | 58.6                | 55.8                |
| MMBench_V11<sub>test</sub>                 | 83.1               | 74.6              | 83.2                | 82.6                | 81.7                |
| MMVet<sub>GPT-4-Turbo</sub>                | 69.1               | 64.0              | 62.8                | 67.1                | 62.9                |
| HallBench<sub>avg</sub>                    | 55.0               | 45.6              | 50.1                | 52.9                | 54.7                |
| MathVista<sub>testmini</sub>               | 63.8               | 63.9              | 64.4                | 68.2                | 67.8                |
| Avg Score                                  | 74.9               | 71.7              | 73.1                | 75.6                | 75.6                |

*All numbers are directly extracted from Table 2 and Table 3 of the Eagle 2.5 Tech Report.*


## Citation
If you find this project useful, please cite our work:
```latex
@booktitle{chen2025eagle2.5,
    title={Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models},
    author={Chen, Guo and Li, Zhiqi and Wang, Shihao and Jiang, Jindong and Liu, Yicheng and Lu, Lidong and Huang, De-An and Byeon, Wonmin and Le, Matthieu and Ehrlich, Max and Lu, Tong and Wang, Limin and Catanzaro, Bryan and Kautz, Jan and Tao, Andrew and Yu, Zhiding and Liu, Guilin},
    booktitle={NeurIPS},
    year={2025}
}
```

```latex
@article{li2025eagle2,
    title={Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models}, 
    author={Zhiqi Li and Guo Chen and Shilong Liu and Shihao Wang and Vibashan VS and Yishen Ji and Shiyi Lan and Hao Zhang and Yilin Zhao and Subhashree Radhakrishnan and Nadine Chang and Karan Sapra and Amala Sanjay Deshmukh and Tuomas Rintamaki and Matthieu Le and Ilia Karmanov and Lukas Voegtle and Philipp Fischer and De-An Huang and Timo Roman and Tong Lu and Jose M. Alvarez and Bryan Catanzaro and Jan Kautz and Andrew Tao and Guilin Liu and Zhiding Yu},
    journal={arXiv:2501.14818},
    year={2025}
}
```

```latex
@inproceedings{shi2025eagle,
    title = {Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders}, 
    author={Min Shi and Fuxiao Liu and Shihao Wang and Shijia Liao and Subhashree Radhakrishnan and De-An Huang and Hongxu Yin and Karan Sapra and Yaser Yacoob and Humphrey Shi and Bryan Catanzaro and Andrew Tao and Jan Kautz and Zhiding Yu and Guilin Liu},
    booktitle={ICLR},
    year={2025}
}
```


## License/Terms of Use
- The code is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file. Portions of the code in this repo are reused and subject to their original licenses. Some files have been modified, with appropriate attribution and additional license headers added where applicable.
- The pretrained Eagle 2 & 2.5 model weights are released under the [NVIDIA License](LICENSE_Model). The models are research preview intended for non-commercial use only, and are subject to the following licenses and terms:
  - Model License of Qwen2.5-7B-Instruct: [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE)
  - Eagle 2 - Model License of PaliGemma SigLIP: [Gemma License](https://ai.google.dev/gemma/terms)
  - Eagle 2 - Model License of CLIP ConvNeXt: [MIT](https://choosealicense.com/licenses/mit/)
  - Eagle 2.5 - Model License of SigLIP2: [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
  - Models are improved using Qwen.
- For code contributions to Eagle, please refer to the [Contribution Guide](CONTRIBUTING.md).
- Users are reminded to ensure that their use of the dataset and model weights is in compliance with all applicable laws and regulations.


## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-HR](https://github.com/luogen1996/LLaVA-HR) and [InternVL](https://github.com/OpenGVLab/InternVL): The Eagle codebase has integrated modified components from these repositories. Many thanks for the great open-source projects.
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): We use derivatives of these repositories for evaluation. Many thanks for the wonderful tools.
- Thanks to [Cambrian](https://cambrian-mllm.github.io), [LLaVA-One-Vision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/), [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) and many other works for the great efforts in open-sourcing data.
- The team would like to give special thanks to the NVIDIA TSE Team, including Chen Fu, Yuchao Jin, Le An, and Josh Park, for their exceptional work on the optimized TensorRT and edge deployment of Eagle.
