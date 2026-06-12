<div align="center">

#  🦅  Eagle: Frontier Vision-Language Models with Data-Centric Strategies

<p>
    <img src="Eagle/assets/Eagle.png" alt="Eagle" width="500" height="auto">
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-NVIDIA%20License-red.svg)](./Eagle2_5/LICENSE_MODEL)

[[📘Eagle Report](Eagle/Eagle.pdf)] [[📘Eagle 2 Report](Eagle2_5/Eagle2.pdf)] [[📘Eagle 2.5 Report](Eagle2_5/Eagle2.5.pdf)] [[📘LocateAnything Report](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)]

[[🤗Model Collection](https://huggingface.co/collections/nvidia/eagle)] [[🤗LocateAnything Demo](https://huggingface.co/spaces/nvidia/LocateAnything)] [[🌐Project Page](https://nvlabs.github.io/Eagle/)]

</div>


## Updates
- [2026/06] 🔥 LocateAnything now supports [batch inference](./Embodied/) with a pure FlashAttention runtime — efficient inference on A100, RTX 4090, and other non-Hopper/Blackwell GPUs.
- [2026/05] 🔥 Release [LocateAnything](./Embodied/) — A generalist vision-language grounding model based on Eagle.
- [2025/12] 🎉 A native resolution variant of the Eagle model is adopted as the VLM backbone of [GR00T-N1.6](https://huggingface.co/collections/nvidia/gr00t-n16). Check out the [tech blog](https://research.nvidia.com/labs/gear/gr00t-n1_6/) for more details.
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

### 🌟 Get Started
- 📚 [Getting started with LocateAnything](./Embodied/README.md)
- 📚 [Getting started with Eagle 2.5](./Eagle2_5/document/0.onboarding.md)
- 📚 [README for Eagle](./Eagle/README.md)

## The Eagle VLM Family

**Eagle** is a family of frontier vision-language models (VLMs) from NVIDIA that explore data-centric strategies across general-purpose multimodal understanding, long-context reasoning, and embodied applications.

Beyond advancing multimodal foundation model research, Eagle has also served as a research and development platform that supports multiple flagship NVIDIA efforts across enterprise intelligence and Physical AI, including but not limited to [Llama-Nemotron Nano VLM](https://huggingface.co/blog/nvidia/llama-nemotron-nano-vl), [Nemotron VLMs](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) ([V2 Nano VL](https://arxiv.org/abs/2511.03929) / [V3 Nano Omni](https://arxiv.org/abs/2604.24954)), [NeMo Retriever](https://developer.nvidia.com/nemo-retriever) ([Llama Nemoretriever Colembed](https://arxiv.org/abs/2507.05513)), [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t) [N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/) ([Tech Report](https://arxiv.org/abs/2503.14734)) / [N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/) / [N1.6](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/), and [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/).


| Model | Features | Summary |
|:--|:--|:--|
| **[LocateAnything](./Embodied/)** | Generalist grounding, detection and pointing | Fast & high quality vision-language grounding with Parallel Box Decoding |
| **[Eagle 2.5](./Eagle2_5/)** | Frontier VLM with SOTA image & video understanding | Framework & data strategy for long-context multimodal understanding |
| **[Eagle 2](./Eagle/README.md)** | Frontier VLM with SOTA image understanding | Exploring the post-training data strategies for frontier VLMs |
| **[Eagle](./Eagle/README.md)** | VLMs with mixture-of-encoders | Exploring the design space for vision-centric VLMs |


## Features and Capabilities

<div align="center">

### Generalist Grounding & Understanding

</div>

<table>
<tr>
<td width="60.4%" align="center" valign="top">
<video src="https://github.com/user-attachments/assets/814e042c-baf4-41ba-b7c9-655e909f82d6" autoplay loop muted playsinline controls width="100%"></video>

<b>Dense Object Detection</b><br>
<sub>LocateAnything performs diverse localization tasks under a unified VLM — document understanding, GUI grounding, dense object detection, and OCR.</sub>
</td>
<td width="39.6%" align="center" valign="top">
<video src="https://github.com/user-attachments/assets/154b5f61-e26e-451b-9518-88c63d437cc4" autoplay loop muted playsinline controls width="100%"></video>

<b>Fast Decoding Speed</b><br>
<sub>Parallel Box Decoding (PBD) vs. Quantized Coordinate Decoding — PBD predicts each bounding box atomically in a single forward pass for substantially faster throughput.</sub>
</td>
</tr>
</table>




<div align="center">

### Long-Context Multimodal Understanding & Reasoning

</div>

<table>
<tr>
<td width="50%" align="center"  valign="middle">
<video src="https://github.com/user-attachments/assets/17cb1630-758a-436b-b92a-5c1d1fda6b12" autoplay loop muted playsinline controls width="100%"></video>

<b>User Prompt</b><br>
<sub>Analyze the video and divide it into distinct sections. For each section, create a title using a few words or phrases, and a detailed caption describing the content of the section, and indicate when it begins in seconds. Multiple sections are separated by line breaks..</sub>
</td>
<td width="50%" align="left" valign="top">
<div align="center">
<b>VLM Captioning</b>
</div>
<sub>
1.05 seconds, Introduction<br>
detailed caption: The clip begins with a majestic scene of a warrior standing atop a mountaintop, surrounded by mystical energy. The warrior is adorned in ornate, decorated armor with intricate patterns and elaborate designs. The background features a misty, ethereal landscape with mountains in the distance. The warrior then climbs a large stone adorned with intricate engravings. The stone is massive, set amidst a mystical and serene environment with other towering rocks nearby. As the warrior ascends, the camera zooms in to reveal the detailed engravings on the stone, showcasing its ancient and mystical nature. The scene transitions to the warrior reaching the top of the stone and spreading his arms wide, standing victorious. Suddenly, the focus shifts to a mystical figure in flowing robes, who appears to be a sage or a mystical character, standing amidst the rocky landscape. This figure is illuminated by an otherworldly light, suggesting his power and wisdom. The clip concludes with this mystical character appearing calm and serene, hinting at a connection to the warrior and the stone, underscoring themes of power, wisdom, and mystical journey.<br>
</sub><br>
<details>
<summary><b>Show More</b></summary>
<sub>
5.99 seconds, Explaining game genre<br>
detailed caption: The clip begins with a character dressed in ornate armor, moving stealthily through a forest and up a mountain. The camera focuses on the intricate details of the armor, which has elaborate patterns and is adorned with gold accents. As the character ascends, a large, mystical landscape with towering trees and distant mountains comes into view. The character is then seen standing on the mountaintop, with an ethereal glow surrounding them. The camera shifts to show a wide view of the sky with clouds, creating a dramatic backdrop. Next, the character engages in combat, wielding a large weapon amidst an army dressed similarly, with a focus on their coordinated attack. The action intensifies as the character fights against a large, stone statue, which has a menacing expression and rough, textured surface. The clip continues with another character in white robes, who seems to be casting spells or invoking some form of power. The environment transitions to a snowy battlefield where the main character battles against another warrior, engaging in dynamic combat moves. The scene is filled with dramatic lighting effects, showing the two warriors clashing amidst a snowy landscape with large statues looming in the background. The clip concludes with the two characters continuing their intense battle, with the main character executing elaborate and powerful strikes.<br><br>
517.10 seconds, Discussing controls<br>
detailed caption: The clip begins with two warriors locked in combat, using long, decorated staffs to strike each other in an icy landscape. The scene is intense, with the warriors demonstrating precise and powerful movements. As they clash, the camera shifts to focus on a mystical figure wearing ornate armor and a crown, who appears to be engaging in a magical ritual or challenge. The warrior in white robes seems to be performing a spell, summoning glowing lights and energy. The camera captures the intricate details of the armor, showing dragon and lion motifs intricately etched into the metal. As the scene progresses, the warrior in white robes is seen riding a large, mystical beast, which attacks the main warrior, creating a swirling vortex of energy. The background features a mountainous landscape, adding to the epic and fantastical atmosphere. The clip concludes with the warrior in white robes standing triumphantly, holding his staff aloft, as the spell they cast manifests before them, demonstrating their power and mastery over the mystical forces.<br><br>
614.53 seconds, Talking about story & characters<br>
detailed caption: The clip begins with a wide shot of a vast forest with tall trees, and mountains visible in the distance partially covered by mist. The sky is bright and clear, suggesting a peaceful morning. The scene then transitions to a close-up of a mystical, swirling black rock with intricate patterns upon its surface. A hand, presumably belonging to a character named Wukong, reaches out and touches the rock, causing it to shift and reveal a hidden passage. As the hand touches the rock, a transformation appears to begin, symbolizing awakening or revelation. Next, the scene shifts to a different location where Wukong stands on a rocky terrain with trees and a mountainous backdrop, suggesting a connection to nature and ancient wisdom. Wukong is dressed in simple, flowing robes, and with an air of contemplation, gazes off into the distance. The clip transitions to another scene where Wukong is now seen in a more rugged outfit, equipped with a large staff, indicating a readiness for adventure or confrontation. The clip captures the essence of a journey filled with mystery, self-discovery, and the pursuit of enlightenment. Throughout the clip, text appears in Chinese characters, providing context and narrative to the unfolding story, including the names 'Wukong' (in both the original script and pinyin) and 'Peng Lao Dang,' adding depth to the characters and their world.<br><br>
698.69 seconds, Describing visuals & sound<br>
detailed caption: The clip begins with an ancient stone structure partially submerged in the ocean, its smooth, curved surface hinting at its historical significance. As the camera pans out, large waves crash against the structure, creating a powerful and dramatic atmosphere. The water is a deep blue, and the waves crash with great force, splashing white foam around. In the next scene, the setting shifts to a lush forest at dusk, where dense foliage is illuminated by golden light filtering through the trees. The scene captures the tranquility and beauty of nature, with the play of light and shadow creating a mystical ambiance. The forest then transforms into a rocky mountainous terrain, where a waterfall cascades down a cliff. The water is clear, and the rocks are rugged, emphasizing the raw power of nature. Finally, the clip returns to the mystical realm, where a large, ancient stone stands in a serene landscape. This stone is covered in intricate carvings and stands as a testament to the enduring legacy of the place. The camera lingers on the details of the carvings, highlighting the artistry and craftsmanship of the ancient civilization. The clip concludes with a close-up of the stone, capturing the fine details and the texture of the carvings, leaving a lasting impression of the historical and natural beauty depicted throughout the clip.<br><br>
738.62 seconds, Closing<br>
detailed caption: The clip begins with a character with white hair and a beard, wearing a loose robe and a beaded necklace, extending their hand towards a large, swirling rock on a rocky mountain. The rock features intricate designs that appear to be swirling patterns. The action proceeds as the character touches the rock, and the rock starts to glow with a mystical light, suggesting a magical transformation or unlocking of power. As the character's hand makes contact with the rock, the surroundings become more luminous, highlighting the magical ambiance. The scene transitions to the character walking away from the rock, moving down a series of rocky steps into a valley. As the character descends, more characters dressed in similar attire come into view, implying a gathering or a quest. The video captures the mystical atmosphere of the mountainous landscape, emphasizing the character's interaction with the rock and the mystical surroundings.
</sub>
</details>

</td>
</tr>
</table>
<div align="center">

### Applications across Virtual and Physical Worlds

</div>
<table>
<tr>
<td width="41.7%" align="center">
<img src="./Eagle2_5/assets/Population.png" width="100%">
<br>
<b>Document & General VQA</b>

<sub>**Question:** Based on this graph, how did the population of Puerto Rico change over the last decade? **Answer:** The graph illustrates the population growth of each state in the United States between 2010 and 2020. The majority of states experienced an increase, with only two experiencing a decline. Puerto Rico is among these two states. The number of people living in Puerto Rico decreased by approximately 365,000 over the last decade. This represents a decrease of about 12%.</sub>
</td>
<td width="58.3%" align="center">
<img src="./Embodied/assets/images/LocateAnything3D.png" width="100%">
<b>3D Perception & Spatial Intelligence</b>

<sub>Examples of 3D object detection in the wild by LocateAnything3D.</sub>
</td>
</tr>
</table>


<div align="center">
<img src="./Embodied/assets/images/Smart_City.png" width="98.5%">
<b>Smart City & Metropolis</b><br>
<sub> An example of zero-shot ultra-dense pedestrian detection in the wild for a road crossing in Shibuya, Tokyo, one of the busiest areas in the world.</sub>
</div>


## Model Zoo

### 📦 LocateAnything Models
| Model Name  | Date       |   LLM Backbone   |  Vision Encoder  | Max Length | Download |
| ----------- |------------| ---------------- | ---------------- | ---------- | ------- |
| LocateAnything-3B | 2026.05.26 | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | [MoonViT-SO-400M](https://huggingface.co/moonshotai/MoonViT-SO-400M) | 25K | 🤗 [HF](https://huggingface.co/nvidia/LocateAnything-3B) |

### 📦 Eagle 2.5 Models
| Model Name  | Date       |   LLM Backbone   |  Vision Encoder  | Max Length | Download |
| ----------- |------------| ---------------- | ---------------- | ---------- | ------- |
| Eagle2.5-8B | 2025.04.16 | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-512) | 128K | 🤗 [HF](https://huggingface.co/nvidia/Eagle2.5-8B) |

### 📦 Eagle 2 Models
| Model Name  | Date       |   LLM Backbone   |  Vision Encoder  | Max Length | Download |
| ----------- |------------| ---------------- | ---------------- | ---------- | ------- |
| Eagle2-1B | 2025.01.11 | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448)  | 16K | 🤗 [HF](https://huggingface.co/nvidia/Eagle2-1B) |
| Eagle2-2B | 2025.01.11 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448)  | 16K | 🤗 [HF](https://huggingface.co/nvidia/Eagle2-2B) |
| Eagle2-9B | 2025.01.11 | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)     |  [SigLIP](https://huggingface.co/google/paligemma-3b-pt-448) + [ConvNext](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)  | 16K | 🤗 [HF](https://huggingface.co/nvidia/Eagle2-9B) |


### 📦 Eagle Models

| Model Name  | Date       |   LLM Backbone   |  Vision Encoder  | Download |
| ----------- |------------| ---------------- | ---------------- | ---------- |
| Eagle-X4-8B-Plus | 2024.09.16 | [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |  [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)+[ConvNeXt](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)+[EVA](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)+[Pix2Str](https://huggingface.co/google/pix2struct-large) | 🤗 [HF](https://huggingface.co/NVEagle/Eagle-X4-8B-Plus) |
| Eagle-X4-13B-Plus | 2024.09.16 | [vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |  [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)+[ConvNeXt](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)+[EVA](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)+[Pix2Str](https://huggingface.co/google/pix2struct-large) | 🤗 [HF](https://huggingface.co/NVEagle/Eagle-X4-13B-Plus) |
| Eagle-X5-34B-Plus | 2024.09.16 | [Yi-34B](https://huggingface.co/01-ai/Yi-34B) |  [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)+[ConvNeXt](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)+[EVA](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)+[Pix2Str](https://huggingface.co/google/pix2struct-large)+[SAM](https://huggingface.co/facebook/sam-vit-large) | 🤗 [HF](https://huggingface.co/NVEagle/Eagle-X5-34B-Plus) |
| Eagle-X5-7B | 2024.09.16 | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |  [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)+[ConvNeXt](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)+[EVA](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)+[Pix2Str](https://huggingface.co/google/pix2struct-large)+[SAM](https://huggingface.co/facebook/sam-vit-large) | 🤗 [HF](https://huggingface.co/NVEagle/Eagle-X5-7B) |
| Eagle-X5-13B | 2024.09.16 | [vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |  [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)+[ConvNeXt](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)+[EVA](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)+[Pix2Str](https://huggingface.co/google/pix2struct-large)+[SAM](https://huggingface.co/facebook/sam-vit-large) | 🤗 [HF](https://huggingface.co/NVEagle/Eagle-X5-13B) |
| Eagle-X5-13B-Chat | 2024.09.16 | [vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |  [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)+[ConvNeXt](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)+[EVA](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth)+[Pix2Str](https://huggingface.co/google/pix2struct-large)+[SAM](https://huggingface.co/facebook/sam-vit-large) | 🤗 [HF](https://huggingface.co/NVEagle/Eagle-X5-13B-Chat) |



## Citation
If you find this project useful, please consider citing our works:
```latex
@article{wang2025locateanything,
    title={LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding},
    author={Shihao Wang and Shilong Liu and Yuanguo Kuang and Xinyu Wei and Yangzhou Liu and Zhiqi Li and Yunze Man and Guo Chen and Andrew Tao and Guilin Liu and Jan Kautz and Lei Zhang and Zhiding Yu},
    journal={arXiv:2605.27365},
    year={2026}
}
```

```latex
@inproceedings{man2025locateanything3d,
    title   = {LocateAnything3D: Vision-Language 3D Detection with Chain-of-Sight},
    author  = {Yunze Man and Shihao Wang and Guowen Zhang and Johan Bjorck and Zhiqi Li and Liang-Yan Gui and Jim Fan and Jan Kautz and Yu-Xiong Wang and Zhiding Yu},
    journal = {arXiv preprint arXiv:2511.20648},
    year    = {2025},
}
```

```latex
@inproceedings{chen2025eagle2.5,
    title={Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models},
    author={Guo Chen and Zhiqi Li and Shihao Wang and Jindong Jiang and Yicheng Liu and Lidong Lu and De-An Huang and Wonmin Byeon and Matthieu Le and Max Ehrlich and Tong Lu and Limin Wang and Bryan Catanzaro and Jan Kautz and Andrew Tao and Zhiding Yu and Guilin Liu},
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
- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file. Portions of the code in this repo are reused and subject to their original licenses. Some files have been modified, with appropriate attribution and additional license headers added where applicable.
- The pretrained model weights are released under either the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/deed.en) or the [NVIDIA License](./Eagle2_5/LICENSE_MODEL). The models are research preview intended for non-commercial use only.
- Eagle models are improved using Qwen.
- For code contributions to Eagle, please refer to the [Contribution Guide](CONTRIBUTING.md).
- Users are reminded to ensure that their use of the dataset and model weights is in compliance with all applicable laws and regulations.


## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-HR](https://github.com/luogen1996/LLaVA-HR) and [InternVL](https://github.com/OpenGVLab/InternVL): The Eagle codebase has integrated modified components from these repositories. Many thanks for the great open-source projects.
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): We use derivatives of these repositories for evaluation. Many thanks for the wonderful tools.
- Thanks to [Cambrian](https://cambrian-mllm.github.io), [LLaVA-One-Vision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/), [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) and many other works for the great efforts in open-sourcing data.
- The team would like to give special thanks to the NVIDIA TSE Team, including Chen Fu, Yuchao Jin, Le An, and Josh Park, for their exceptional work on the optimized TensorRT and edge deployment of Eagle.