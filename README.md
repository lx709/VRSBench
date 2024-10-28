
<font size='5'>**VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding**</font>

Xiang Li, Jian Ding, Mohamed Elhoseiny

<a href='https://vrsbench.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2406.12384'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href='https://huggingface.co/datasets/xiang709/VRSBench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'>

## Related Projects

<font size='5'>**RSGPT: A Remote Sensing Vision Language Model and Benchmark**</font>

[Yuan Hu](https://scholar.google.com.sg/citations?user=NFRuz4kAAAAJ&hl=zh-CN), Jianlong Yuan, Congcong Wen, Xiaonan Lu, [Xiang Li☨](https://xiangli.ac.cn)

<a href='https://github.com/Lavender105/RSGPT'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2307.15266'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<font size='5'>**Vision-language models in remote sensing: Current progress and future trends**</font>

[Xiang Li☨](https://xiangli.ac.cn), [Congcong Wen](https://wencc.xyz/), [Yuan Hu](https://scholar.google.com.sg/citations?user=NFRuz4kAAAAJ&hl=zh-CN), Zhenghang Yuan, [Xiao Xiang Zhu](https://www.professoren.tum.de/en/zhu-xiaoxiang)

<a href='[https://arxiv.org/abs/2307.15266](https://ieeexplore.ieee.org/abstract/document/10506064/)'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


# VRSBench
<center>
    <img src="fig_example.png" alt="VRSBench is a Versatile Vision-Language Benchmark for Remote Sensing Image Understanding.">
</center>

VRSBench is a Versatile Vision-Language Benchmark for Remote Sensing Image Understanding. It consists of 29,614 remote sensing images with detailed captions, 52,472 object refers, and 3123,221 visual question-answer pairs. It facilitates the training and evaluation of vision-language models across a broad spectrum of remote sensing image understanding tasks. 

## 🗓️ TODO
- [x] **[2024.10.15]** Release evaluation code.
- [x] **[2024.10.15]** Release code and models of baseline models.
- [x] **[2024.06.19]** We release the instructions and code for calling GPT-4V to get initial annotations.
- [x] **[2024.06.19]** We release the VRSBench, A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding. VRSBench contains 29,614 images, with 29,614 human-verified detailed captions, 52,472 object references, and 123,221 question-answer pairs. check [VRSBench Project Page](https://vrsbench.github.io/).

## Using `datasets`

The dataset can be downloaded from [link](https://huggingface.co/datasets/xiang709/VRSBench) and used via the Hugging Face `datasets` library. To load the dataset, you can use the following code snippet:

```python
from datasets import load_dataset
fw = load_dataset("xiang709/VRSBench", streaming=True)
```

## Dataset curation
To construct our VRSBench dataset, we employed multiple data engineering steps, including attribute
extraction, prompting engineering, GPT-4 inference, and human verification. 

- Attribute Extraction: we extract image information, including the source and resolution, as well as object information—such as the object category, bounding box, color, position (absolute and relative), and size (absolute and relative)—from existing object detection datasets. Please check ```extract_patch_json.py``` for attribute extraction details.
- Prompting Engineering: We carefully design instructions to prompt GPT-4V to create detailed image captions, object referring, and question-answer pairs. Please check ```instruction.txt``` for detailed instructions.
- GPT-4 inference: Given input prompts, we call OpenAI API to automatically generate image captions, object referring, and question-answer pairs based on the prompts. Use the ```extract_patch_json.py``` to get initial annotations for image captioning, visual grounding, and VQA tasks using GPT-4V.
- Human verification: To improve the quality of the dataset, we engage human annotators to validate each annotation generated by GPT-4V.


## Model Training
For the above three tasks, we benchmark state-of-the-art models, including [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4), [Mini-Gemini](https://github.com/dvlab-research/MGM), and [GeoChat](https://github.com/mbzuai-oryx/GeoChat), to demonstrate the potential of LVMs for remote sensing image understanding. To ensure a fair comparison, we reload the models that are initially trained on large-scale image-text alignment datasets, and then finetune each method using the training set of our RSVBench dataset. For each comparing method, we finetune the model on the training set of our RSVBench dataset for 5 epochs. Following GeoChat, we use LoRA finetuning to finetune all comparing methods, with a rank of 64. 

Use the ```prepare_geochat_eval_all.ipynb``` to prepare the VRSBench evaluation file for image captioning, visual grounding, and VQA tasks.

## Benchmark Results
The code and checkpoints of baseline models  can be found at [Onedrive]([https://livereadingac-my.sharepoint.com/:f:/g/personal/kv932422_reading_ac_uk/EglwHuEMnLdKnoji9vMts9IBhHjrqgMi1shs1d2KCzizpA?e=Ozb0Ib](https://drive.google.com/drive/folders/1Z6W-Wq-NwKr6UwkZrwgL0NxzrZas0qPI?usp=sharing)).

### Image Captioning Performance
| Method                   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr | Avg_L |
|--------------------------|--------|--------|--------|--------|--------|---------|-------|-------|
| GeoChat w/o ft       | 13.9   | 6.6    | 3.0    | 1.4    | 7.8    | 13.2    | 0.4   | 36    |
| GPT-4V               | 37.2   | 22.5   | 13.7   | 8.6    | 20.9   | 30.1    | 19.1  | 67    |
| MiniGPT-v2           | 36.8   | 22.4   | 13.9   | 8.7    | 17.1   | 30.8    | 21.4  | 37    |
| LLaVA-1.5            | **48.1** | **31.5** | **21.2** | **14.7** | **21.9** | **36.9** | **33.9** | 49    |
| GeoChat              | 46.7   | 30.2   | 20.1   | 13.8   | 21.1   | 35.2    | 28.2  | 52    |
Mini-Gemini     | 47.6   | 31.1   | 20.9   | 14.3   | 21.5   | 36.8    | 33.5  | 47    |

**Caption**: Detailed image caption performance on the VRSBench dataset. Avg\_L denotes the average word length of generated captions.

### Visual Grounding Performance
| Method          | Acc@0.5 (Unique) | Acc@0.7 (Unique) | Acc@0.5 (Non Unique) | Acc@0.7 (Non Unique) | Acc@0.5 (All) | Acc@0.7 (All) |
|-----------------|------------------|------------------|----------------------|----------------------|---------------|---------------|
| GeoChat w/o ft  | 20.7             | 5.4              | 7.3                  | 1.7                  | 12.9          | 3.2           |
| GPT-4V          | 8.6              | 2.2              | 2.5                  | 0.4                  | 5.1           | 1.1           |
| MiniGPT-v2      | 40.7             | 18.9             | 32.4                 | 15.2                 | 35.8          | 16.8          |
| LLaVA-1.5       | 51.1             | 16.4             | 34.8                 | 11.5                 | 41.6          | 13.6          |
| GeoChat         | 57.4             | 22.6             | 44.5                 | 18.0                 | 49.8          | 19.9          |
| Mini-Gemini     | 41.1             | 9.6              | 22.3                 | 4.9                  | 30.1          | 6.8           |

**Caption**: Visual grounding performance on the papernameAbbrev dataset. Boldface indicates the best performance.

### Visual Question Answering Performance
| Method         | Category | Presence | Quantity | Color | Shape | Size | Position | Direction | Scene | Reasoning | All   |
|----------------|----------|----------|----------|-------|-------|------|----------|-----------|-------|-----------|-------|
| # VQAs         | 5435     | 7789     | 6374     | 3550  | 1422  | 1011 | 5829     | 477       | 4620  | 902       |       |
| GeoChat w/o ft | 48.5     | 85.9     | 19.2     | 17.0  | 18.3  | 32.0 | 43.4     | 42.1      | 44.2  | 57.4      | 40.8  |
| GPT-4V         | 67.0     | 87.6     | 45.6     | 71.0  | 70.8  | 54.3 | 67.2     | 50.7      | 69.8  | 72.4      | 65.6  |
| MiniGPT-v2     | 61.3     | 26.0     | 46.1     | 51.0  | 41.8  | 11.2 | 17.1     | 12.4      | 49.3  | 21.9      | 38.2  |
| LLaVA-1.5      | 86.9     | 91.8     | 58.2     | 69.9  | 72.2  | 61.5 | 69.5     | 56.7      | 83.9  | 73.4      | 76.4  |
| GeoChat        | 86.5     | 92.1     | 56.3     | 70.1  | 73.8  | 60.4 | 69.3     | 53.5      | 83.7  | 73.5      | 76.0  |
| Mini-Gemini    | 87.8     | 92.1     | 58.8     | 74.0  | 75.3  | 58.0 | 68.0     | 56.7      | 83.2  | 74.4      | 77.8  |

**Caption**: Visual question answering performance on the VRSBench dataset. Boldface indicates the best performance.

## Licensing Information
The dataset is released under the [CC-BY-4.0]([https://creativecommons.org/licenses/by-nc/4.0/deed.en](https://creativecommons.org/licenses/by/4.0/deed.en)), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

## Related Projects
- RSGPT. The first GPT-based Large Vision-Language Model in remote sensing. [RSGPT](https://github.com/Lavender105/RSGPT)
- RS-CLIP. A CLIP-based Vision-Language Model for remote sensing scene classification. [RS-CLIP](https://www.sciencedirect.com/science/article/pii/S1569843223003217) 
- Survey. A comprehensive survey about vision-language models in remote sensing. [RSVLM](https://arxiv.org/pdf/2305.05726.pdf).
- MiniGPT-v2. [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)

## 📜 Citation

```bibtex
@article{li2024vrsbench,
  title={VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding},
  author={Xiang Li, Jian Ding, and Mohamed Elhoseiny},
  journal={arXiv:2406.12384},
  year={2024}
}
```

## 🙏 Acknowledgement
Our VRSBench dataset is built based on [DOTA-v2](https://captain-whu.github.io/DOTA/dataset.html) and [DIOR](https://gcheng-nwpu.github.io/#Datasets) datasets.

We are thankful to [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4), [Mini-Gemini](https://github.com/dvlab-research/MGM), and [GeoChat](https://github.com/mbzuai-oryx/GeoChat) for releasing their models and code as open-source contributions.

