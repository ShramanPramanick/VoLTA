# VoLTA: Vision-Language Transformer with Weakly-Supervised Local-Feature Alignment

[**VoLTA: Vision-Language Transformer with Weakly-Supervised Local-Feature Alignment**]()                                     
Shraman Pramanick*, Li Jing*, Sayan Nag*, Jiachen Zhu, Hardik J Shah, Yann LeCun, Rama Chellappa                
Transactions on Machine Learning Research (TMLR), 2023     
[arxiv](https://arxiv.org/abs/2210.04135) | [project page](https://shramanpramanick.github.io/VoLTA/)

> **TL;DR:** We introduce VoLTA (Vision-Language Transformer with weakly-supervised local-feature Alignment), a new vision-language pre-training (VLP) paradigm that only utilizes image-caption data but achieves fine-grained region-level image understanding, eliminating the use of expensive bounding box annotations.

<img src="/Figures/VoLTA_overview.png" alt="VoLTA" width="820" />

## 📢 News

- [October, 2023] We release the first version of the VoLTA codebase.
- [August, 2023] VoLTA is accepted by **TMLR**.

## 📁 Repository Structure

The contents of this repository are structured as follows:

```bash
VoLTA
    ├── Pre-training
    ├── Multimodal_Fine_Grained
    │   │── REC
    │   │── LVIS
    │   │── COCO_det
    ├── Multimodal_Coarse_Grained
    │   │── VQAv2
    │   │── NLVR2
    │   │── IRTR
    │   │── Captioning
```

## 🛠️ Environment Preparation

```bash
conda create -n python=3.8.13 volta
conda activate volta
conda install pip
pip install -r requirements.txt
```

## ✉️ Contact
This repository is created and maintained by [Shraman](https://shramanpramanick.github.io/) and [Sayan](https://sayannag.github.io/). Questions and discussions are welcome via spraman3@jhu.edu.

## 🙏 Acknowledgements

The codebase for this work is built on the [FIBER](https://github.com/microsoft/FIBER), [GOT](https://github.com/LiqunChen0606/Graph-Optimal-Transport) and [Barlow Twins](https://github.com/facebookresearch/barlowtwins) repository. We would like to thank the respective authors for their contribution, and the Meta AI team for discussions and feedback. Shraman Pramanick and Rama Chellappa were partially supported by an ONR MURI Grant N00014-20-1-2787.

## 📄 License
VoLTA is licensed under a [MIT License](./LICENSE).

## 🎓 Citation

```
@article{pramanick2023volta,
  title={VoLTA: Vision-Language Transformer with Weakly-Supervised Local-Feature Alignment},
  author={Pramanick, Shraman and Jing, Li and Nag, Sayan and Zhu, Jiachen and Shah, Hardik and LeCun, Yann and Chellappa, Rama},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

