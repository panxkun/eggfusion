# EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly

<p align="center">
    <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-Paper-red.svg?style=flat-square" alt="arXiv Paper"/>
    </a>
    <a href="https://zju3dv.github.io/eggfusion/" target="_blank">
        <img src="https://img.shields.io/badge/Project%20Page-eggfusion-blue.svg?style=flat-square" alt="Project Page"/>
    </a>
    <a href="LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="Apache License 2.0"/>
    </a>
</p>


This repository contains the implementation of following paper:

> [**EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly**](https://arxiv.org/abs/xxxx.xxxxx)<br>
> ***SIGGRAPH ASIA 2025***, HongKong<br>
> [Xiaokun Pan](https://github.com/panxkun)<sup>1</sup>, Zhenzhe Li<sup>1</sup>, [Zhichao Ye](https://github.com/oneLOH)<sup>2</sup>, [Hongjia Zhai](https://zhaihongjia.github.io/)<sup>1</sup>, [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)<sup>1</sup> <br>
> **<sup>1</sup> State Key Lab of CAD&CG, Zhejiang University <sup>2</sup>SenseTime Research** <br>
> (Corresponding authors: Guofeng Zhang and Zhichao Ye)<br>

![Teaser](assets/teaser.png)

## Overview
<p align="center">
    <img src="assets/overview.gif" alt="Overview" style="max-width: 100%; height: auto;">
</p>

EGG-Fusion is a real-time 3D reconstruction system leverages geometry-aware Gaussian surfels for real-time dense mapping and tracking. 



## License

EGG-Fusion is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Citation

If you found this code/work to be useful in your own research, please considering citing the following:
```bibtex
@inproceedings{eggfusion2025,
  title     = {{EGG-Fusion}: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly},
  author    = {Pan, Xiaokun and Li, Zhenzhe and Ye, Zhichao and Zhai, Hongjia and Zhang, Guofeng},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  year      = {2025},
  month     = dec,
  address   = {Hong Kong, Hong Kong},
  publisher = {ACM},
  pages     = {1--11},
  doi       = {10.1145/3757377.3763878},
}
``