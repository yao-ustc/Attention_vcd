

# Attention VDC

 Attention VDC 从注意力层面修改模型对图片的理解认知，得到注意力层面的正负样本图像，从而使用对比解码或mDPO方法，减小幻觉

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/yao-ustc/Attention_vcd/">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">特征空间层面解决视觉模型幻觉</h3>
  <p align="center">
    快速开始你的项目！
    <br />
    <a href="https://github.com/yao-ustc/Attention_vcd"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/yao-ustc/Attention_vcd">查看Demo</a>
    ·
    <a href="https://github.com/yao-ustc/Attention_vcd/issues">报告Bug</a>
    ·
    <a href="https://github.com/yao-ustc/Attention_vcd/issues">提出新特性</a>
  </p>

</p>


 本篇README.md面向开发者
 
## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [开发的架构](#开发的架构)
- [部署](#部署)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
  - [如何参与开源项目](#如何参与开源项目)
- [版本控制](#版本控制)
- [作者](#作者)
- [鸣谢](#鸣谢)

### 上手指南



###### 开发前的配置要求

1. 环境配置
```sh
conda create -n att_vcd python=3.10 -y
conda activate att_vcd
pip install --upgrade pip 
pip install -e .
```
2. 更改环境中的transformers
   

###### **安装步骤**


```sh
git clone https://github.com/yao-ustc/Attention_vcd.git
```

### 文件目录说明
eg:

```
filetree 
├── ARCHITECTURE.md
├── LICENSE.txt
├── README.md
├── /account/
├── /bbs/
├── /docs/
│  ├── /rules/
│  │  ├── backend.txt
│  │  └── frontend.txt
├── manage.py
├── /oa/
├── /static/
├── /templates/
├── useless.md
└── /util/

```





### 开发的架构 

请阅读[ARCHITECTURE.md](https://github.com/yao-ustc/Attention_vcd/blob/master/ARCHITECTURE.md) 查阅为该项目的架构。

### 部署

暂无

### 使用到的框架

- [LLaVA](https://getbootstrap.com)
- [VCD](https://jquery.com)

### 贡献者

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

zyyao888@ustc.edu

qq:2744179563  

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/yao-ustc/Attention_vcd/blob/master/LICENSE.txt)

### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders)

<!-- links -->
[your-project-path]:yao-ustc/Attention_vcd
[contributors-shield]: https://img.shields.io/github/contributors/yao-ustc/Attention_vcd.svg?style=flat-square
[contributors-url]: https://github.com/yao-ustc/Attention_vcd/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/yao-ustc/Attention_vcd.svg?style=flat-square
[forks-url]: https://github.com/yao-ustc/Attention_vcd/network/members
[stars-shield]: https://img.shields.io/github/stars/yao-ustc/Attention_vcd.svg?style=flat-square
[stars-url]: https://github.com/yao-ustc/Attention_vcd/stargazers
[issues-shield]: https://img.shields.io/github/issues/yao-ustc/Attention_vcd.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/yao-ustc/Attention_vcd.svg
[license-shield]: https://img.shields.io/github/license/yao-ustc/Attention_vcd.svg?style=flat-square
[license-url]: https://github.com/yao-ustc/Attention_vcd/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian




