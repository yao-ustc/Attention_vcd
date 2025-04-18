
# Attention VDC

Attention VDC modifies the model's understanding of images at the attention level, generating positive and negative sample images at the attention layer. This enables the use of contrastive decoding or mDPO methods to reduce hallucinations.

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

  <h3 align="center">Addressing Visual Model Hallucinations at the Feature Space Level</h3>
  <p align="center">
    Get started with your project quickly!
    <br />
    <a href="https://github.com/yao-ustc/Attention_vcd"><strong>Explore the project documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/yao-ustc/Attention_vcd">View Demo</a>
    ·
    <a href="https://github.com/yao-ustc/Attention_vcd/issues">Report Bug</a>
    ·
    <a href="https://github.com/yao-ustc/Attention_vcd/issues">Request Feature</a>
  </p>

</p>

This README.md is intended for developers.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Frameworks Used](#frameworks-used)
- [Contributors](#contributors)
  - [How to Contribute](#how-to-contribute)
- [Version Control](#version-control)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

1. **Environment Setup**
   ```sh
   conda create -n att_vcd python=3.10 -y
   conda activate att_vcd
   pip install --upgrade pip
   pip install -e .
   ```

2. **Modify Transformers in the Environment**
   [Provide specific instructions for modifying the transformers library if needed.]

### Installation

```sh
git clone https://github.com/yao-ustc/Attention_vcd.git
```

## Directory Structure

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

## Architecture

Please refer to [ARCHITECTURE.md](https://github.com/yao-ustc/Attention_vcd/blob/master/ARCHITECTURE.md) for details on the project's architecture.

## Deployment

Currently, there are no deployment instructions available.

## Frameworks Used

- [LLaVA](https://getbootstrap.com)
- [VCD](https://jquery.com)

## Contributors

Please read **CONTRIBUTING.md** to see the list of developers who have contributed to this project.

### How to Contribute

Contributions make the open-source community a great place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Version Control

This project uses Git for version management. You can view the available versions in the repository.

## Author

zyyao888@ustc.edu

QQ: 2744179563

*You can also view all contributors to this project in the contributors list.*

## License

This project is licensed under the MIT License. See [LICENSE.txt](https://github.com/yao-ustc/Attention_vcd/blob/master/LICENSE.txt) for details.

## Acknowledgments

- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders)

<!-- links -->
[your-project-path]: yao-ustc/Attention_vcd
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
