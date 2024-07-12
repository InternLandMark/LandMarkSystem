# LandMarkSystem
<!-- <p align="center">
    <img src="https://img.shields.io/badge/Trainer-Ready-green"/>
    <img src="https://img.shields.io/badge/Renderer-Ready-green"/>
    <img src="https://img.shields.io/badge/Framework-Ready-green"/>
    <img src="https://img.shields.io/badge/Documentation-Preview-purple"/>
    <img src="https://img.shields.io/badge/License-MIT-orange"/>
</p> -->

<div align="center">

<p align="center">
    <picture>
    <img src="https://raw.githubusercontent.com/InternLandMark/LandMark_Documentation/4f09c93cbec0ad50d27ac52f858e7a6c541168d6/pictures/intern_logo.svg" width="350">
    </picture>
</p>

<p align="center">
    <picture>
    <img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/logo.png?raw=true" width="650">
    </picture>
</p>

<p align="center">
    <a href="https://landmark.intern-ai.org.cn/">
    <font size="4">
    🏠HomePage
    </font>
    </a>
    |
    <a href="./docs/en/quickstart.md">
    <font size="4">
    📘Usage
    </font>
    </a>
    |
    <a href="./docs/en/install.md">
    <font size="4">
    🛠️Installation
    </font>
    </a>
    |
    <a href="landmarksystem.readthedocs.io">
    <font size="4">
    📑Documentation
    </font>
    </a>
    |
    <a href="https://city-super.github.io/gridnerf/">
    <font size="4">
    ✍️TechnicalReport (TBD)
    </font>
    </a>
    |
    </a>
</p>

</div>


### Latest News 🔥

- [2024/07]: We release LandMarkSystem, which is the first open-source system for large-scale scene reconstruction training and rendering.

# 💻 About
LandMarkSystem provides a multi-algorithms system for 3D reconstruction tasks in any scale, offering a high degree of flexibility and efficiency through its modular design and advanced parallelization strategies. This system has the potential to significantly advance the state of the art in large-scale scene reconstruction by enabling the deployment of NeRF and 3DGS algorithms in a manner that is both scalable and resource-efficient.

# 🎨 Support Features

| Model         | Training       |                |                |                | Rendering      |                |                |                |                |
|---------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|               | Single GPU     | Model Parallel | Data Parallel  | Dynamic Loading| Single GPU     | Model Parallel | Data Parallel  | Dynamic Loading| Kernel Optimization   |
| GridNeRF      | ✅              | ✅              | ✅              | -              | ✅              | ✅              | ✅              | ✅              | -              |
| InstantNGP    | ✅              | -              | ✅              | -              | ✅              | -              | ✅              | ✅              | -              |
| Nerfacto      | ✅              | -              | ✅              | -              | ✅              | -              | ✅              | ✅              | -              |
| Vanilla GS    | ✅              | -              | ⌛              | ⌛             | ✅              | -              | ⌛              | ⌛             | ✅              |
| Scaffold GS   | ✅              | -              | ⌛              | ⌛             | ✅              | -              | ⌛              | ⌛             | ✅              |
| Octree GS     | ✅              | -              | ⌛              | ⌛             | ✅              | -              | ⌛              | ⌛             | ✅              |

In the table above, ✅ indicates that the feature is supported in the current version, and ⌛ indicates that the feature will be supported in the next open-source release.

# 🚀 Quickstart

Please refer to [Usage](./docs/en/quickstart.md) to start installation, training and rendering.

For more details, please check [Documentation](landmarksystem.readthedocs.io).

# 📖Learn More

For more information about the system architecture and training/rendering performance, please refer to our technical report (TBD).

<!-- # 🤝 Authors
The main work comes from the LandMark Team, Shanghai AI Laboratory.<br>

<img src="https://github.com/InternLandMark/LandMark_Documentation/blob/main/pictures/shailab_logo2.jpg?raw=true" width="450">

Here are our honorable Contributors:

<a href="https://github.com/InternLandMark/LandMarkSystem/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=InternLandMark/LandMarkSystem" />
</a> -->