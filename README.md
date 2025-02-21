<p align="center">
  <a href="https://github.com/GesturXYZ/gestur">
    <img src="./logo.png" alt="Gestur banner" width="230" height="100" style="margin-bottom: -15px;">
  </a>
  <h3 align="center">Gestur - 3D Interaction Without Hardware</h3>
  <p align="center">
    Gestur is an innovative software solution that enables 3D navigation and interaction using only a camera and hand gestures, eliminating the need for traditional input devices.
    <br>
    <a href="https://github.com/GesturXYZ/gestur/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://github.com/GesturXYZ/gestur/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p>

## Table of contents

- [Introduction](#introduction)
- [Features](#features)
- [What's included](#whats-included)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Bugs and feature requests](#bugs-and-feature-requests)
- [Creators](#creators)
- [Collaborators](#collaborators)
- [License](#license)

## Introduction

Gestur is a software-based alternative to hardware controllers like the SpaceMouse, allowing precise manipulation of 3D environments using only a camera. Designed for professionals in CAD, animation, and 3D modeling, Gestur offers an intuitive and natural interface for interacting with digital spaces.

## Features

* Utilizes the [MonocularRGB 3D Hand Pose](https://github.com/FORTH-ModelBasedTracker/MonocularRGB_3D_Handpose_WACV18) library for precise hand tracking.
* Real-time hand gesture recognition for 3D navigation.
* No external hardware required—just a camera.
* Compatible with major 3D modeling software.
* Customizable gesture mappings for different workflows.
* Open-source and extensible for additional features.

## What's included

Within the repository, you'll find:

```text
root/
├── gestur/
│   ├── ...
├── docs/
├── examples/
├── LICENSE
├── vendor/  # Contains third-party libraries like MonocularRGB 3D Hand Pose
├── logo.png
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

### Prerequisites
Ensure you have the following installed:

- Python 3.9 or higher
- OpenCV
- MediaPipe
- NumPy

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Setup
Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/xavi-burgos99/gestur.git
cd gestur
```

## Usage

To start Gestur, simply run:

```sh
python main.py
```

Position your hand within the camera frame, and use gestures to navigate the 3D environment.

## Configuration

Gestur allows customization of gestures and controls via a JSON configuration file:

```json
{
  "sensitivity": 1.2,
  "gestures": {
    "pan": "swipe",
    "zoom": "pinch",
    "rotate": "twist"
  }
}
```

Modify `config.json` to adjust settings as needed.

## Bugs and feature requests

Have a bug or feature request? Please search existing issues before opening a new one:

- [Report a bug](https://github.com/GesturXYZ/gestur/issues/new?template=bug.md)
- [Request a feature](https://github.com/GesturXYZ/gestur/issues/new?template=feature.md&labels=feature)

## Creators

This project was developed by **Xavi Burgos** as part of the [Gestur](https://github.com/GesturXYZ) initiative.

### Xavi Burgos
- Website: [xburgos.es](https://xburgos.es)
- LinkedIn: [@xavi-burgos](https://www.linkedin.com/in/xavi-burgos/)
- GitHub: [@xavi-burgos99](https://github.com/xavi-burgos99)
- X (Twitter): [@xavi_burgos14](https://x.com/xavi_burgos14)

## Collaborators
Special thanks to contributors who support the project.

## License
This project is licensed under a custom license that allows **use and modification** for any purpose, with attribution required, but **prohibits distribution** without permission. See the [LICENSE](LICENSE) file for details.

