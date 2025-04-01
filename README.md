# EcoDatum -- [Quality over Quantity: Boosting Data Efficiency Through Ensembled Multimodal Data Curation]

This repository provides the implementation of **EcoDatum**, a data curation framework introduced in the paper [Quality over Quantity: Boosting Data Efficiency Through Ensembled Multimodal Data Curation](https://arxiv.org/abs/2502.08211) by Jinda Xu et al. EcoDatum enhances dataset quality by integrating various unimodal and multimodal data curation operators within a weak supervision ensemble framework, leading to improved model training efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Achievement](#achievement)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the era of big data, effectively curating web-crawled datasets is crucial for optimizing model performance. Traditional heuristic curation methods often fail to capture complex features, leading to biases and the exclusion of relevant data. EcoDatum addresses these challenges by strategically integrating various data curation operators within a weak supervision ensemble framework, utilizing automated optimization to score each data point effectively. This approach significantly improves data curation quality and efficiency, outperforming existing state-of-the-art techniques.

## Features

- **Ensembled Multimodal Data Curation**: Combines multiple data curation operators to enhance dataset quality.
- **Quality-Guided Deduplication**: Ensures balanced feature distributions by removing redundant data based on quality metrics.
- **Automated Optimization**: Utilizes a composite metric and a small labeled dataset to fine-tune the integration of curation operators.
- **Improved Model Training Efficiency**: Demonstrated to enhance model performance across diverse evaluation datasets.

## Installation

To use EcoDatum, clone this repository and install the required dependencies:

```bash
git clone git@github.com:Daming-W/ecodatum.git
cd ecodatum
pip install -r requirements.txt
```

## Usage

EcoDatum can be used to curate datasets before training visual-language models. Here's a basic example of how to apply EcoDatum to your dataset:


For detailed usage and customization options, please refer to the [documentation](https://github.com/yourusername/ecodatum/docs).

## Achievement

EcoDatum has been evaluated on the [DataComp leaderboard](https://www.datacomp.ai/dcclip/leaderboard.html), achieving an average performance score of 0.182 across 38 diverse evaluation datasets. This represents a 28% improvement over the DataComp baseline method, demonstrating its effectiveness in improving dataset curation and model training efficiency.

## Contributing

We welcome contributions to EcoDatum! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For more information, please refer to our [paper](https://arxiv.org/abs/2502.08211).
