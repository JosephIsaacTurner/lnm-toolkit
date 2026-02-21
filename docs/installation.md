# Installation

To get started with `lnm-toolkit`, you'll need to install the project and its dependencies.

## 1. Prerequisites

The `lnm-toolkit` depends on:
- Python 3.8+
- [PRISM](https://github.com/JosephIsaacTurner/prism) library (must be installed manually)
- Standard scientific libraries: `pandas`, `numpy`, `scikit-learn`, `nilearn`, `nibabel`, `scipy`

## 2. Installing PRISM

Since `prism` is not currently available on PyPI, you will need to clone and install it manually:

```bash
git clone https://github.com/JosephIsaacTurner/prism
cd prism
pip install -e .
```

## 3. Installing lnm-toolkit

Next, clone the `lnm-toolkit` repository and install it in editable mode:

```bash
git clone https://github.com/JosephIsaacTurner/lnm-toolkit
cd lnm-toolkit
pip install -e .
```

## 4. Documentation Setup (Optional)

If you'd like to build the documentation locally, install the documentation dependencies:

```bash
pip install mkdocs-material mkdocstrings[python]
```

To serve the docs:

```bash
mkdocs serve
```
