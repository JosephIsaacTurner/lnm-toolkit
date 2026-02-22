# Installation

To get started with `lnm-toolkit`, you'll need to install the project and its dependencies.

## 1. Prerequisites

The `lnm-toolkit` requires:
- Python 3.9+
- Standard scientific libraries: `pandas`, `numpy`, `scikit-learn`, `nilearn`, `nibabel`, `scipy`
- **PRISM**: The permutation-based statistical backend.

## 2. Quick Install

The easiest way to install `lnm-toolkit` and its dependencies (including `prism-neuro`) is directly from the source repository:

```bash
git clone https://github.com/JosephIsaacTurner/lnm-toolkit
cd lnm-toolkit
pip install -e .
```

This will automatically install `prism-neuro` from PyPI.

## 3. Alternative: Manual PRISM Installation

If you need the latest development version of PRISM, you can install it manually before installing the toolkit:

```bash
pip install prism-neuro
```

Or from source:

```bash
git clone https://github.com/JosephIsaacTurner/prism
cd prism
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
