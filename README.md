# SPCoral: A package for **Sp**atial **C**ross multi-**O**mics **R**egistration and **A**na**L**ysis

<div align=left><img width = '180' height ='180' src ="LOGO_spcoral.png"/></div>

## Brief Introduction
**SPCoral** is a Python package designed for diagonal integration of spatial multi-omics data from adjacent tissue slices. It comprises two modules: Alignment and Integration. The alignment module employs graph neural networks and Gromov-Wasserstein optimal transport to perform slice-to-slice positional alignment without relying on shared molecular features. The integration module builds upon these aligned coordinates, using a cross-modal attention mechanism to fuse molecular features across different modalities and resolutions. SPCoral supports a variety of downstream applications, including spatial domain identification, cross-omics prediction, spatial cell-cell communication inference, and other spatial multi-omics analyses.

## Pipeline

<div align=center><img width = '800' height ='1200' src ="spcoral_introduction.png"/></div>

## Installation
pip:
```
    pip install spcoral
```
github:
```
    cd spcoral
    python setup.py build
    python setup.py install

```

