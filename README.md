# Spectral Mechanics Analysis

This repository contains tools for analyzing the mechanical properties of materials, particularly biological samples, using spectral analysis techniques. It's designed to work with trajectory data from particle tracking experiments, such as those obtained from dark-field microscopy of microbeads in cells or hydrogels.

## Context

Understanding the mechanical properties of biological materials is crucial for many areas of research, from cell biology to tissue engineering. Traditional methods for probing these properties often require specialized equipment like optical tweezers. This package aims to provide a more accessible approach by analyzing the natural motion of particles within a material.

## Features

- Calculation of Power Spectral Density (PSD) from trajectory data
- Noise peak detection and filtering in PSD data
- Fitting of various viscoelastic models to PSD data
- Calculation of Mean Back Relaxation (MBR) from trajectory data
- Simulation of trajectories based on PSD or viscoelastic models

## Getting Started

pip install git+https://github.com/wusche1/SpectralMechanicsAnalysis.git 
from SpectralMechanicsAnalysis import ...
