# Step Distillation Comparison

This document provides detailed quality comparisons between the original 480p I2V model and the step-distilled model.

## Overview

The step-distilled model reduces inference steps from 50 to 8 (or 12 steps recommended) while maintaining comparable visual quality to the original model. On RTX 4090, this achieves up to 75% reduction in end-to-end generation time, enabling a single RTX 4090 to generate videos within 75 seconds. This document showcases side-by-side comparisons to demonstrate that the distillation process does not significantly degrade output quality. For even faster generation, you can also try 4 steps, which provides faster speed with slightly reduced quality.

## Comparison Results

The following table shows side-by-side comparisons between the original 480p I2V model (50 steps) and the step-distilled model (8 steps). The comparisons demonstrate that the step-distilled model maintains comparable visual quality while achieving significant speedup.

| Original Model (50 steps) | Step-Distilled Model (8 steps) |
|---------------------------|--------------------------------|
| <video src="https://github.com/user-attachments/assets/eb019065-880a-4979-a0d3-091efd7ad34c" width="400"> | <video src="https://github.com/user-attachments/assets/09a87faa-4c19-4fa5-899b-1ec8e2fe811a" width="400"> |




## Usage Notes

- **8 or 12 steps**: Recommended default setting, provides the best balance between speed and quality
- **4 steps**: Faster generation with slightly reduced quality, suitable for rapid prototyping

Detailed usage instructions can be found in [Usage](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/README.md#-usage).

