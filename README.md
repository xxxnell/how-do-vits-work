

# How Do Vision Transformers Work?

This repository provides a PyTorch implementation of ["How Do Vision Transformers Work? (ICLR 2022 Spotlight)"](https://openreview.net/forum?id=D78Go4hVcxO) In the paper, we show that the success of multi-head self-attentions (MSAs) for computer vision is ***NOT due to its weak inductive bias and capturing long-range dependency***. 
In particular, we address the following three key questions of MSAs and Vision Transformers (ViTs): 

***Q1. What properties of MSAs do we need to better optimize NNs?***  

A1. MSAs have their pros and cons. MSAs improve NNs by flattening the loss landscapes. A key feature is their data specificity, not long-range dependency. On the other hand, ViTs suffers from non-convex losses.


***Q2. Do MSAs act like Convs?***  

A2. MSAs and Convs exhibit opposite behaviors—e.g., MSAs are low-pass filters, but Convs are high-pass filters. It suggests that MSAs are shape-biased, whereas Convs are texture-biased. Therefore, MSAs and Convs are complementary.


***Q3. How can we harmonize MSAs with Convs?***  

A3. MSAs at the end of a stage (not a model) significantly improve the accuracy. Based on this, we introduce *AlterNet* by replacing Convs at the end of a stage with MSAs. AlterNet outperforms CNNs not only in large data regimes but also in small data regimes.


👇 Let's find the detailed answers below!


### I. What Properties of MSAs Do We Need to Improve Optimization?

<p align="center">
<img src="resources/vit/loss-landscape.png" style="width:90%;">
</p>

MSAs improve not only accuracy but also generalization by flattening the loss landscapes. ***Such improvement is primarily attributable to their data specificity, NOT long-range dependency*** 😱 On the other hand, ViTs suffers from non-convex losses. Their weak inductive bias and long-range dependency produce negative Hessian eigenvalues in small data regimes, and these non-convex points disrupt NN training. Large datasets and loss landscape smoothing methods alleviate this problem.


### II. Do MSAs Act Like Convs?

<p align="center">
<img src="resources/vit/fourier.png" style="width:90%;">
</p>

MSAs and Convs exhibit opposite behaviors. Therefore, MSAs and Convs are complementary. For example, MSAs are low-pass filters, but Convs are high-pass filters. Likewise, Convs are vulnerable to high-frequency noise but that MSAs are vulnerable to low-frequency noise: it suggests that MSAs are shape-biased, whereas Convs are texture-biased. In addition, Convs transform feature maps and MSAs aggregate transformed feature map predictions. Thus, it is effective to place MSAs after Convs.


### III. How Can We Harmonize MSAs With Convs?

<p align="center">
<img src="resources/vit/architecture.png" style="width:90%;">
</p>

Multi-stage neural networks behave like a series connection of small individual models. In addition, MSAs at the end of a stage play a key role in prediction. Based on these insights, we propose design rules to harmonize MSAs with Convs. NN stages using this design pattern consists of a number of CNN blocks and one (or a few) MSA block. The design pattern naturally derives the structure of the canonical Transformer, which has one MLP block for one MSA block.

<br />

<p align="center">
<img src="resources/vit/alternet.png" style="width:90%;">
</p>

Based on these design rules, we introduce AlterNet by replacing Conv blocks at the end of a stage with MSA blocks. ***Surprisingly, AlterNet outperforms CNNs not only in large data regimes but also in small data regimes.*** This contrasts with canonical ViTs, models that perform poorly on small amounts of data. For more details, see below (["How to Apply MSA to Your Own Model"](#how-to-apply-msa-to-your-own-model) section).



This repository is based on [the official implementation of "Blurs Behaves Like Ensembles: Spatial Smoothings to Improve Accuracy, Uncertainty, and Robustness"](https://github.com/xxxnell/spatial-smoothing).  In this paper, we show that a simple (non-trainable) 2 ✕ 2 box blur filter improves accuracy, uncertainty, and robustness simultaneously by ensembling spatially nearby feature maps of CNNs. MSA is not simply generalized Conv, but rather a generalized (trainable) blur filter that complements Conv. Please check it out!




## Getting Started 

The following packages are required:

* pytorch
* matplotlib
* notebook
* ipywidgets
* timm
* einops
* tensorboard
* seaborn (optional)

We mainly use docker images `pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime` for the code. 

See ```classification.ipynb``` for image classification. Run all cells to train and test models on CIFAR-10, CIFAR-100, and ImageNet. 

**Metrics.** We provide several metrics for measuring accuracy and uncertainty: Acuracy (Acc, ↑) and Acc for 90% certain results (Acc-90, ↑), negative log-likelihood (NLL, ↓), Expected Calibration Error (ECE, ↓), Intersection-over-Union (IoU, ↑) and IoU for certain results (IoU-90, ↑), Unconfidence (Unc-90, ↑), and Frequency for certain results (Freq-90, ↑). We also define a method to plot a reliability diagram for visualization.

**Models.** We provide AlexNet, VGG, pre-activation VGG, ResNet, pre-activation ResNet, ResNeXt, WideResNet, ViT, PiT, Swin, MLP-Mixer, and Alter-ResNet by default. timm implementations also can be used.




## Visualizing the Loss Landscapes

Refer to ```losslandscape.ipynb``` for exploring the loss landscapes. It requires a trained model. Run all cells to get predictive performance of the model for weight space grid. We provide [a sample loss landscape result](resources/results/cifar100_resnet_mcdo_18_x10_losslandscape.csv).


## Evaluating Robustness on Corrupted Datasets

Refer to ```robustness.ipynb``` for evaluation corruption robustness on [corrupted datasets](https://github.com/hendrycks/robustness) such as CIFAR-10-C and CIFAR-100-C. It requires a trained model. Run all cells to get predictive performance of the model on datasets which consist of data corrupted by 15 different types with 5 levels of intensity each. We provide [a sample robustness result](resources/results/imagenet_alexnet_dnn_corrupted.csv).


## How to Apply MSA to Your Own Model

<p align="center">
<img src="resources/vit/buildup_v.gif" style="width:90%;">
</p>

We find that MSA complements Conv (not replaces Conv), and *MSA closer to the end of a stage* improves predictive performance significantly. Based on these insights, we propose the following build-up rules:

1. Alternately replace Conv blocks with MSA blocks from the end of a baseline CNN model. 
2. If the added MSA block does not improve predictive performance, replace a Conv block located at the end of an earlier stage with an MSA 
3. Use more heads and higher hidden dimensions for MSA blocks in late stages.

In the animation above, we replace Convs of ResNet with MSAs one by one according to the build-up rules. Note that several MSAs in `c3` harm the accuracy, but the MSA at the end of `c2` improves it. As a result, surprisingly, the model with MSAs following the appropriate build-up rule outperforms CNNs even in the small data regime, e.g., CIFAR!




## Caution: Investigate Loss Landscapes and Hessians With l2 Regularization on Augmented Datasets

Two common mistakes ⚠️ are investigating loss landscapes and Hessians (1) *'without considering l2 regularization'* on (2) *'clean datasets'*. However, note that NNs are optimized with l2 regularization on augmented datasets. Therefore, it is appropriate to visualize *'NLL + l2'* on *'augmented datasets'*. Measuring criteria without l2 on clean datasets would give incorrect (even opposite) results.



## Citation

If you find this useful, please consider citing 📑 the paper and starring 🌟 this repository. Please do not hesitate to contact Namuk Park (email: namuk.park at gmail dot com, twitter: [xxxnell](https://twitter.com/xxxnell)) with any comments or feedback.

```
@inproceedings{park2022how,
  title={How Do Vision Transformers Work?},
  author={Namuk Park and Songkuk Kim},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```


## License

All code is available to you under Apache License 2.0. CNN models build off the torchvision models which are BSD licensed. ViTs build off the [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and [Vision Transformer - Pytorch](https://github.com/lucidrains/vit-pytorch) which are Apache 2.0 and MIT licensed.

Copyright the maintainers.



