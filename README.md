# PyTorch-AdaHAN
An unofficial PyTorch implementation of the HAN and AdaHAN models presented in the ["Learning Visual Question Answering by Bootstrapping Hard Attention"](https://arxiv.org/pdf/1808.00300.pdf) research paper, by Mateusz Malinowski, Carl Doersch, Adam Santoro, and Peter Battaglia of DeepMind, London.

## Paper Overview

**Soft Attention vs. Hard Attention**  

Visual attention is the selective weighting of importance given to spatial locations in an input image. Attention can be seperated into two main categories: *soft attention* and *hard attention*. With soft attention, models learn to re-weight the importance of 
various regions of images, but regions are never completely filtered out. This is an important distinction between the two categories: hard attention learns to completely ignore entire selected regions of input images.

Soft Attention:

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/soft_attention.PNG)

*image credit: [Show, Attend and Tell: Neural Image CaptionGeneration with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)*


Hard Attention:

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/hard_attention.PNG)

*image credit: ["Learning Visual Question Answering by Bootstrapping Hard Attention"](https://arxiv.org/pdf/1808.00300.pdf)*


The key advantage of visual attention is that it allows models to perform computationally expensive operations (such as visual question answering) on a selected subset of regions instead of on prohibitively large quantities of sensory input. Hard attention, because it discards entire regions of inputs, has the potential to be even more computationally efficient than soft attention. However, the major drawback of hard attention (and the main reason why soft attention has been much more widely used and successful) is that it was generally viewed as non-differentiable. To explain more simply, with a deep neural network, models can minimize loss values by nudging the parameters responsible for attention weights assigned to spatial regions. With hard attention, this isn't possible, because later processing steps are performed only on a subset of regions, which zeros out the flow of gradient information to weights tied to regions which were ignored.

**The Hard Attention Network (HAN) and Adaptive Hard Attention Network (AdaHAN)**

The main contribution of this paper is a hard attention mechanism that is differentiable and can achieve comparable or even slightly better performance than soft attention models on complex visual question and answering (VQA) tasks. These models bootstrap on one of the interesting breakthrough from [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/), which is the finding that feature L2-activations (the magnitude of the dot-product of activations across channels) at spatial locations in deeper layers are correlated with how strongly semantically meaningful features are detected at those locations. I don't feel qualified to discuss interpretability further, but the general idea is that spatial locations that are semantically meaningful have large L2-norm values, which are differentiable.

HAN, AdaHAN Model Diagram:

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/han_model.PNG)

*image credit: ["Learning Visual Question Answering by Bootstrapping Hard Attention"](https://arxiv.org/pdf/1808.00300.pdf)*

The HAN and AdaHAN models utilize this finding by using the L2-norm as the region-selection mechanism. For the VQA task, this is done by encoding the input question to a dimension identical to the number of channels of the encoded input image. Then, the encoded question is broadcasted and summed with each feature vector of the encoded image. At this point, L2-activations are calculated at each spatial location. HAN requires an additional hyperparameter, *k*, which is how many of embedded positions to attend to (those unselected are zeroed out). AdaHAN instead utilizes the softmax function over the L2-activations (in a sense, making them compete against each other) and selecting the *adaptive* number of locations that surpass a certain threshold value. This allows for the model to vary the regions to attend to on a case by case basis.



## Experimentation Notes
I wrote a program to synthetically generate very simple images of a barn scene, and questions and answers to pair with each image. Below are examples of the hard-attention mechanism applied to validation images over the course of my training sessions:

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/attn_img_0.png)

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/attn_img_1.png)

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/attn_img_3.png)

My laptop's performance leaves a lot to be desired, so I generated small datasets, usually between 500-1000 training samples and 100-200 validation samples. That's probably why the validation performance usually dropped pretty quickly, when my model started overfitting to the training data. However, it still usually bottomed out at the 30 percent accuracy mark, just a little better than randomly selecting one of the four possible classes for each question and image.

Training results:

![](https://github.com/gnouhp/PyTorch-AdaHAN/blob/master/repo_images/accuracy_plot.png)


In hindsight, my synthetic datasets were way too simplistic and there isn't enough variance to prevent the models from learning the locations of the image that relevant more often than not across the entire dataset. I would love to run various experiments on much more complex datasets in the future, if I can get access to more computing resources.

**My thoughts and questions on the topic:**

1. Model performance across independent training sessions was very variable.

2. The correspondence between the input image and its latent features is highly complicated and may be very important for making future breakthroughs with hard attention mechanisms. Calculating the receptive field of latent features is pretty tricky and I believe new research is still being done on topic as well. I would be very interested in research on the overlap in the input space that inevitably occurs when sampling from the embeddded space.

3. During training, I observed that the percentage of pixels attended to in the input image almost always came with an increase in performance, which is not very surprising. What would be good methods to punish the network from selecting too many overlapping and unnecessary regions to attend to?

4. I am not 100 percent convinced that the AdaHAN softmax on the presence vector is sufficient to lower the number of attended regions, or that the weights tied to embedded locations are "competing" against each other in some sort of zero-sum game to get the highest L2-norms. How can we be sure that the softmax operation, which is conditioned on the loss from the answer, doesn't learn to select embedded positions that correspond to a maximal area of the input image?


**Citations**
["Learning Visual Question Answering by Bootstrapping Hard Attention"](https://arxiv.org/pdf/1808.00300.pdf), by Mateusz Malinowski, Carl Doersch, Adam Santoro, and Peter Battaglia of DeepMind, London.
[The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/), by Olah, et al., "The Building Blocks of Interpretability", Distill, 2018.
