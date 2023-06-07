---
bibliography: "bibliography.bib"
link-citations: true
urlcolor: "blue"
---

![](20230607_Title_Page_Internet.svg)

![](20230607_Eidesstattliche_Erklaerung_Internet.svg)

## Goal of this project

The goal of this project is to generate colorful images using a beta-variational auto-encoder
trained on a dataset of images of flowers.

A secondary goal is to describe the nature of such models, within the context of generative
models and, more specifically, auto-encoders.

## Choice of model

The model of choice is the $β$-variational auto-encoder ($β$-VAE). @higgins2017betavae

Note that a $β$-VAE is just a slightly modified variant of a variational auto-encoder (VAE).
In this document, an attempt to highlight the difference it made in later sections.

The original paper which introduced VAEs was first published in 2013 and has recently
been updated. @kingma2022autoencoding
An comprehensive introduction to variational auto-encoders has been published
by the original authors. @DBLP:journals/corr/abs-1906-02691

While usual auto-encoders compress high-dimensional data during encoding
and deterministically reconstruct instances from compressed instances during decoding,
where the reconstructed instance are as similar as possible to the original uncompressed instances,
variational auto-encoders (VAEs) aim to generate different, or even completely new, instances,
that still have some similarity with any of the the original instances.

Therefore, via a VAE the goal of this project can be reached.

Note that the set of purposes, or use cases, of VAEs intersects with the sets of purposes,
or use cases, of generative adversarial models (GANs) and diffusion models.
These three kinds of models can be used to generate new instances that do not exist in reality,
which are however still somewhat similar to instances, or to combinations of instances,
that actually exist in reality, the original instances.

This very purpose is the main reason why the choice of the type of artificial neural networks to be
implemented in this project has fallen on VAE. A secondary reason for this choice is the interesting
fact that VAEs are essentially a notable probabilistic approach to reach the goal.
Furthermore, at least one other student in the course already worked with GANs,
so working with a VAE is, form this perspective, something until now unexplored
as far as this course is concerned.
Finally, diffusion models have been briefly considered, but since they are quite recent approaches,
still subject to a lot of research, and since their complexity is significantly higher
than the complexity of VAEs, the choice in this project is indeed VAEs.

## This document

This document is written in Markdown and transformed to PDF using the approach described by @scimd.

## Disclaimer regarding this document

This document documents the work that has been performed during this project.
The work has been performed and documented by the author in the context of an educational course.
This document is not to be regarded as a research document, blog article, nor recitation.
It details aspects of the kind of artificial neural network applied to reach the project goal.
To some extend, it also compares this kind of model to a few other kinds of models in the large
and growing category of generative models.

## The code

The software is implemented using the IDE. @vscode The code is hosted publicly on GitHub. @github
It comes in the form of a Jupyter notebook
that can most conveniently be ran on Google Colaboratory. @colab

The link to the code hosted on GitHub is:

https://github.com/aisven/generative-model-experiment-1

## Disclaimer regarding the code

The code has been written in a relatively verbose free style for studying purposes.
It is not production-ready and it is not to be regarded as research-related or blog-article code.
Inline comments have been written to verbosely describe aspects that came up during coding.
Assertions have been added to confirm certain conditions along the way.

The author is aware that production-ready code on the other hand would usually, for example,
follow principles of clean code, with automated formatting, linting, spell-checking, etc.,
contain python type-hints in a much more consistent manner,
be automatically tested via test cases written with pytest or similar libraries,
be modularized and packaged properly, with selected functions moved to 1st-party libraries,
allow for proper packaging, export, import, etc. of readily trained models,
be compatible with open source or proprietary machine learning pipelines (think MLOps, DevOps),
be integrated with respective tooling to visualize metrics about model performance,
ship together with benchmarks, examples, cook book or user guide, etc.,
be optimized to some extend.

## The dataset

The dataset of flowers used in this project is the "102 Category Flower Dataset" curated and
published by Maria-Elena Nilsback and Andrew Zisserman in the research area of computer vision
around the year 2009. @flwrds This dataset conveniently in provided by PyTorch as one of the
integrated datasets. @flwrdspt In the Jupyter notebook, there is a code block which creates
and instance of this dataset and triggers the automated download, just as suggested by PyTorch.
In this way, the dataset does not need to be downloaded in a custom way.

## The pre-processing pipeline

A pre-processing pipeline is defined in a code block in the Jupyter notebook. It mainly serves
in reducing the images in size, thereby in particular getting rid of most of the areas where
gras or earth or other plants could be present in the background. Basically, we generally are
interested in the mid section of the images, where typically one or more blossoms are located.

However, the pre-processing also includes an attempt to smoothen or blur the images to some extend,
in an improvised way. This topic is not the main focus of this project, but it deserves a paragraph.

After all, at the time the dataset was curated by searching the internet for photos of flowers,
the images did not have a high quality as it would be considered high quality today.
The smoothing helps a bit in reducing artifacts and noise in the images.
It also makes certain features of blossoms less important, again, mainly by blurring and
reduction in size. For example, the importance and defined appearance of fine lines in the leaves
of the blossoms is reduced.

This is by no means a perfect pre-processing pipeline. Experts in image processing, or for example
photographers or graphics artists, might even find it to be naive. Yet, in order to reach the
project goal, the pre-processing helps channeling the data towards images of color, where distinct
features of flowers nor unwanted artifacts should not play a role.

## The results

Selected generated colorful images can be found in the results directory in the project,
hosted on GitHub:

https://github.com/aisven/generative-model-experiment-1/results/images

## Categorization of the Variational Auto-encoder (VAE)

This section associates the VAE with different categories of models into which it falls.
Also, notable related kinds of models are mentioned.
Finally, the difference between a VAE and a $β$-VAE is briefly explained.

This section is not a serious literature review.
Its purpose is to give the reader an idea
where variational auto-encoders fit in with the task of image generation as of today.

### Category Auto-encoder

At its roots, any VAE is an auto-encoder. However, it is not an auto-encoder optimized for
reconstruction of known instances, but an auto-encoder modified
to obtain generation of new instances.

Since it is an auto-encoder, a VAE has some layers forming its encoder and some other layers forming
its decoder.

It can be split into two separate, yet associated parts, the encoder and the decoder, if desired,
so as to store the encoded data coming out of the encoder, to then later, whenever, pass it to the
decoder to obtain generated data. In case of a VAE this use case is usually of lesser interest,
unless perhaps the encoded data is to be stored and analyzed to some extend. Of primary interest
is the generated data coming out of the decoder.

Depending on the data, different kinds of layers can be added. For example, when images are
concerned, convolutional layers can be added to the front of the encoder to automatically highlight
features in the images while de-emphasizing less relevant aspects of the image, thereby also
reducing the dimensionality of the data. If convolutional layers are added, respective so-called
de-convolutional layers need to be added to the end of the decoder.

### Category Unsupervised Learning

Just like auto-encoders, variational auto-encoders fall into the category of unsupervised learning.
Only the input data $X$ is required. No labels are required, and accordingly,
there is no target variable $y$.

### Category Latent Variable Model

A VAE is a latent variable model, since it is essentially an attempt to learn about latent variables
and their associated probability distributions of a latent space. The assumption thereby is, that
the input data, which is given, and which in other words, can be observed, is governed, or in other
words represented or described best, in some unknown unobservable latent space.

### Category Generative Model

A VAE is a generative model, since after all, it models a joint probability distribution of the
observed data together with, in other words joined by, the generated data. Thereby, in fact, the
VAE allows for and tries to introduce variance on purpose, in a somewhat constrained manner,
in order to be able to generate new instances that deviate from observed instances. Deviate,
that is, much more than it would be the case with a classical auto-encoder for example.

The category of generative models stems from statistics. It is certainly related to,
but needs to be distinguished from, the category described in the next sub-section.

### Category Generative Artificial Intelligence

This is a broad category representing generative AI. This category typically includes the category
described in the next sub-section, namely deep generative models, yet it needs to be distinguished
from the above mentioned category of generative models.

A VAE falls also into the category of generative AI, being remembered with its invention and
respective advances, roughly in parallel with GANs, especially in the years from roughly
2014 to 2017.

### Category Deep Generative Model

A VAE is a deep generative model, since it is to a significant extend based on,
or in other words constructed as, an artificial deep neural network.

### Notable related kinds of models

The following kinds of models need to be mentioned as well when talking about VAEs, since these
other kinds of models stand for a significant portion of the major advances in generative AI
between roughly the years 2014 and 2023.

These kinds of models are only mentioned here very briefly and without any details, just to put
the feasibility of image generation via VAEs into perspective.

In so-called generative adversarial models (GANs), the generator part is essentially a latent
variable model. A GAN is also a deep generative model. @goodfellow2014generative

The concept of generative pre-trained transformers (GPTs) has been under development for
quite a while, and sparked a lot of hype in the AI community in the year 2019.
Notably, the later gone-viral version of ChatGPT by OpenAI
is based on a very large GPT model. @brown2020language

Currently en vogue through manifestation in the concrete gone-viral model called DALL-E 2
are a combination of GPTs, things like contrastive language-image pre-training (CLIP),
and diffusion models. Diffusion models are considered deep generative networks,
latent variable models and of course, generative AI. @ramesh2022hierarchical

From a use case perspective, combinations of GPTs, contrastive language-image pre-training (CLIP),
and diffusion models, facilitate text-to-image transformation, i.e. an image is generated according
to a description entered into a prompt in the form of text. In other words, the system is prompted,
or asked, to generate an image with some content as described, and then, a bit later in that
inference pipeline, the actual image is generated by the diffusion model.

While VAEs are still interesting, especially from a probabilistic viewpoint, with many VAE variants
having been developed and with research still ongoing, it needs to be acknowledged that VAEs do
not represent the state-of-the-art in creative image generation at all.

### Now where does the $β$-VAE fit in?

Basically, the $β$-VAE is just a slightly modified VAE. It introduces a weighting of the
regularization term in the loss function of the VAE. The weighting is a single decimal parameter
called $β$. It is a hyperparameter. This modification allows the $β$-VAE to be trained with more
weight put on the regularization term, which has been shown can lead to more disentangled, as
opposed to correlated, features in the latent space. Properly exploited, this can allow the
$β$-VAE to learn about the latent space that let's the features turn out to be meaningful. A common
example of this is that it has been shown that in generated faces, the fact whether or not the
person is smiling, can turn out to be represented by one of the dimensions, i.e. features, in the
latent space.

In this project, this aspect of disentanglement of features in the latent space is not actually
exploited nor examined. It is however implemented, with the hyperparameter $β$ simply being set to
some value taken from a range like $2.0$ to $8.0$,
as encountered in the original $β$-VAE paper. @higgins2017betavae

## Architecture of the Variational Auto-encoder

### The primary goal of a VAE

The main original goal of the VAE is to generate new instances that are new yet still similar to,
yet not just reconstructions of observed instances.

### Learning a representation of the latent space

A VAE learns a representation of the latent space. The latent variables in the latent space are
sometimes referred to as features in the latent space. Usually, there is one such latent variable
per dimension of the latent space. In a typical VAE, each latent variable is statistically modeled
by a probability distribution.

In the following, we denote the number of dimensions in the latent space by $l$.
We use $i$ to denote the $i$-th such dimension.

### The encoder

The encoder maps input data to data in the latent space.

First, it usually, yet not necessarily, performs a typical form of dimensionality reduction of the
input data, depending on the type of data and the dimensionality of the input space. In case of
images this is usually achieved by having two convolutional layers.

Then, the data is passed as input to two parallel layers. These two layers are to represent the
means and the standard deviations of the probability distributions of the latent variables,
aka. latent features, in the latent space.

In the most usual case, all learned probability distributions of the latent space are
individual normal distributions (aka. Gaussian distributions).

$$\mathcal{N_i}(μ_i,σ_i)$$

The two layers, or more specifically their vectors of weight values, are aptly called $𝛍$ and $𝛔$.
During training, the values in these two important layers are learned just as weights are usually
learned in linear layers.

### Sampling and the re-parametrization trick

In order to introduce variation, data is sampled from the learned probability distributions.
Technically, if data was sampled using common implementations of say the normal distribution,
back-propagation would not be possible. Instead of sampling directly from the normal
distributions, the two layers $μ$ and $σ$ are regularly connected to the first layer of the
decoder, so that normal training of a neural network can be applied, using back-propagation and
a typical optimizer such as ADAM.

To introduce variation during the forward pass and also during inference, values are sampled from
the standard normal distribution:

$$\mathcal{N}(1,1)$$

In fact one such value is sampled per latent feature.
These values are roughly somewhere in the range $-2.5$ to $2.5$, normally distributed aroung $0.0$.
Somewhat rarely, they are smaller or larger, as per the nature of the standard normal distribution.
These sampled values form the vector $𝛆$. Conceptually, the vector $𝛆$ can be thought of as
a standard normal random vector.

Next, a linear combination of the vectors $𝛍$, $𝛔$, $𝛆$ is computed as follows,
which yields the vector $𝐳$:

$$𝐳 = 𝛍 + 𝛔 * 𝛆$$

Notably, the vector $𝐳$ is a vector in the learned latent space.

Conceptually, the vector $𝐳$ can be thought of as a normal random vector, since it is also a linear
transformation of the standard normal vector $𝛆$ via the vectors $𝛔$ and $𝛍$.

Each value $z_i$ deviates from the mean $μ_i$, sometimes more, sometimes less,
depending on the value of $ε_i$ and the value of $σ_i$,
always according to the learned probability distribution of
the $i$-th dimension of the represented latent space.

In other words, for each sampling operation, the standard deviation is scaled by the random factor
epsilon. This way of sampling in a variational auto-encoder, which enables the use of sampling while
retaining the possibility of back-propagation, is known in the artificial intelligence
community as the re-parametrization trick. From a statistics viewpoint, this is not really a
trick, but rather an alternative way to implement sampling from normal distributions.

### The decoder

The decoder maps the data vector z to the output space.

For decoding images that have been reduced by convolutional layers in the encoder, the decoder
would contain respective de-convolutional layers.

In case the values in the different channels of an image are supposed to be in range $0.0$ to $1.0$,
as it may be required by a typical image format, the decoder can make use of the regular
sigmoid function to bring the data values into that particular range.

The output of the decoder is a generated image.

## Training the Variational Auto-encoder

The training of a VAE is almost equal to the training of an auto-encoder. There is a notable
difference in the error function. Since a VAE is essentially a probabilistic model, the notion
of distance between two probability distributions can play a role.

### Priors lead the way

In the case of a VAE, typically, a prior probability distribution is formulated for each dimension
in the latent space, with the intend to let each learned probability distribution roughly converge
to the shape of the respective prior. In the most simple variant, all such priors are simply taken
to be normal distributions with variance $1$:

$$\mathcal{N}(m_i,1)$$

### The loss function

The loss function of a VAE consists of two terms,
a reconstruction term
and a regularization term.
In case of a $β$-VAE, the regularization term $R_2$ is multiplied by the hyperparameter $β$.

$$loss = reconstruction + β * regularization$$

### The reconstruction term

The reconstruction term of a VAE can be the very same as it would be in a normal auto-encoder.
It is used to measure the difference between the given input instance and a generated output
instance. In case of images, often times, the loss is subject to a reduction in terms of a sum,
as opposed to an average, which is why the actual numerical values of the loss can be larger than
in other optimization scenarios.

In the case of the VAE implemented in this project, the channels of the images indeed contain
values in range $0.0$ to $0.1$. This fact it exploited in the following way. Instead of using a
more complicated formula for the reconstruction term, the binary cross entropy loss function is
used.

One aspect for this choice is that the binary cross entropy is computationally efficient.
Another aspect is that the optimization objective to minimize the loss can perfectly be
represented by minimizing the output of the binary cross entropy function in this case.

Note that in case of regular auto-encoders, the reconstruction term is already enough, as the goal
is to generate output instances that are as similar as possible to the input instances. In other
words, regular auto-encoders try to replicate the original as close as possible.

### The regularization term

A VAE needs a regularization term in addition to the reconstruction term to make it learn,
or in other words, to steer training to shape, the probability distributions of the latent space
according to the given prior distributions.

Different measures for the distance between probability distributions exist. In case of a VAE,
an effective and well-known distance measure can be applied, namely the Kullback–Leibler divergence.
In a VAE, where the prior distributions are all based on relatively simple normal distributions
with variance $1$, a much simplified variant of the KL-divergence can be implemented:

$$ k = \sum_{i=1}^{l} σ_i^2 + μ_i^2 + \log σ_i - 1 $$

Thus, the distance between the learned probability distributions in the latent space and the
prior probability distributions becomes part of the loss and is therefore minimized, to some extend,
during training.

### Wrapping up

The regularization with the concept of having the probability distributions aligned to priors,
has two distinct goals.

First, if any two instances are close in the latent space, they should also
be close in the output space. This goal is called continuity.

Second, instances in the output space should be reasonably well defined, as opposed to being
garbage. This goal is referred to as completeness.

Together with the third goal, to obtain disentangled meaningful features in the latent space,
as aimed for by the introduction of the hyperparameter $β$ to emphasize the regularization,
$β$-variational auto-encoders can be excellent models for sensible dimensionality reduction.
In statistics and data analytics, they can serve as an alternative to more traditional methods
such as the principal component analysis (PCA),
in particular when dealing with high-dimensional data.

# References
