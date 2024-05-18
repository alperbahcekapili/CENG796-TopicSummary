
# Diffusion Mechanism


<figure>
<img src="intro_image_1.png" alt="drawing" width="30%"/>
<img src="intro_image_2.png" alt="drawing" width="30%"/>
<img src="intro_image_3.png" alt="drawing" width="30%"/>
<figcaption style="text-align: center">Figure 1: Generation Examples</figcaption>
</figure>


**What is diffusion ?**

Diffusion is a mechanism where the ultimate goal is to generate the beautiful images you see above. Of course the mechanism is not only used to generate images, with proper formulation diffusion models can approximate any type of data distribution.

Diffusion has two main processes:

* **Forward diffusion:** Gradually adds niose to the input
* **Reverse denoising:** Learns to generate the data with denoising
<figure>
<img src="denoising_example.png" alt="https://developer.nvidia.com/blog/improving-diffusion-models-as-an-alternative-to-gans-part-1/" width="100%" />
<figcaption style="text-align: center">Figure 2: Diffusion Process Overview</figcaption>
</figure>


<br><br><br>


**Theoretical Backgorund**

You can view diffusion as following. The main goal is to convert complex distribution into a simpler target distribution by means of transition kernel T.


x<sub>0</sub> ~ p<sub>complex</sub> ==> T(x<sub>0</sub>) ~ p<sub>prior</sub>

This kernels are modeled as repeated actions in diffusion. As you can see from the Figure 2, at each timestep *t<sub>i</sub>* model iteratively denoises the input. Thus at each timestep approaching to the target input distribution. How can we show that mathematically ?

$p_{prior}(x) = \int q(x|x') p_ {prior}(x')dx'$

If transition kernel q has the above property, then repeatedly applying this kernel leads samples towards $p_{prior}$.

<figure>
<img src="diffusion_over_time.png" alt="https://ayandas.me/blogs/2021-12-04-diffusion-prob-models.html">
<figcaption style="text-align: center">Figure 3: Diffusion Over Time</figcaption>
</figure>


However we are able to do this operation in discrete timesteps.

$x_t \sim  q(x|x'= x_{t-1} ), \forall t>0$

*t* is finite and typically sufficent in practive.

Because transition kernel is repeatedly applied we can see overall process as Markov chain.

$q(X_t|X_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) $

$q(X_T) = p_{prior}(X_T) = N(x_t; 0, I) $

To be able to generate the data, we need reverse diffusion process. 

$x_T \sim N(0,1) \to T^{-1}(x_T) \sim p_{data}$

Process $T^{-1}$ learns from the data

<figure>
<img src="fwddiff.gif" alt="https://ayandas.me/blogs/2021-12-04-diffusion-prob-models.html" width="49%">
<img src="revdiff.gif" alt="https://ayandas.me/blogs/2021-12-04-diffusion-prob-models.html" width="49%">
<figcaption style="text-align: center">Figure 4: Forward and Backward Diffusion</figcaption>
</figure>


## Forward Diffusion Process

Forward diffusion process is fixed. Starting from data $x_0$, forward diffusion process adds noise to the data with variance $\beta_t$ 

$q(X_t|X_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I) \to q(x_{1:T}|x_0)=\prod_{t=1}q(X_t|x_{t-1})$ 

Using Gaussian's linearity over *t*, we can directly express $q(x_t|x_0)$ as a shortcut. We do not need to sample iteratively in forward process. Thus we can furhter speed up the training process.

Define $ \bar{\alpha_t} = \prod_{s=1}^t(1-\beta_s) \to q(x_t|x_0) = N(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I))$

Sample: 
$ x_t = \sqrt{\bar{\alpha}_t}x_0+\sqrt{(1-\bar{\alpha}_t)}\epsilon$ where $\epsilon \sim N(0,1)$

Because we can obtain $x_t$ from the $x_0$ the forward process is much faster this way.

<br>

**How are $\beta_t$ values choosen ?**

There are multiple approaches that one can follow. Some of them are as following:

* Linearly schedule $\beta$ values from $\beta_1=10^{-4}$ to $\beta_T=0.02$ [Denoising Diffusion Probabilistic Models
](https://arxiv.org/abs/2006.11239)
* Learn $\beta$ values together with the model [Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." International conference on machine learning. PMLR, 2015.](https://proceedings.mlr.press/v37/sohl-dickstein15.html).


**What happens during forward diffusion process ?**

![Distribution change during forward diffusion ](dist_change.png)

$q(x_T) = \int q(x_0, x_t) = \int q(x_0) q(x_t | x_0) dx_0$

We can sample $x_t \sim q(x_t)$ by first sampling $x_0 \sim q(x_0)$ and then sampling $x_t \sim q(x_t | x_0)$
q  in the $x_t \sim q(x_t | x_0)$ is the transition kernel.

## Backward Diffusion

At this step the main goal is to denoise the $x_t$ so that at each time step we iteratively denoise the $x_t$ a bit. At the end, we will get rid of the noise and reach to original data $x_0$

![alt text](backward_diff.png)

$p(x_T)=N(x_T;0,I)$

$p_\theta(X_{t-1}|x_t)=N(x_{t-1};\mu_\theta(x_t,t),\sigma^2_t I) \to p_\theta(x_{0:T}) = p(X_T) \prod_{t=1}^Tp_{\theta}(x_{t-1}|x_t)$ 

$\mu_\theta \to$ is a trainable network to estimate $p(x_{t-1}|x_t)$ This can be U-net like model or denoising autoencoder etc. 

! One important point to note is that network is shared across all time steps. So takes *t* (time step) as input as well.

! As you recall $ \bar{\alpha_t}$ is defined as follows.: $ \bar{\alpha_t} = \prod_{s=1}^t(1-\beta_s)$. Because $\beta$ values can be predefined and learned troughout the training, $ \bar{\alpha_t}$ can be fixed or can be learned as well.



# Learning how to diffuse

## Tractability of the Reverse Diffusion

Within forward diffusion, we often use Gaussian noise to add noise to the input. However, in reverse diffusion, we need to learn the noise distribution. This is a challenging task because the noise distribution is not known. Let's remember the reverse diffusion process:

1. We start with a sample $x_T$ from the noise distribution $q(x_T) = N(x_T; 0, I)$
2. We apply the reverse diffusion process by applying the reverse transition kernel $q(x_{t-1}|x_t)$ to the sample $x_T$ to get $x_{T-1}$

That ultimately leads to the following formula:

$p_{data}(x_0: x_T) = p_{data}(x_0) \prod_{t=1}^{T} q(x_{t-1}|x_t)$

However, the reverse transition kernel is not recoverable, because it's the true denoising distribution, which is also a function of the real data distribution, and we don't know that. This makes the reverse diffusion process intractable.

To make the reverse diffusion process tractable, we need to approximate the reverse transition kernel. There are 2 main assumptions that make the reverse diffusion tractable:

1. **Assumption 1: Noise is Gaussian:** We assume that the noise distribution is Gaussian, which is a common assumption in diffusion models. This assumption simplifies the reverse diffusion process because we can learn the noise distribution from the data. This is the main assumption that makes the reverse diffusion tractable. 

As we already know the noise distribution is Gaussian, we only need to learn the mean and variance of the noise distribution at each time step. This is done by training a neural network to predict the mean and variance of the noise distribution at each time step, which yields:

$q(x_{t-1}|x_{t:T}) = N(x_{t-1}; \mu_{\theta}(x_{t:T}), \sigma_{\theta}(x_{t:T})I)$

We are doing some redundant calculations here, because we are predicting the mean and variance of the noise distribution at each time step. 

**RECALL**: We were delibaretly adding noise to the input at each time step with scheduled variance. So, we may choose to obtain it from the schedule directly.

$q(x_{t-1}|x_{t:T}) = N(x_{t-1}; \mu_{\theta}(x_{t:T}), \sigma_{t}I)$

**For the sake of generality, we will be using $\sigma_{t}$ notation in the rest of the document.** 

2. **Assumption 2: Noise is Independent:** We assume that the noise at each time step is independent. This assumption simplifies the reverse diffusion process because we can learn the noise distribution at each time step independently. This is also known as the Markovian assumption. 

**It should be known that;** the process was already tractable since the first assumption, but, it's just hard to train all those parameters. Besides of that, as we recall from the forward diffusion process, we were adding noise to each timestep by sampling independently from the previous timesteps. By predicting the noise, instead of the resulting image, we can safely assume the markovian property.

Markovian assumption yields:

$q(x_{t-1}|x_{t}) = N(x_{t-1}; \mu_{\theta}(x_{t}), \sigma_{t}I)$

In the end, we can approximate the data distribution by applying the reverse diffusion process with the approximated reverse transition kernel:

$q(x_{0:T}) = p_{data}(x_0) \prod_{t=1}^{T} q(x_{t-1}|x_t)$



