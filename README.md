# GANs and VAEs | Learning 
Part of a series of respositories dedicated to the __personal learning__ of different _neural network

Structure of the code:
- data: data folder container
- models: models folder container
- notes: personal notes container
- results: results from the neural network


## VAE | Variational Autoencoder
The initial tutorial followed is the tensorflow alpha 2.0 (tutorial)[https://www.tensorflow.org/alpha/tutorials/generative/cvae] Convolutional Variational Autoencoder, base on the MNIST dataset. 

## Architecture
In our VAE example, we use two small ConvNets for the generative and inference network. Since these neural nets are small, we use `tf.keras.Sequential` to simplify our code. Let `x` and `z` denote the observation and latent variable respectively in the following descriptions.

### Generative Network
This defines the generative model which takes a latent encoding as input, and outputs the parameters for a conditional distribution of the observation, i.e. `p(x|z)`. Additionally, we use a unit Gaussian prior `p(z)` for the latent variable.

### Inference Network
This defines an approximate posterior distribution `q(z|x)`, which takes as input an observation and outputs a set of parameters for the conditional distribution of the latent representation. In this example, we simply model this distribution as a diagonal Gaussian. In this case, the inference network outputs the mean and log-variance parameters of a factorized Gaussian (log-variance instead of the variance directly is for numerical stability).

### Reparameterization Trick
During optimization, we can sample from `q(z|x)` by first sampling from a unit Gaussian, and then multiplying by the standard deviation and adding the mean. This ensures the gradients could pass through the sample to the inference network parameters.


### Iterations
1. MNIST dataset 