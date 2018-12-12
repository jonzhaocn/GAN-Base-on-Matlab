# GAN-Base-on-Matlab

## data
### mnist_uint8
[download link](https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/data/mnist_uint8.mat)

## Example
### example_1
* network structure:
```
generator.layers = {
    struct('type', 'input', 'output_shape', [100, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [3136, batch_size], 'activation', 'leaky_relu')
    struct('type', 'reshape', 'output_shape', [7,7,64, batch_size])
    struct('type', 'conv2d_transpose', 'output_shape', [14, 14, 32, batch_size], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'conv2d_transpose', 'output_shape', [28, 28, 1, batch_size], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'sigmoid')
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [28, 28, 1, batch_size])
    struct('type', 'conv2d', 'output_maps', 32, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'sub_sampling', 'scale', 2)
    struct('type', 'conv2d', 'output_maps', 64, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'sub_sampling', 'scale', 2)
    struct('type', 'reshape', 'output_shape', [3136, batch_size])
    struct('type', 'fully_connect', 'output_shape', [1, batch_size], 'activation', 'sigmoid')
};
```
* result:

<p align="center">
    <img src="https://github.com/JZhaoCH/GAN-Base-on-Matlab/blob/master/readme_images/1.png">
</p>

<p align="center">
    <img src="https://github.com/JZhaoCH/GAN-Base-on-Matlab/blob/master/readme_images/2.png">
</p>

### example_2
* network structure:
```
generator.layers = {
    struct('type', 'input', 'output_shape', [100, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [1024, batch_size], 'activation', 'relu')
    struct('type', 'fully_connect', 'output_shape', [28*28, batch_size], 'activation', 'sigmoid') 
    struct('type', 'reshape', 'output_shape', [28, 28, 1, batch_size])
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [28,28,1, batch_size])
    struct('type', 'reshape', 'output_shape', [28*28, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [1024, batch_size], 'activation', 'relu')
    struct('type', 'fully_connect', 'output_shape', [1, batch_size], 'activation', 'sigmoid') 
};
```
* result:

<p align="center">
    <img src="https://github.com/JZhaoCH/GAN-Base-on-Matlab/blob/master/readme_images/3.png">
</p>

## Reference
1. `https://grzegorzgwardys.wordpress.com/2016/04/22/8/`
2. `Dumoulin V, Visin F. A guide to convolution arithmetic for deep learning[J]. 2016.`
3. `https://github.com/rasmusbergpalm/DeepLearnToolbox/tree/master/CNN`
4. `http://neuralnetworksanddeeplearning.com/index.html`
