# GAN-Base-on-Matlab

## Example
### example_1
model:
```
generator.layers = {
    struct('type', 'input', 'output_shape', [batch_size, 100]) 
    struct('type', 'fully_connect', 'output_shape', [batch_size, 3136], 'activation', 'leaky_relu')
    struct('type', 'reshape', 'output_shape', [batch_size, 7,7,64])
    struct('type', 'conv2d_transpose', 'output_shape', [batch_size, 14, 14, 32], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'conv2d_transpose', 'output_shape', [batch_size, 28, 28, 1], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'sigmoid')
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [batch_size, 28, 28, 1])
    struct('type', 'conv2d', 'output_maps', 32, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'sub_sampling', 'scale', 2)
    struct('type', 'conv2d', 'output_maps', 64, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'sub_sampling', 'scale', 2)
    struct('type', 'reshape', 'output_shape', [batch_size, 3136])
    struct('type', 'fully_connect', 'output_shape', [batch_size, 1], 'activation', 'sigmoid')
};
```
result:

### example_2
model:
```
generator.layers = {
    struct('type', 'input', 'output_shape', [batch_size, 100]) 
    struct('type', 'fully_connect', 'output_shape', [batch_size, 1024], 'activation', 'relu')
    struct('type', 'fully_connect', 'output_shape', [batch_size, 28*28], 'activation', 'sigmoid') 
    struct('type', 'reshape', 'output_shape', [batch_size, 28, 28, 1])
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [batch_size, 28,28,1])
    struct('type', 'reshape', 'output_shape', [batch_size, 28*28]) 
    struct('type', 'fully_connect', 'output_shape', [batch_size, 1024], 'activation', 'relu')
    struct('type', 'fully_connect', 'output_shape', [batch_size, 1], 'activation', 'sigmoid') 
};
```
result:

## Reference
1. `https://grzegorzgwardys.wordpress.com/2016/04/22/8/`
2. `Dumoulin V, Visin F. A guide to convolution arithmetic for deep learning[J]. 2016.`
3. `https://github.com/rasmusbergpalm/DeepLearnToolbox/tree/master/CNN`