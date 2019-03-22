clear;
clc;
% -----------load mnist data
load('mnist_uint8', 'train_x');
train_x = double(reshape(train_x, 60000, 28, 28))/255;
% train_x:[height, width, channel, images_index]
train_x = permute(train_x,[3,2,4,1]);
batch_size = 60;
% ----------- model
generator.layers = {
    struct('type', 'input', 'output_shape', [100, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [28*28*6, batch_size])
    struct('type', 'reshape', 'output_shape', [28, 28, 6, batch_size])
    struct('type', 'atrous_conv2d', 'kernel_size', 5, 'output_maps', 3, 'padding', 'same', 'rate', 2, 'activation', 'leaky_relu')
    struct('type', 'batch_norm', 'activation', 'leaky_relu')
    struct('type', 'conv2d', 'kernel_size', 5, 'output_maps', 1, 'padding', 'same', 'activation', 'sigmoid')
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [28, 28, 1, batch_size])
    struct('type', 'reshape', 'output_shape', [28*28, batch_size])
    struct('type', 'fully_connect', 'output_shape', [1024, batch_size], 'activation', 'leaky_relu')
    struct('type', 'fully_connect', 'output_shape', [1, batch_size], 'activation', 'sigmoid')
};
args = struct('batch_size', batch_size, 'epoch', 10, 'learning_rate', 0.001, 'optimizer', 'adam', 'results_folder', 'results');
[generator, discriminator] = gan_train(generator, discriminator, train_x, args);