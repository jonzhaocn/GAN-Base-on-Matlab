% https://blog.csdn.net/huachao1001/article/details/79131814
clear;
clc;
addpath('../base');
% -----------input
input(1,:,:,1)=[1,0,1;0,2,1;1,1,0];
input(1,:,:,2)=[2,0,2;0,1,0;1,0,0];
input(1,:,:,3)=[1,1,1;2,2,0;1,1,1];
input(1,:,:,4)=[1,1,2;1,0,1;0,2,2];
% ----------filter
filter(:,:,1,1)=[1,0,1;-1,1,0;0,-1,0];
filter(:,:,2,1)=[-1,0,1;0,0,1;1,1,1];
filter(:,:,3,1)=[0,1,1;2,0,1;1,2,1];
filter(:,:,4,1)=[1,1,1;0,2,1;1,0,1];
filter(:,:,1,2)=[1,0,2;-2,1,1;1,-1,0];
filter(:,:,2,2)=[-1,0,1;-1,2,1;1,1,1];
filter(:,:,3,2)=[0,0,0;2,2,1;1,-1,1];
filter(:,:,4,2)=[2,1,1;0,-1,1;1,1,1];
filter = rot90(filter,2);
% -----------layer---------
layer.filter = filter;
layer.output_shape = [1,12,12,2];
layer.padding = 'same';
layer.stride = 2;
% layer.input_shape = size(input);
layer.input_shape = [1, 6, 6, 4];
layer.kernel_size = 3;
% ------------
layer = check_conv2d_transpose_shape(layer);
output = conv2d_transpose(input, layer);
a = squeeze(output(1,:,:,1));