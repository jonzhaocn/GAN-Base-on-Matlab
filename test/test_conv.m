% https://blog.csdn.net/huachao1001/article/details/79120521
clear;
clc;
addpath('../base');
% -------input
input(1,:,:,1)=[1,0,1,2,1;0,2,1,0,1;1,1,0,2,0;2,2,1,1,0;2,0,1,2,0];
input(1,:,:,2)=[2,0,2,1,1;0,1,0,0,2;1,0,0,2,1;1,1,2,1,0;1,0,1,1,1];
% -------filter
filter(:,:,1,1)=[1,0,1;-1,1,0;0,-1,0];
filter(:,:,2,1)=[-1,0,1;0,0,1;1,1,1];
filter = rot90(filter,2);
output = conv2d(input,filter,"same");
a = squeeze(output(1,:,:,1));