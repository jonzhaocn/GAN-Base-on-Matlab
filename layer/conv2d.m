% input, filter both have 4 dims£¬input:[None,height,width,channels],filter:[height,width,input_channels,output_channels]
% the stride of conv is 1
function output = conv2d(input, filter, padding)
   [batch_size,in_height,in_width,in_channel] = size(input);
   [filter_height,filter_width,filter_in_channel,out_channel] = size(filter);
   % https://www.tensorflow.org/api_guides/python/nn#Convolution
       
   if strcmp(padding, 'valid')
       % there is no padding
       out_height = in_height-filter_height+1;
       out_width = in_width-filter_width+1;
   elseif strcmp(padding, 'same')
       out_height = in_height;
       out_width = in_width;
       p_top = floor(filter_height/2);
       p_left = floor(filter_width/2);
       input = padding_height_width_in_array(input, p_top, p_top, p_left, p_left);
   else
       error('padding of conv2d should be same or valid');
   end
   
   output = zeros(out_height, out_width, batch_size, out_channel);
   input = permute(input,[2,3,4,1]);
   for jj = 1:out_channel
       output(:,:,:,jj) = squeeze(convn(input(:,:,:,:), flip(filter(:,:,:,jj), 3), "valid"));
   end
   output = permute(output, [3,1,2,4]);
end