% input, filter both have 4 dims£¬input:[None,height,width,channels],filter:[height,width,input_channels,output_channels]
% the stride of conv is 1
function output = conv2d(input, layer)
    filter = layer.filter;
    [in_height, in_width, in_channel, batch_size] = size(input);
    [filter_height, filter_width, filter_in_channel, out_channel] = size(filter);
    % https://www.tensorflow.org/api_guides/python/nn#Convolution

    if strcmp(layer.padding, 'valid')
        % there is no padding
        out_height = in_height-filter_height+1;
        out_width = in_width-filter_width+1;
    elseif strcmp(layer.padding, 'same')
        out_height = in_height;
        out_width = in_width;
        p_top = layer.padding_shape(1);
        p_left = layer.padding_shape(2);
        input = padding_height_width_in_array(input, p_top, p_top, p_left, p_left);
    else
        error('padding of conv2d should be same or valid');
    end

    output = zeros(out_height, out_width, out_channel, batch_size);
    for jj = 1:out_channel
        output(:,:,jj,:) = convn(input, flip(filter(:,:,:,jj), 3), "valid");
    end
end