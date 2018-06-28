% input, weights both have 4 dims£¬input:[None,height,width,channels],weights:[height,width,input_channels,output_channels]
% the stride of conv is 1
function output = conv2d(input, layer)
    weights = layer.weights;
    [in_height, in_width, in_channel, batch_size] = size(input);
    [weights_height, weights_width, weights_in_channel, out_channel] = size(weights);
    % https://www.tensorflow.org/api_guides/python/nn#Convolution
    switch layer.padding
        case 'valid'
            % there is no padding
            out_height = in_height-weights_height+1;
            out_width = in_width-weights_width+1;
        case 'same'
            out_height = in_height;
            out_width = in_width;
            input = padding_height_width_in_array(input, layer.padding_shape);
        otherwise
            error('padding of conv2d should be same or valid');
    end

    output = zeros(out_height, out_width, out_channel, batch_size);
    for jj = 1:out_channel
        output(:,:,jj,:) = convn(input, flip(weights(:,:,:,jj), 3), "valid");
    end
end