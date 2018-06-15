% input, filter both have 4 dims£¬input:[height,width,channels,batch_size],filter:[height,width,input_channels,output_channels]
% outputshape:[height,width,output_channels,batch_size]
function output = conv2d_transpose(input, layer)
    filter = layer.filter;
    [in_height, in_width, in_channel, batch_size] = size(input);
    [filter_height, filter_width, filter_in_channel, filter_out_channel] = size(filter);
    out_channel = layer.output_shape(3);
    switch layer.padding
        case 'valid'
            if layer.stride == 1
                out_height = in_height + filter_height - 1;
                out_width = in_width + filter_width - 1;
            else
                out_height = layer.stride * (in_height - 1) + filter_height;
                out_width = layer.stride * (in_width - 1) + filter_width;
                input = insert_zeros_into_array(input, layer.stride);
            end
            input = padding_height_width_in_array(input, layer.padding_shape);
        case 'same'
            if layer.stride == 1
                % kernel size should be a odd
                out_height = in_height;
                out_width = in_width;
                input = padding_height_width_in_array(input, layer.padding_shape);
            else
                out_height = layer.output_shape(1);
                out_width = layer.output_shape(2);
                % insert 0
                input = insert_zeros_into_array(input, layer.stride);
                % padding in top,bottom,left,right
                input = padding_height_width_in_array(input, layer.padding_shape+layer.a_padding_shape);
            end
        otherwise
            error('padding of conv2d transpose should be same or valid')
    end
    output = zeros(out_height, out_width, out_channel, batch_size);
    for jj=1:out_channel
        output(:,:,jj,:) = convn(input, flip(layer.filter(:,:,:,jj), 3), "valid");
    end
end