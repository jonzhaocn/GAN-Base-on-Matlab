function layer = setup_conv2d_transpose_layer(input_shape, layer)
    % --------check field names
    required_fields = {'type', 'kernel_size', 'output_shape', 'stride', 'padding'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    % ----------init-------------
    layer.input_shape = input_shape;
    output_shape = layer.output_shape;
    % --
    in_height = input_shape(1);
    in_width = input_shape(2);
    input_maps = input_shape(3);
    % --
    out_height = output_shape(1);
    out_width = output_shape(2);
    output_maps = output_shape(3);
    % --
    stride = layer.stride;
    kernel_size = layer.kernel_size;
    % --------check outputshape-----------
    switch layer.padding
        % ---------valid-----------
        case 'valid'
            if stride == 1
                % valid,stride=1 => full
                if ~isequal([in_height, in_width] + kernel_size - 1, [out_height, out_width])
                    error('conv2d transpose layer:padding is valid, wrong output height/width [%s], output height/width should be [%s]',...
                        num2str([out_height, out_width]), num2str([in_height, in_width] + kernel_size - 1));
                end
            else
                % i-k should be a multiple of stride
                if ~isequal( mod([in_height, in_width] - kernel_size, stride) , [0, 0])
                    error('conv2d transpose layer: stride>1, padding is valid, wrong input shape or kernel size, input height/width - kernel_size should be a multiple of stride, input height/width is [%s], kernel_size is %d, stride is %d', ...
                        num2str([in_height, in_width]), kernel_size, stride);
                end
                % if the stride > 1
                if ~isequal(stride * ([in_height, in_width]-1) + kernel_size, [out_height, out_width])
                    error('conv2d transpose layer: stride>1, padding is valid, wrong output height/width: [%s], output height/width should be [%s]',...
                        num2str([out_height, out_width]), num2str(stride * ([in_height, in_width]-1) + kernel_size));
                end
            end
            layer.padding_shape = [kernel_size-1, kernel_size-1, kernel_size-1, kernel_size-1];
        % -------same---------    
        case 'same'
            if mod(kernel_size,2)==0
                error('conv2d transpose layer, padding is same, kernel size should be a odd');
            end
            if stride == 1
                if ~isequal([in_height, in_width], [out_height, out_width])
                    error('conv2d transpose layer: stride==1, padding is same, wrong output height/width [%s], output height/width should be [%s]',...
                        num2str([out_height, out_width]), num2str([in_height, in_width]));
                end
                layer.padding_shape = [floor(kernel_size/2), floor(kernel_size/2), floor(kernel_size/2), floor(kernel_size/2)];
            else
                % https://www.tensorflow.org/versions/master/api_guides/python/nn#Convolution
                % verify that if we can get the output shape from conving the
                % input
                if ~isequal(ceil([out_height, out_width]/stride), [in_height, in_width])
                    error(['conv2d transpose layer: stride>1, padding is same, wrong output shape [%s]', num2str([out_height, out_width])])
                end
                % calculate the padding
                if mod(out_height, stride)==0
                    pad_along_height = max(kernel_size - stride, 0);
                else
                    pad_along_height = max(kernel_size - mod(out_height, stride), 0);
                end
                if mod(out_width, stride)==0
                    pad_along_width = max(kernel_size - stride, 0);
                else
                    pad_along_width = max(kernel_size - mod(out_width, stride), 0);
                end
                % pad_along_height should be a even
                for i=0:stride-1
                    temp = pad_along_height + i;
                    if mod(temp, 2) == 0
                        pad_along_height = temp;
                        break;
                    end
                end
                % pad_along_width should be a even
                for i=0:stride-1
                    temp = pad_along_width + i;
                    if mod(temp, 2) == 0
                        pad_along_width = temp;
                        break;
                    end
                end
                % p is the padding in top,bottom,left,right, top=bottom,
                % left=right
                padding_shape = [pad_along_height/2, pad_along_height/2, pad_along_width/2, pad_along_width/2];
                % a_padding_shape is the padding in right and bottom
                a_padding_shape_bottom = mod(out_height + pad_along_height - kernel_size, stride);
                a_padding_shape_right = mod(out_width + pad_along_width - kernel_size, stride);
                % verify the padding_shape and a_padding_shape are right
                if ~isequal(stride * ([in_height, in_width]-1) + [a_padding_shape_bottom, a_padding_shape_right] + kernel_size - [pad_along_height, pad_along_width], [out_height, out_width])
                    error('something wrong in setup conv2d transpose layer function ');
                end
                % padding in top, bottom, left, right
                layer.padding_shape = padding_shape;
                layer.a_padding_shape = [0, a_padding_shape_bottom, 0, a_padding_shape_right];
            end
        otherwise
            error('padding of conv2_transpose:only support valid or same');
    end    
    layer.filter = normrnd(0, 0.01, kernel_size, kernel_size, input_maps, output_maps);
    layer.biases = normrnd(0, 0.01, 1, output_maps);
    layer.filter_m = 0;
    layer.filter_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end