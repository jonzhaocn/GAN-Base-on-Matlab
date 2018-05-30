function layer = check_conv2d_transpose_layer(layer)
    % ----------init-------------
    in_height = layer.input_shape(2);
    in_width = layer.input_shape(3);
    out_height = layer.output_shape(2);
    out_width = layer.output_shape(3);
    stride = layer.stride;
    kernel_size = layer.kernel_size;
    % --------calculate-----------
    if strcmp(layer.padding, 'valid')
        if stride == 1
            % valid,stride=1 => full
            if ~isequal([in_height, in_width] + kernel_size - 1, [out_height, out_width])
                error(['wrong conv2d_transpose output shape,output height width:', num2str([out_height, out_width]), ', should be ', num2str([in_height, in_width] + kernel_size - 1)]);
            end
        else
            % i-k should be a multiple of stride
            if ~isequal( mod([in_height, in_width] - kernel_size, stride) , [0, 0])
                error(['wrong input shape or kernel size, i-k should be a multiple of s', ' i:',num2str([in_height, in_width]), ', k:', num2str(kernel_size), ', stride:', num2str(stride)]);
            end
            % if the stride > 1
            if ~isequal(stride * ([in_height, in_width]-1) + kernel_size, [out_height, out_width])
                error(['wrong output shape,output height width:', num2str([out_height, out_width]), ', should be :',num2str(stride * ([in_height, in_width]-1) + kernel_size)]);
            end
        end
    elseif strcmp(layer.padding, 'same')
        % -------same stride=1-----
        if stride == 1
            if ~isequal([in_height, in_width], [out_height, out_width])
                error('wrong output shape')
            end
            % save padding_shape
            layer.padding_shape = [floor(kernel_size/2), floor(kernel_size/2)];
        % ----------same stride>1-----------
        else
            % https://www.tensorflow.org/versions/master/api_guides/python/nn#Convolution
            % verify that if we can get the output shape from conving the
            % input
            if ~isequal(ceil([out_height, out_width]/stride), [in_height, in_width])
                error(['error output shape:', num2str([out_height, out_width])])
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
            padding_shape = [pad_along_height/2, pad_along_width/2];
            % a_padding_shape is the padding in right and bottom 
            a_padding_shape = mod([out_height, out_width] + 2*padding_shape - kernel_size, stride);
            % verify the padding_shape and a_padding_shape are right
            if ~isequal(stride * ([in_height, in_width]-1) + a_padding_shape + kernel_size - 2*padding_shape, [out_height, out_width])
                error('something wrong in check_conv2d_transpose_shape');
            end
            % padding in top, bottom, left, right
            layer.padding_shape = padding_shape;
            layer.a_padding_shape = a_padding_shape;
        end
    else
        error('padding of conv2_transpose:only support valid or same');
    end
end