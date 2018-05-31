function layer = setup_conv2d_transpose_layer(input_shape, layer)
    % --------check field names
    required_fields = {'type', 'kernel_size', 'output_shape', 'stride', 'padding'};
    optional_fields = {'activation'};
    check_layer_field_names(layer, required_fields, optional_fields);
    % ----------init-------------
    layer.input_shape = input_shape;
    output_shape = layer.output_shape;
    
    in_height = input_shape(2);
    in_width = input_shape(3);
    out_height = output_shape(2);
    out_width = output_shape(3);
    stride = layer.stride;
    kernel_size = layer.kernel_size;
    % --------check outputshape-----------
    if strcmp(layer.padding, 'valid')
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
    elseif strcmp(layer.padding, 'same')
        % -------same stride=1-----
        if stride == 1
            if ~isequal([in_height, in_width], [out_height, out_width])
                error('conv2d transpose layer: stride==1, padding is same, wrong output height/width [%s], output height/width should be [%s]',...
                    num2str([out_height, out_width]), num2str([in_height, in_width]));
            end
            % save padding_shape
            layer.padding_shape = [floor(kernel_size/2), floor(kernel_size/2)];
        % ----------same stride>1-----------
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
            padding_shape = [pad_along_height/2, pad_along_width/2];
            % a_padding_shape is the padding in right and bottom 
            a_padding_shape = mod([out_height, out_width] + 2*padding_shape - kernel_size, stride);
            % verify the padding_shape and a_padding_shape are right
            if ~isequal(stride * ([in_height, in_width]-1) + a_padding_shape + kernel_size - 2*padding_shape, [out_height, out_width])
                error('something wrong in setup conv2d transpose layer function ');
            end
            % padding in top, bottom, left, right
            layer.padding_shape = padding_shape;
            layer.a_padding_shape = a_padding_shape;
        end
    else
        error('padding of conv2_transpose:only support valid or same');
    end
    % ----- setting
    if numel(input_shape)==3
        input_maps = 1;
    elseif numel(input_shape)==4
        input_maps = input_shape(end);
    else
        error('error input_shape in conv2d transpose layer, dims of input should be 3 or 4, now input shape is [%s]', num2str(input_shape))
    end
    
    layer.filter = normrnd(0, 0.02, kernel_size, kernel_size, input_maps, output_shape(end));
    layer.biases = normrnd(0, 0.02, 1, output_shape(end));
    layer.filter_m = 0;
    layer.filter_v = 0;
    layer.biases_m = 0;
    layer.biases_v = 0;
end