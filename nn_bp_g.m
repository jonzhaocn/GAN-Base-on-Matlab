function generator = nn_bp_g(generator, discriminator)
    n = numel(generator.layers);
    % get other's error term
    for l = n:-1:2
        if l==n
            back_layer = discriminator.layers{2};
        else
           back_layer = generator.layers{l+1};
        end
        % --------------conv
        if strcmp(back_layer.type, 'conv2d')
            generator.layers{l}.d = get_error_term_from_conv2d_layer(back_layer);
        % -------------- fully connect
        elseif strcmp(back_layer.type, 'fully_connect')
            generator.layers{l}.d = back_layer.d * back_layer.weights';
        % --------------reshape
        elseif strcmp(back_layer.type, 'reshape')
            d = back_layer.d;
            generator.layers{l}.d = reshape(d, [size(d, 1) ,back_layer.input_shape(2:end)]);
        % --------------conv transpose
        elseif strcmp(back_layer.type, 'conv2d_transpose')
            generator.layers{l}.d = get_error_term_from_conv2d_transpose_layer(back_layer);
        elseif strcmp(back_layer.type, 'sub_sampling')
            generator.layers{l}.d = get_error_term_from_sub_sampling_layer(back_layer);
        elseif strcmp(back_layer.type, 'atrous_conv2d')
            generator.layers{l}.d = get_error_term_from_atrous_conv2d_layer(back_layer);
        else
            error(['error net.layers{l}.type:', back_layer.type]);
        end
        generator.layers{l}.d = generator.layers{l}.d .* delta_activation_function(generator.layers{l});
    end
    %% calculate every layer's gradient
    for l = 2:n
        % ---------conv
        if strcmp(generator.layers{l}.type, 'conv2d')
            [dfilter, dbiases] = calculate_gradient_for_conv2d_layer(generator.layers{l-1}.a, generator.layers{l});
            generator.layers{l}.dfilter = dfilter;
            generator.layers{l}.dbiases = dbiases;
        % ---------fully connect
        elseif strcmp(generator.layers{l}.type, 'fully_connect')
            d = generator.layers{l}.d;
            a = generator.layers{l-1}.a;
            generator.layers{l}.dweights = a'*d / size(d, 1);
            generator.layers{l}.dbiases = mean(d, 1);
        % ----------reshape
        elseif strcmp(generator.layers{l}.type, 'reshape')
            continue;
        % ---------conv transpose
        elseif strcmp(generator.layers{l}.type, 'conv2d_transpose')
            [dfilter, dbiases] = calculate_gradient_for_conv2d_transpose_layer(generator.layers{l-1}.a, generator.layers{l});
            generator.layers{l}.dfilter = dfilter;
            generator.layers{l}.dbiases = dbiases;
        elseif strcmp(generator.layers{l}.type, 'sub_sampling')
            continue
        elseif strcmp(generator.layers{l}.type, 'atrous_conv2d')
            [dfilter, dbiases] = calculate_gradient_for_atrous_conv2d_layer(generator.layers{l-1}.a, generator.layers{l});
            generator.layers{l}.dfilter = dfilter;
            generator.layers{l}.dbiases = dbiases;
        else
            error(['error net.layers{l}.type:',generator.layers{l}.type]);
        end
    end
end