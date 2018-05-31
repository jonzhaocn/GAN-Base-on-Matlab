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
            generator.layers{l}.d = get_error_term_from_fully_connect_layer(back_layer);
        % --------------reshape
        elseif strcmp(back_layer.type, 'reshape')
            generator.layers{l}.d = get_error_term_from_reshape_layer(back_layer);
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
        front_a = generator.layers{l-1}.a;
        % ---------conv
        if strcmp(generator.layers{l}.type, 'conv2d')
            [dfilter, dbiases] = calculate_gradient_for_conv2d_layer(front_a, generator.layers{l});
            generator.layers{l}.dfilter = dfilter;
            generator.layers{l}.dbiases = dbiases;
        % ---------fully connect
        elseif strcmp(generator.layers{l}.type, 'fully_connect')
            [dweights, dbiases] = calculate_gradient_for_fully_connect_layer(front_a, generator.layers{l});
            generator.layers{l}.dweights = dweights;
            generator.layers{l}.dbiases = dbiases;
        % ----------reshape
        elseif strcmp(generator.layers{l}.type, 'reshape')
            continue;
        % ---------conv transpose
        elseif strcmp(generator.layers{l}.type, 'conv2d_transpose')
            [dfilter, dbiases] = calculate_gradient_for_conv2d_transpose_layer(front_a, generator.layers{l});
            generator.layers{l}.dfilter = dfilter;
            generator.layers{l}.dbiases = dbiases;
        elseif strcmp(generator.layers{l}.type, 'sub_sampling')
            continue
        elseif strcmp(generator.layers{l}.type, 'atrous_conv2d')
            [dfilter, dbiases] = calculate_gradient_for_atrous_conv2d_layer(front_a, generator.layers{l});
            generator.layers{l}.dfilter = dfilter;
            generator.layers{l}.dbiases = dbiases;
        else
            error(['error net.layers{l}.type:',generator.layers{l}.type]);
        end
    end
end