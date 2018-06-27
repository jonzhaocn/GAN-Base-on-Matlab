function generator = nn_bp_g(generator, discriminator)
    n = numel(generator.layers);
    % get other's error term
    for l = n:-1:2
        if l==n
            back_layer = discriminator.layers{2};
        else
           back_layer = generator.layers{l+1};
        end
        switch back_layer.type
            case 'conv2d'
                generator.layers{l}.d = get_error_term_from_conv2d_layer(back_layer);
            case 'fully_connect'
                generator.layers{l}.d = get_error_term_from_fully_connect_layer(back_layer);
            case 'reshape'
                generator.layers{l}.d = get_error_term_from_reshape_layer(back_layer);
            case 'conv2d_transpose'
                generator.layers{l}.d = get_error_term_from_conv2d_transpose_layer(back_layer);
            case 'sub_sampling'
                generator.layers{l}.d = get_error_term_from_sub_sampling_layer(back_layer);
            case 'atrous_conv2d'
                generator.layers{l}.d = get_error_term_from_atrous_conv2d_layer(back_layer);
            case 'batch_norm'
                generator.layers{l}.d = get_error_term_from_batch_norm_layer(back_layer);
            otherwise
                error('error net.layers{l}.type:%s', back_layer.type);
        end
        generator.layers{l}.d = generator.layers{l}.d .* delta_activation_function(generator.layers{l});
    end
    %% calculate every layer's gradient
    for l = 2:n
        front_a = generator.layers{l-1}.a;
        switch generator.layers{l}.type
            case 'conv2d'
                [dfilter, dbiases] = calculate_gradient_for_conv2d_layer(front_a, generator.layers{l});
                generator.layers{l}.dfilter = dfilter;
                generator.layers{l}.dbiases = dbiases;
            case 'fully_connect'
                [dweights, dbiases] = calculate_gradient_for_fully_connect_layer(front_a, generator.layers{l});
                generator.layers{l}.dweights = dweights;
                generator.layers{l}.dbiases = dbiases;
            case {'reshape', 'sub_sampling'}
                continue
            case 'conv2d_transpose'
                [dfilter, dbiases] = calculate_gradient_for_conv2d_transpose_layer(front_a, generator.layers{l});
                generator.layers{l}.dfilter = dfilter;
                generator.layers{l}.dbiases = dbiases;
            case 'atrous_conv2d'
                [dfilter, dbiases] = calculate_gradient_for_atrous_conv2d_layer(front_a, generator.layers{l});
                generator.layers{l}.dfilter = dfilter;
                generator.layers{l}.dbiases = dbiases;
            case 'batch_norm'
                [dweights, dbiases] = calculate_gradient_for_batch_norm_layer(front_a, generator.layers{l});
                generator.layers{l}.dweights = dweights;
                generator.layers{l}.dbiases = dbiases;
            otherwise
                error('error net.layers{l}.type:%s',generator.layers{l}.type);
        end
    end
end