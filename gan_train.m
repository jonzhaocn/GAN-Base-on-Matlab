function [generator, discriminator] = gan_train(g_structure, d_structure, train_images, args)
    % ------
    setup_environment();
    % ----------- setting
    options.epoch = 1;
    options.batch_size = 10;
    options.learning_rate = 0.001;
    options.optimizer = 'sgd';
    options = argparse(options, args);
    % -----------
    images_count = size(train_images, 4);
    batch_num = ceil(images_count / options.batch_size);
    switch options.optimizer
        case 'sgd'
            nn_applygrads = @nn_applygrads_sgd;
        case 'adam'
            nn_applygrads = @nn_applygrads_adam;
        otherwise
            error('unsupported optimizer type:%s', options.optimizer);
    end
    % ----------
    generator = nn_setup(g_structure);
    discriminator = nn_setup(d_structure);
    
    for e=1:options.epoch
        kk = randperm(images_count);
        for t=1:batch_num
            % perpare data
            batch_index_start =  (t - 1) * options.batch_size + 1;
            batch_index_end = min( t* options.batch_size, numel(kk));
            images_real = train_images(:, :, :, batch_index_start:batch_index_end);
            noise = unifrnd(-1, 1, 100, options.batch_size);
            % tranning
            % -----------generator is fixed£¬update discriminator
            generator = nn_ff(generator, noise);
            images_fake = generator.layers{end}.a;
            discriminator = nn_ff(discriminator, images_fake);
            logits_fake = discriminator.layers{end}.z;
            discriminator = nn_bp_d(discriminator, logits_fake, ones(size(logits_fake)));
            generator = nn_bp_g(generator, discriminator);
            generator = nn_applygrads(generator, options.learning_rate);
            % -----------discriminator is fixed£¬update generator
            generator = nn_ff(generator, noise);
            images_fake = generator.layers{end}.a;
            images = cat(4, images_fake, images_real);
            discriminator = nn_ff(discriminator, images);
            logits = discriminator.layers{end}.z;
            labels = ones(size(logits));
            labels(1:size(images_fake, 4)) = 0;
            discriminator = nn_bp_d(discriminator, logits, labels);
            discriminator = nn_applygrads(discriminator, options.learning_rate);
            % ----------------output loss
            if t == batch_num || mod(t, 100)==0
                c_loss = sigmoid_cross_entropy(logits(:, 1:options.batch_size), ones(1, options.batch_size));
                d_loss = sigmoid_cross_entropy(logits, labels);
                fprintf('epoch:%d, t:%d, c_loss:"%f",d_loss:"%f"\n', e, t, c_loss, d_loss);
            end
            if t == batch_num || mod(t, 100)==0
                path = sprintf('./pics/epoch_%d_t_%d.png',e,t);
                save_images(images_fake, [4, 4], path);
                fprintf('save_sample:%s\n', path);
            end
        end
    end
end