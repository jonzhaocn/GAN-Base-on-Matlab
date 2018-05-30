function output = sub_sample(input, scale)
    % input [batch_size, height, width, channel]
    input = permute(input, [2,3,1,4]);
    % average pool 
    % refer: https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/CNN/cnnff.m
    if scale>1
        output = convn(input,ones(scale,scale)/(scale^2),"valid");
        output = output(1:scale:end, 1:scale:end, :, :);
        output = permute(output, [3,1,2,4]);
    else
        error('stride should > 1');
    end
end