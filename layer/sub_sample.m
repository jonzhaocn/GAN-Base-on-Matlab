function output = sub_sample(input, scale)
    % average pool 
    % refer: https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/CNN/cnnff.m
    if scale>1
        output = convn(input,ones(scale,scale)/(scale^2),"valid");
        output = output(1:scale:end, 1:scale:end, :, :);
    else
        error('stride should > 1');
    end
end