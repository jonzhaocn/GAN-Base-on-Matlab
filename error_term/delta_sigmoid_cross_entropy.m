% sigmoid_cross_entropy is the derivative respect to logits£¬the logits have not
% been activated
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
% sigmoid_cross_entropy = max(logits, 0) - logits * labels + log(1 + exp(-abs(logits)))
% logits are the output of discriminator£¬labels are the right
% classification of images
function result = delta_sigmoid_cross_entropy(logits, labels)
    % --------
    temp1 = logits;
    temp1(logits>=0) = 1;
    temp1 = max(temp1,0);
    % --------
    temp2 = logits;
    temp2(temp2>=0) = -1;
    temp2(temp2<0) = 1;
    % --------
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end