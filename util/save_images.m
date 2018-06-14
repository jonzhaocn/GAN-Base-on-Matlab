% 保存图片，便于观察generator生成的images_fake
function save_images(images, count, path)
    [image_height, image_width, image_channel, ~] = size(images);
    row = count(1);
    col = count(2);
    I = zeros(row*image_height, col*image_width, image_channel);
    for i = 1:row
        for j = 1:col
            r_s = (i-1)*image_height+1;
            c_s = (j-1)*image_width+1;
            index = (i-1)*col + j;
            pic = reshape(images(:, :, :, index), image_height, image_width, image_channel);
            I(r_s:r_s+image_height-1, c_s:c_s+image_width-1, :) = pic;
        end
    end
    imwrite(I, path);
end