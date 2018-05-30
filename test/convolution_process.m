clc;
clear;
addpath('../base');
input = create_array('a', 9, 9);
filter = create_array('k', 3, 3);
filter = insert_zeros_into_array(filter,2);
filter = rot90(filter,2);
conv_process(input,filter);
function result = create_array(name, height, width)
    result = cell(height, width);
    for i = 1:height
        for j=1:width
            result(i, j) = cellstr([name, int2str(i), '_' ,int2str(j)]);
        end
    end
end
function result = insert_zeros_into_array(array, stride)
    [height, width] = size(array);
    result = cell(height+(height-1)*(stride-1), width+(width-1)*(stride-1));
    for i=1:size(result,1)
        for j=1:size(result,2)
            result(i,j) = cellstr('0');
        end
    end
    result(1:stride:end,1:stride:end) = array;
end
function result = padding_height_width(array, p_top, p_bottom, p_left, p_right)
    [height,width] = size(array);
    result = cell(size(array)+[p_top+p_bottom, p_left+p_right]);
    for i=1:size(result,1)
        for j=1:size(result,2)
            result(i,j) = cellstr('0');
        end
    end
    result(1+p_top:p_top+height, 1+p_left:p_left+width) = array;
end
function conv_process(input, filter)
    [input_h, input_w] = size(input);
    [filter_h, filter_w] = size(filter);
    for i = 1:input_h-filter_h+1
        for j=1:input_w-filter_w+1
            result = strcat('error_term(', num2str(i), ',', num2str(j), ')=\t');
            for kk = 1:filter_h
                for ll = 1:filter_w
                    if ~strcmp(input{i+kk-1,j+ll-1}, '0') && ~strcmp(filter(kk, ll), '0')
                        result = strcat(result , char(input(i+kk-1, j+ll-1)) , '*' , char(filter(kk, ll)), '\t\t');
                    end
                end
            end
            result = strcat(result, '\n');
            fprintf(result);
        end
    end
end