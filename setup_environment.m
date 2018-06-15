function setup_environment()
    root = root_nn();
    for dir_name = {'util', 'activation', 'error_term', 'gradient', 'layer', 'nerual_network_flow'}
        addpath(fullfile(root, char(dir_name)));
    end
end
function root_path = root_nn()
    root_path = fileparts(mfilename('fullpath'));
end