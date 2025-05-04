

% data: voxels*IC
function fun_write_nifti(data, MaskBrainfile, main_dir)
    % Load mask information
    info = niftiinfo(MaskBrainfile);
    temp = niftiread(MaskBrainfile);           
    temp = double(temp);              
    VoxIdx = find(temp ~= 0); % non-zero voxel indexes in the mask

    % Create a folder for individual IC files
    output_folder = [main_dir 'ICs/'];
    mkdir(output_folder);

    % Initialize an empty 4D array
    IC_4D = [];

    % Process each IC
    for idx = 1:size(data, 2)
        % Create a blank 3D volume
        IM = zeros(size(temp));     
        IM(VoxIdx) = data(:, idx);    % Map z-scored IC values to non-zero voxels

        % Save to the 4D array
        if idx == 1
            IC_4D = IM;  % First 3D volume initializes the 4D array
        else
            IC_4D = cat(4, IC_4D, IM);  % Concatenate along the 4th dimension
        end

        % Save the individual IC as a 3D NIfTI file
        niftiwrite(single(IM), [output_folder 'IC' sprintf('%03d', idx) '.nii'], info);
    end

    % Create a new info structure for the 4D NIfTI file
    info_4d = info;  % Copy the 3D info structure
    info_4d.ImageSize = [size(temp), size(data, 2)];  % [x, y, z, num_ICs]
    info_4d.PixelDimensions = [info.PixelDimensions, 1];  % Append a value for the 4th dimension

    output_folder_all_ICs = [main_dir 'ICs_combined/'];
    mkdir(output_folder_all_ICs);
    % Save the entire 4D NIfTI file
    output_file_4d = [output_folder_all_ICs 'all_IC_components.nii'];
    niftiwrite(single(IC_4D), output_file_4d, info_4d);

    % Display completion messages
    disp(['All ICs saved in folder: ' output_folder_all_ICs]);
    disp(['Individual ICs saved in folder: ' output_folder]);
end