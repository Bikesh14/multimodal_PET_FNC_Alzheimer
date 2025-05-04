
function reconstruct_and_save_fnc(IC_FNC, main_dir)
    output_folder = [main_dir 'FNCs/'];
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Process each IC
    for idx = 1:size(IC_FNC, 1)

        full_matrix= icatb_vec2mat(IC_FNC(idx, :)');
        disp(size(full_matrix))
        % Save the matrix to a file
        mat_filename = sprintf('FNC_IC%03d.mat', idx);
        save(fullfile(output_folder, mat_filename), 'full_matrix');
        
        % Create and save a heatmap of the correlation matrix
        fig = figure('visible', 'off'); % Create a figure that doesn't pop up
        imagesc(full_matrix); % Display the matrix as a heatmap
        colormap(jet); 
        colorbar; 
        title(sprintf('Correlation Matrix Heatmap: IC%03d', idx), 'Interpreter', 'none');
        axis square; % Make the plot square
        
        % Save the heatmap as an image
        img_filename = sprintf('FNC_IC%03d.png', idx);
        saveas(fig, fullfile(output_folder, img_filename));
        close(fig); 
        
        % Display save messages
        disp(['Saved correlation matrix to ' fullfile(output_folder, mat_filename)]);
    end
end
