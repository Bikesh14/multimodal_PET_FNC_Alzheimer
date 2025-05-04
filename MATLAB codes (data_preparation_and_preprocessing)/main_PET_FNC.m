%% add path
addpath(genpath('/data/users2/airaji/sources_common/toolbox/GroupICATv4.0c/'))
addpath(genpath('/data/users2/airaji/sources_common/toolbox/'))
addpath(genpath('/data/users2/airaji/sources_common/src/'))
addpath('/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC')
addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/GroupICAT'))

%% Load FNC Data
clc; 

clear all; 
close all;
addpath('/data/users4/aballem1/adni_analysis/');
load('/data/users4/aballem1/adni_ica_multiscale/data/adni_demo_fnc_age_corrected_dx_corrected.mat','adni_demo_subset','adni_fnc_meta_subset','adni_fnc_subset','scan_baseline_gap','final_dx','first_dx','dx_transition','dx_transition_non_empty');

adni_demo_subset.DX = string(final_dx);
adni_demo_subset.AGE = adni_demo_subset.AGE_CORRECTED;

adni_demo_subset = [adni_demo_subset adni_fnc_meta_subset];
adni_data = adni_demo_subset;
adni_fnc = adni_fnc_subset;
clear adni_demo_subset adni_fnc_meta_subset adni_fnc_subset dx_transition_non_empty dx_transition final_dx first_dx scan_baseline_gap

%% Extracting SubID and study date from FNC 
fnc_subjectIDs = adni_data.RID; 
% Convert numeric RID to string with leading zeros
fnc_subIDLastDigits = string(str2double(fnc_subjectIDs));  % Convert to double first if not already
fnc_subIDLastDigits = strtrim(cellstr(num2str(fnc_subjectIDs, '%04d')));  % Format with leading zeros
fnc_subIDLastDigits = string(fnc_subIDLastDigits);

fnc_studyDates = adni_data.EXAMDATE; 

fnc_studyDates = datetime(fnc_studyDates, 'InputFormat', 'yyyy-MM-dd');
fnc_studyDates.Format = 'dd-MMM-yyyy';  

fnc_meta = table(fnc_subIDLastDigits, fnc_studyDates, 'VariableNames', {'subID', 'studyDate'});

fnc_metadata_filename = 'updated_adni_fnc_metadata.mat';

% Save the table as a mat file
save(fnc_metadata_filename, 'fnc_meta');
disp(['Table saved as ', fnc_metadata_filename]);

%% Define the root directory containing the PET data folders
pet_data_folder_orig = '/data/qneuromark/Data/ADNI/PET_tracers/FBP/FBP_SUVR_BIDS/derivatives/GIFT-BIDS/preproc';
pet_data_folder = '/data/users3/gnagaboina1/PET/';

%%
% Find all subject directories starting with 'sub-'
sub_dirs = dir(fullfile(pet_data_folder_orig, 'sub-*'));
sub_dirs = sub_dirs([sub_dirs.isdir]);  % Filter to include only directories
% Initialize a table to store extracted data
pet_metadata = table('Size', [0, 2], 'VariableTypes', {'string', 'datetime'}, 'VariableNames', {'subID', 'studyDate'});

% Loop through each subject directory
for i = 1:length(sub_dirs)
    sub_directory_name=sub_dirs(i).name;
    disp(sub_directory_name)
    subIDMatch = regexp(sub_dirs(i).name, 'sub-(\d+)', 'tokens', 'once');
    subID = string(subIDMatch{1});  

    % Find all session directories within each subject directory
    ses_dirs = dir(fullfile(pet_data_folder_orig, sub_directory_name, 'ses-*'));
    ses_dirs = ses_dirs([ses_dirs.isdir]);  % Filter to include only directories

    % Loop through each session directory
    for j = 1:length(ses_dirs)
        sessionID = ses_dirs(j).name;
        % Extract study date from session ID (format: 'ses-YYMMDD')
        dateMatch = regexp(sessionID, 'ses-(\d{6})', 'tokens', 'once');
        studyDateStr = dateMatch{1};  % Extracted date in 'YYMMDD' format

        % Convert the study date string from 'YYMMDD' to datetime
        pet_studyDate = datetime(studyDateStr, 'InputFormat', 'yyMMdd', 'Format', 'dd-MMM-yyyy');
        % Append the extracted data to the table
        pet_metadata = [pet_metadata; {subID, pet_studyDate}];
    end
end

new_pet_metadata_filename = 'longitudinal_pet_metadata.mat';
save(new_pet_metadata_filename, 'pet_metadata');

% writetable(pet_metadata, 'PET_ORIGINAL.csv')
disp(['Updated file saved as: ', new_pet_metadata_filename]);

%% Load data from files
loadedFncMetadata = load('updated_adni_fnc_metadata.mat');
fnc_metadata = loadedFncMetadata.fnc_meta;

loadedPetMetadata = load('longitudinal_pet_metadata.mat');
pet_metadata = loadedPetMetadata.pet_metadata;

% Get all unique subject IDs
unique_subIDs = unique(pet_metadata.subID);
disp(unique_subIDs)
%% Process each unique subID using the function for the closest date match

% Initialize tables to hold all results
all_matched_results = table();

for i = 1:numel(unique_subIDs)
    subID = unique_subIDs(i);
    [closest_result, num_closest] = match_pet_fnc(fnc_metadata, pet_metadata, subID);
    % Append results
    all_matched_results = [all_matched_results; closest_result];
end

%%
% Save closest results to a file
save('all_matched_results.mat', 'all_matched_results');
disp(['Total number of closest matches: ', num2str(height(all_matched_results))]);

% Filter for closest results where difference is less than 30 days
filtered_matched_results = all_matched_results(all_matched_results.daysDifference <= 30, :);

% Save filtered closest results to a CSV file
save('filtered_matched_results.mat', 'filtered_matched_results');
disp(['Total number of closest matches under 30 days: ', num2str(height(filtered_matched_results))]);
% 

%% Making the data ready for ICA

loadedFncMetadata = load('updated_adni_fnc_metadata.mat');
fncMetaData = loadedFncMetadata.fnc_meta;
loadedMatchedData = load('filtered_matched_results.mat');
matchedData = loadedMatchedData.filtered_matched_results;

MaskBrainfile = '/data/users2/airaji/data/ROIsandMasks/mask_common.nii'; 
maskData = niftiread(MaskBrainfile); 

finalConcatenatedData = [];  % Initialize combined data matrix
fileNotFoundErrorCount=0;

filtered_adni_data = adni_data([], :);  % Initialize the filtered ADNI data to keep the adni_data info of only matched subsets

% csvFileName = 'fnc_dataset.csv';
% % Create feature column names
% featureNames = arrayfun(@(x) sprintf('F%d', x), 1:numFncFeatures, 'UniformOutput', false);
% header = [featureNames, {'DX'}];

% % Open CSV file and write header only once
% fid = fopen(csvFileName, 'w');
% fprintf(fid, '%s,', header{1:end-1});
% fprintf(fid, '%s\n', header{end});
% fclose(fid);

matched_age=[];
% Loop through each entry in the matched data
for i = 1:height(matchedData)
    % Extract subID and study dates, formatted correctly
    subID = char(matchedData.subID(i));
    petStudyDate = datestr(matchedData.pet_studyDate(i), 'yymmdd');
    fncStudyDate = matchedData.fnc_studyDate(i); 

    % Construct a file search pattern to find specific files
    filePattern = fullfile(pet_data_folder, ['sub-', subID, '_ses-', petStudyDate, '*_petNewComp*']);
    petFiles = dir(filePattern);
    % Check if the appropriate PET files are found
    if isempty(petFiles)
        warning(['No PET file found for subID: ', subID, ' with date: ', petStudyDate]);
        fileNotFoundErrorCount=fileNotFoundErrorCount+1;
    elseif length(petFiles) > 1
        error(['Multiple PET files found for subID: ', subID, ' with date: ', petStudyDate]);
    else
        % Only one PET file should be processed
        petFilePath = fullfile(pet_data_folder, petFiles(1).name);
        disp(['PET file path:', petFilePath])
        % Read PET data, apply the mask, and vectorize
        petDataVol = niftiread(petFilePath);
        maskedPet = petDataVol(maskData ~= 0);
        maskedPetVectorized = reshape(maskedPet, 1, []);
        % FNC data matching
        fncIndex = find((fncMetaData.subID == subID) & (fncMetaData.studyDate == fncStudyDate), 1);
        if isempty(fncIndex)
            error(['FNC data not found for subID: ', char(subID), ' on date: ', char(fncStudyDate)]);
        end

        % Extract the ADNI data row
        filtered_adni_data = [filtered_adni_data; adni_data(fncIndex, :)];  % Append the data row to the filtered data
        % Extract the FNC data row
        fncData = adni_fnc(fncIndex, :);
        % Vectorize the FNC data into a row vector
        fncMatrixVectorized = reshape(fncData, 1, []);
        
        % Extracting DX for this particular FNC
        ageLabel = adni_data.AGE_CORRECTED(fncIndex);
        matched_age = [matched_age; ageLabel];
%         % Convert DX to string if it's categorical
%         rowToWrite = [fncMatrixVectorized, string(dxLabel)];
%         writematrix(rowToWrite, csvFileName, 'WriteMode', 'append');


        disp(['Concatenating FNC and PET for sub- ', subID]);
        % Concatenate FNC and PET data horizontally
        fncPetConcatenated = [fncMatrixVectorized, maskedPetVectorized];

        % Append to the resulting matrix
        finalConcatenatedData = [finalConcatenatedData; fncPetConcatenated]; 
    end
end

% Save the filtered ADNI data to a file
save('filtered_adni_data.mat', 'filtered_adni_data');
disp('Filtered ADNI data has been saved to filtered_adni_data.mat');

disp(['No. of PET files not found in the directory: ', num2str(fileNotFoundErrorCount)])
% Save the final concatenated data to a .mat file
save('final_concatenated_data.mat', 'finalConcatenatedData');
disp('Final concatenated data saved successfully to final_concatenated_data.mat');

disp("Shapes of masked PET, and FNC..")
disp(size(maskedPetVectorized))
disp(size(fncMatrixVectorized))
disp(['Size of finalConcatenatedData: ', num2str(size(finalConcatenatedData, 1)), ' rows and ', ...
      num2str(size(finalConcatenatedData, 2)), ' columns']);

save('age_matched_records.mat', "matched_age");

%%
numPetFeatures = size(maskedPetVectorized, 2)
% numPetFeatures= 68235;
numFncFeatures = size(fncMatrixVectorized, 2)
% numFncFeatures= 5460;

%% z-score normalization of FNC and PET values of each sessions

% Extract FNC and PET components
FNC_data = finalConcatenatedData(:, 1:numFncFeatures);
PET_data = finalConcatenatedData(:, numFncFeatures+1:end);

csvFileName = 'FNC_data_before_ICA.csv';
% Save the matrix to a CSV file
writematrix(FNC_data, csvFileName);

%% Save FNC matrix to a MAT-file
save('FNC_data_filtered_subjects.mat', 'FNC_data');
save('PET_data_filtered_subjects.mat', 'PET_data');

% Initialize matrices to hold standardized data
standardized_FNC = zeros(size(FNC_data));
standardized_PET = zeros(size(PET_data));

% Standardize each subject's FNC and PET data
for i = 1:size(finalConcatenatedData, 1)  % Loop over each subject
    % Standardize FNC data for subject i
    meanFNC = mean(FNC_data(i, :));
    stdFNC = std(FNC_data(i, :));
    standardized_FNC(i, :) = (FNC_data(i, :) - meanFNC) / stdFNC;
    % Standardize PET data for subject i
    meanPET = mean(PET_data(i, :));
    stdPET = std(PET_data(i, :));
    standardized_PET(i, :) = (PET_data(i, :) - meanPET) / stdPET;
end

% Concatenate the standardized FNC and PET data horizontally
standardizedConcatenatedData = [standardized_FNC, standardized_PET];

save('standardized_concatenated_data.mat', 'standardizedConcatenatedData');
disp('Standardized concatenated data saved successfully to standarized_concatenated_data.mat');

%% Verifying if the data is standarized
load('standardized_concatenated_data.mat');
% Number of subjects
numSubjects = size(finalConcatenatedData, 1);
% numSubjects= 552;
% Prepare vectors to store the means and standard deviations
meansFNC = zeros(numSubjects, 1);
stdsFNC = zeros(numSubjects, 1);
meansPET = zeros(numSubjects, 1);
stdsPET = zeros(numSubjects, 1);

% Compute mean and standard deviation for each subject's standardized data
for i = 1:numSubjects
    meansFNC(i) = mean(standardized_FNC(i, :));
    stdsFNC(i) = std(standardized_FNC(i, :));
    meansPET(i) = mean(standardized_PET(i, :));
    stdsPET(i) = std(standardized_PET(i, :));
end

% Display or analyze the means and standard deviations
disp('Means and Standard Deviations for FNC:');
disp(table((1:numSubjects)', meansFNC, stdsFNC, 'VariableNames', {'Subject', 'Mean', 'StdDev'}));

disp('Means and Standard Deviations for PET:');
disp(table((1:numSubjects)', meansPET, stdsPET, 'VariableNames', {'Subject', 'Mean', 'StdDev'}));

%% AIC and BIC calculation for model order selection
% Perform PCA
[coeff, score, latent, tsquared, explained] = pca(standardizedConcatenatedData');

% Number of observations
n = size(standardizedConcatenatedData', 1);

AIC = zeros(length(latent), 1);
BIC = zeros(length(latent), 1);

% Loop through possible number of components k
for k = 1:length(latent)

    % Calculate AIC and BIC
    % Simplified formula from paper: https://link.springer.com/article/10.1007/s44199-021-00002-4
    AIC(k) = 2 * k + n* sum(log(latent(k+1:end)));
    BIC(k) = log(n) * k + n* sum(log(latent(k+1:end)));
end

results = table((1:length(latent))', AIC, BIC, 'VariableNames', {'no_of_PC', 'AIC', 'BIC'});

% Save the table as a CSV file
writetable(results, 'aic_bic_values.csv');

% Identify the optimal number of components by AIC and BIC
[minAIC, optimalPCA_AIC] = min(AIC);
[minBIC, optimalPCA_BIC] = min(BIC);

fprintf('Optimal number of components by AIC: %d\n', optimalPCA_AIC);
fprintf('Optimal number of components by BIC: %d\n', optimalPCA_BIC);

figure;
numComponents = length(AIC);

plot(1:numComponents, AIC, 'b-', 'LineWidth', 2);
hold on;
plot(1:numComponents, BIC, 'r-', 'LineWidth', 2);

% Mark the minimum points for AIC and BIC
plot(optimalPCA_AIC, AIC(optimalPCA_AIC), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
plot(optimalPCA_BIC, BIC(optimalPCA_BIC), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10);

xlabel('Number of Principal Components');
ylabel('Information Criterion');
title('AIC and BIC for Different Numbers of Principal Components');
legend('AIC', 'BIC', 'Min AIC', 'Min BIC', 'Location', 'best');
grid on;
hold off;

%%
% Calculate PCA
[coeff, score, latent, tsquared, explained] = pca(standardizedConcatenatedData');

% Calculate cumulative variance
cumulativeVariance = cumsum(explained);
%
elbow_value = knee_pt(double(explained), 1:size(coeff,2))
% elbow_value = elbow(1:size(coeff,2), double(explained))

captured_variance=cumulativeVariance(elbow_value)

% Calculate the number of principal components
numPCs = size(coeff, 2);

% Round explained variance and cumulative variance to four decimal places
explainedRounded = round(explained, 4);
cumulativeVarianceRounded = round(cumulativeVariance, 4)

% Create a table with the rounded PCA results
pcaResults = table((1:numPCs)', explainedRounded, cumulativeVarianceRounded, ...
    'VariableNames', {'Principal Component', 'Variance_Explained', 'Cumulative_Variance'});

% Save the table to a CSV file
writetable(pcaResults, 'PCA_Results_data_X_transpose.csv');
% disp(pcaResults)
figure;
plot(1:size(coeff,2), explained);
hold;
y_value_at_res_x = explained(elbow_value); % Get the y-value at index res_x

xLimits = xlim; % Get the current x-axis limits to span the line across the plot
line(xLimits, [y_value_at_res_x y_value_at_res_x], 'Color', 'r', 'LineStyle', '--'); 

text(xLimits(1), y_value_at_res_x, [' Captured Variance: ', num2str(captured_variance, '%.2f%%')], ...
     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'Color', 'red');
hold off;


% %% Performing ICA
% numICs =100;
% %
% num_ica_runs = 20; 
% algorithm = 'infomax';
% [A, ~, grSMs, iq, metric_Q] = myIcasso_AI(standardized_FNC, numICs, num_ica_runs, algorithm, []);
% disp('ICA completed!!!');
% 

%%
load ("IC_mdlordr_11_run_100_sign_corrected.mat", 'A', 'IC_FNC', 'IC_PET')
% load ("IC_mdlordr_100_run_20.mat", 'A', 'IC_FNC', 'IC_PET')
numICs = size(A, 2);

% %% Correcting for sign ambiguity for IC for the 10th IC (to create final weight and IC matrix)

% Extract IC_FNC and IC_PET
% IC_FNC = grSMs(:, 1:numFncFeatures);
% IC_PET = grSMs(:, numFncFeatures+1:end);

% A(:, 10) = -A(:, 10);
% IC_PET(10, :)= -IC_PET(10, :);
% IC_FNC(10, :) = -IC_FNC(10, :);
% 
% A = -A;
% IC_PET= -IC_PET;
% IC_FNC = -IC_FNC;

% disp(['Size of matrix A: ', mat2str(size(A))]);
% disp(['Size of IC matrix: ', mat2str(size(grSMs))]);


% save('IC_mdlordr_100_run_20.mat', 'IC_PET', 'IC_FNC', 'A');
% save('IC_FNC_only_mdlordr_100_run_20.mat', 'IC_FNC', 'A');


% Check sizes to ensure correctness
% disp(['Size of IC_FNC: ', mat2str(size(IC_FNC))]);
% disp(['Size of IC_PET: ', mat2str(size(IC_PET))]);

% Mapping PET part to .nii and FNC part to corelation matrix
output_dir='/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ICs/';
MaskBrainfile = '/data/users2/airaji/data/ROIsandMasks/mask_common.nii'; 

fun_write_nifti(IC_PET', MaskBrainfile, output_dir);
%%
reconstruct_and_save_fnc(IC_FNC, output_dir);


%% visualize spatial map 
thr = 1.2;
structFile = ['/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ch2bet_3x3x3.nii'];
image_values = 'positive';
display_type = 'Ortho'; % 'Ortho' ,'montage'

addpath(genpath('/data/users3/bbimali1/Toolbox/plotting_ai/'))
output_dir = '/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ICs/spatialmaps/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

for i = 1:numICs
    path = sprintf('/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ICs/ICs/IC%03d.nii', i);
    
    icatb_image_viewer_ar(path,'threshold',thr,'structFile',structFile, 'convert_to_zscores', 'yes','image_values', 'positive','display_type',display_type,'isComposite','yes');

    % Get the current figure handle
    fig_handle = gcf;

    % Specify the filename to save
    filename = sprintf('IC%03d.png', i);

    % Construct the full output path
    full_output_path = fullfile(output_dir, filename);

    % Save the figure
    saveas(fig_handle, full_output_path);
    
    
%     % Specify the filename to save
%     filename_hd = sprintf('hd_IC%03d.png', i);
%     full_output_path = fullfile(output_dir, filename_hd);
%     % Save the figure at high resolution (600 dpi)
%     print(fig_handle, full_output_path, '-dpng', '-r600');

    close(fig_handle);
end

%%
% Load the filtered ADNI data
loadedFilteredAdniData = load('filtered_adni_data.mat');
filtered_adni_data = loadedFilteredAdniData.filtered_adni_data;

%% Site Correction on ADNI Data and weight matrix
% Load site and subject ID data
sites = filtered_adni_data.SITE;
subjects = filtered_adni_data.RID;
dx = filtered_adni_data.DX;

% Identify unique sites and their corresponding indices
[uniqueSites, ~, siteIndices] = unique(sites);
% Initialize array for counting unique subjects at each site
subjCountPerSite = zeros(length(uniqueSites), 1);
uniqueDXPerSite = cell(length(uniqueSites), 1); % Store unique DX values for each site

% Calculate number of sessions per site
for i = 1:length(uniqueSites)
    siteID = uniqueSites(i);
    subjectIDsAtSite = subjects(siteIndices == i);
    dxAtSite = dx(siteIndices == i); % Get the DX values for this site

    subjCountPerSite(i) = length(unique(subjectIDsAtSite));
    dxAtSite = dx(siteIndices == i); % Get the DX values for this site
    uniqueDXPerSite{i} = unique(dxAtSite);
end

% Filter sites with at least 10 subjects
% sitesToKeep = uniqueSites(subjCountPerSite >= 10);
sitesToKeep = uniqueSites(subjCountPerSite >= 10 & cellfun(@(x) length(x) > 1, uniqueDXPerSite));


disp(['Number of unique SITES after filtering: ', num2str(length(sitesToKeep))]);
% Get the indices of filtered_adni_data that match the sitesToKeep
indicesToKeep = ismember(filtered_adni_data.SITE, sitesToKeep);

% Filter filtered_adni_data and matrix A using indicesToKeep
adni_data_final = filtered_adni_data(indicesToKeep, :);
weight_matrix_final = A(indicesToKeep, :); 

save('adni_data_final.mat', 'adni_data_final');
disp('Site corrected filtered ADNI data has been saved.');


% Also save the adjusted matrix A
save('weight_matrix_final.mat', 'weight_matrix_final');
disp('Adjusted matrix A has been saved.');

%% ADNI data analysis

% Define the directory path
analysis_outputDir = '/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/data_analysis';

if ~exist(analysis_outputDir, 'dir')
    mkdir(analysis_outputDir);
end

% Perform analyses using the customized function
[subjectAnalysisInitial, dxAnalysisInitial] = analyze_columns(adni_data);
[subjectAnalysis, dxAnalysis] = analyze_columns(filtered_adni_data);
[subjectAnalysisFinal, dxAnalysisFinal] = analyze_columns(adni_data_final);


% Save results to CSV files in the new folder
writetable(subjectAnalysisInitial, fullfile(analysis_outputDir, 'original_SubjectID_Analysis.csv'));
writetable(dxAnalysisInitial, fullfile(analysis_outputDir, 'original_DX_Analysis.csv'));

writetable(subjectAnalysis, fullfile(analysis_outputDir, 'filtered_data_SubjectID_Analysis.csv'));
writetable(dxAnalysis, fullfile(analysis_outputDir, 'filtered_data_DX_Analysis.csv'));


writetable(subjectAnalysisFinal, fullfile(analysis_outputDir, 'final_SubjectID_Analysis.csv'));
writetable(dxAnalysisFinal, fullfile(analysis_outputDir, 'final_DX_Analysis.csv'));

%% Mixed Effect Modeling

% Load data from .mat files
load('weight_matrix_final.mat'); 
load('adni_data_final.mat'); 
%%
 
% Preparing the dataset
data = table(adni_data_final.RID, adni_data_final.DX, adni_data_final.AGE_CORRECTED, adni_data_final.PTGENDER, adni_data_final.SITE,adni_data_final.PTRACCAT, adni_data_final.hm, 'VariableNames', {'RID', 'DX', 'AGE', 'GENDER', 'SITE', 'RACE', 'HEAD_MOTION'});

% Add IC weights to the table.
for i = 1:size(weight_matrix_final, 2)
    data.(['IC' num2str(i)]) = weight_matrix_final(:, i);
end

% Ensure categorical variables are treated correctly
data.DX = categorical(data.DX);
data.GENDER = categorical(data.GENDER);
data.RACE = categorical(data.RACE);

%%
csv_filename = 'adni_data_final_with_IC_weights.csv';
writetable(data, csv_filename);
disp(['Dataset saved as: ', csv_filename]);
%% Preallocate an array to store model objects
models = cell(numICs, 1);

% Loop through each IC to fit a linear mixed-effects model
for ic = 1:numICs
    formula = sprintf('IC%d ~ 1 + AGE + GENDER + DX + SITE + RACE+ HEAD_MOTION + (1|RID)', ic);
    models{ic} = fitlme(data, formula);
end

% Display results for each model
for ic = 1:numICs
    fprintf('Results for IC %d:\n', ic);
    disp(models{ic})

end

%%
format long;
% Preallocate a structure or cell array to store the results tables
results = cell(numICs, 1);
all_p_values_dementia = [];  % To store all p-values for DX Dementia across ICs
all_p_values_mci = [];       % To store all p-values for DX MCI across ICs

% Loop through each IC to fit a linear mixed-effects model
for ic = 1:numICs
    % Define the model formula
    formula = sprintf('IC%d ~ 1 + AGE + GENDER + DX + SITE + RACE+ HEAD_MOTION + (1|RID)', ic);
    
    % Fit the model
    model = fitlme(data, formula);
    
    % Extract coefficients for DX_Dementia and DX_MCI directly from the Coefficients table
    coeffs = model.Coefficients;
    idx = ismember(coeffs.Name, {'DX_Dementia', 'DX_MCI'});
    selectedCoeffs = coeffs(idx, {'Name', 'tStat', 'pValue'});
    
    % Store the selected coefficients in the results cell array
    results{ic} = selectedCoeffs;
    
    % Extract p-values for DX_Dementia and DX_MCI and store them separately
    p_dementia = selectedCoeffs.pValue(1);  % p-value for DX_Dementia
    p_mci = selectedCoeffs.pValue(2);       % p-value for DX_MCI
    
    % Append p-values to the respective arrays
    all_p_values_dementia = [all_p_values_dementia; p_dementia];
    all_p_values_mci = [all_p_values_mci; p_mci];
end

% Apply Bonferroni correction (adjust p-values)
bonferroni_p_values_dementia = all_p_values_dementia * numICs;
bonferroni_p_values_mci = all_p_values_mci * numICs;

% Display Bonferroni-corrected 
disp('Bonferroni Corrected P-values for Dementia:');
disp(bonferroni_p_values_dementia);

disp('Bonferroni Corrected P-values for MCI:');
disp(bonferroni_p_values_mci);

logFile = fopen('Mixture model_results_log_FNC_only_order_100.txt', 'w');
% Check if the file opened successfully
if logFile == -1
    error('Unable to open the file for writing.');
end


% Now, display the results with corrected p-values for each IC
for ic = 1:numICs
    fprintf('Results for IC %d:\n', ic);
    selectedCoeffs = results{ic};  % Retrieve the results for the current IC
    selectedCoeffs.Bonferroni_pValue = [bonferroni_p_values_dementia(ic); bonferroni_p_values_mci(ic)];
    % Display the results for this IC
    disp(selectedCoeffs);
    fprintf(logFile, 'Results for IC %d:\n', ic);
    
    % Convert result to a string and write it
    resultStr = evalc('disp(selectedCoeffs)');  % Capture output of disp as string
    fprintf(logFile, '%s\n', resultStr);

end

fclose(logFile);


%% Add required toolboxes to the path
addpath(genpath('/data/users4/aballem1/Toolbox/Neuromark2.3FinalPP/'))
outputPath = '/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ICs/FNCs';

%% Processing each IC

for i = 1:numICs
    filename = sprintf('/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ICs/FNCs/FNC_IC%03d.mat', i); 
    load(filename, 'full_matrix'); 

    % Obtain and order the FNC matrix in Neuromark 2.2
    fnc_sub_mat_v3_order = getOrderedMapV3(getUnorderedMapV2(full_matrix));
    % Display and save the ordered map
    showOrderedMap_modified(fnc_sub_mat_v3_order, ['/data/users3/bbimali1/fmri_pet_analysis/longitudinal_PET_FNC/ICs/FNCs/GRP_FNC_ic_' sprintf('%03d', i) '.png']);

end

%% DATA analysis
% Load data
loadedFilteredAdniData = load('filtered_adni_data.mat');
filtered_adni_data = loadedFilteredAdniData.filtered_adni_data;
data = filtered_adni_data(:, {'RID', 'DX', 'AGE_CORRECTED', 'PTGENDER'});

% Find unique RIDs and filter the data
[unique_rids, uniqueIdx] = unique(data.RID);
unique_data = data(uniqueIdx, :);
% unique_data = data;



disp(['Number of unique RIDs: ', num2str(length(unique_rids))]);
dx_values = unique(unique_data.DX);


% Display detailed count of each unique RID
[uniqueRIDs, ~, subjIndices] = unique(data.RID);  % Find unique RIDs and indices
subjectCounts = accumarray(subjIndices, 1);  % Count each RID occurrence
% disp('Table showing the count of each unique RID:')
% disp(table(uniqueRIDs, subjectCounts, 'VariableNames', {'Unique_RIDs', 'Counts'}));

dxAnalysis = tabulate(data.DX);
dxTable = array2table(dxAnalysis, 'VariableNames', {'Group', 'Number of total sessions', 'Percentage'});

dxAnalysis = tabulate(unique_data.DX);
dxTable = array2table(dxAnalysis, 'VariableNames', {'Group', 'Number of unique patients', 'Percentage'});

% Calculate gender distribution
uniqueGenders = unique(unique_data.PTGENDER);

% Initialize a count array
gender_count = zeros(size(uniqueGenders));

% Calculate the count for each gender found in unique_data
for i = 1:length(uniqueGenders)
    gender_count(i) = sum(strcmp(unique_data.PTGENDER, uniqueGenders{i}));
end

% Create a table to display the gender distribution
gender_distribution_table = table(uniqueGenders, gender_count, 'VariableNames', {'Gender', 'Count'});
disp('Gender distribution:');
disp(gender_distribution_table);


% Calculate the average AGE for each DX category
age_dx_summary = arrayfun(@(x) mean(unique_data.AGE_CORRECTED(strcmp(unique_data.DX, dx_values{x}))), 1:length(dx_values));
age_dx_table = table(dx_values, age_dx_summary', 'VariableNames', {'DX', 'Average_Age'});
disp('Average age per DX category:');
disp(age_dx_table);

% Calculate the median AGE for each DX category
age_dx_summary = arrayfun(@(x) median(unique_data.AGE_CORRECTED(strcmp(unique_data.DX, dx_values{x}))), 1:length(dx_values));
age_dx_table = table(dx_values, age_dx_summary', 'VariableNames', {'DX', 'Median_Age'});
disp('Median age per DX category:');
disp(age_dx_table);


%%
%
% Load data
loadedFilteredAdniData = load('filtered_adni_data.mat');
filtered_adni_data = loadedFilteredAdniData.filtered_adni_data;
data = filtered_adni_data(:, {'RID', 'DX', 'AGE_CORRECTED', 'PTGENDER'});

% Find unique RIDs and process data
[unique_rids, uniqueIdx] = unique(data.RID);
unique_data = data(uniqueIdx, :);
% unique_data = data;

% Diagnostics Analysis
dxAnalysis = tabulate(unique_data.DX);
dxTable = array2table(dxAnalysis(:,1:2), 'VariableNames', {'Diagnostic', 'N'});

% Gender Distribution per Diagnostic
male_count = cellfun(@(x) sum(strcmp(unique_data.PTGENDER(unique_data.DX == x), 'Male')), dxTable.Diagnostic);
female_count = cellfun(@(x) sum(strcmp(unique_data.PTGENDER(unique_data.DX == x), 'Female')), dxTable.Diagnostic);
male_age_mean = cellfun(@(x) mean(unique_data.AGE_CORRECTED(unique_data.DX == x & strcmp(unique_data.PTGENDER, 'Male'))), dxTable.Diagnostic);
female_age_mean = cellfun(@(x) mean(unique_data.AGE_CORRECTED(unique_data.DX == x & strcmp(unique_data.PTGENDER, 'Female'))), dxTable.Diagnostic);
male_age_sd = cellfun(@(x) std(unique_data.AGE_CORRECTED(unique_data.DX == x & strcmp(unique_data.PTGENDER, 'Male'))), dxTable.Diagnostic);
female_age_sd = cellfun(@(x) std(unique_data.AGE_CORRECTED(unique_data.DX == x & strcmp(unique_data.PTGENDER, 'Female'))), dxTable.Diagnostic);

% Adding gender counts and average ages to the table
dxTable.Male_N = male_count;
dxTable.Female_N = female_count;
dxTable.Male_AgeMean = male_age_mean;
dxTable.Female_AgeMean = female_age_mean;
dxTable.Male_AgeSD = male_age_sd;
dxTable.Female_AgeSD = female_age_sd;

% Display the table
disp(dxTable);

