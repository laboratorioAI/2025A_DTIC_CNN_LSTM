function multiFolderEvaluation()
%{
        This script automatically traverses subfolders "Mes0", "Mes1", ...
        within "2025-Dataset" (or "EMG-EPN-612 dataset"), and in each one
        executes the logic of 'modelEvaluation' to compute individual
        (train and validation) and global metrics.
        
        At the end, in addition to printing results to the console,
        it generates plots for each user showing the evolution of 4 metrics
        (Classification, Recognition, Overlapping, ProcessingTime)
        across all months.
    
        Requires:
          - The Shared.m class in the path (with constants and utility methods).
          - The evalRecognition.m function in the path (for metric calculations).
          - Data stored in subfolders such as Mes0/trainingJSON, Mes1/trainingJSON, etc.
          - A model located at "Models\model_s30_e10.mat" (or a custom path).
%}

%% 1. List "Mes*" subfolders inside the base directory

baseDataDir = 'EMG-EPN-612 dataset';  
dirInfo     = dir(fullfile(baseDataDir, 'Mes*')); % search folders Mes0, Mes1...
subfolders  = {dirInfo.name};
numMeses    = numel(subfolders);

if numMeses == 0
    error('No se encontraron subcarpetas Mes* en "%s".', baseDataDir);
end

%% 2. Load the model (if the same model is used for all MesX)
modelFileName = "Models\model_cnn_final.mat";
loadedModel   = load(modelFileName);
model         = loadedModel.net;

%% Structure to store TRAIN and VALID results for each month

% resultsAllMeses(iMes).mesName
% resultsAllMeses(iMes).users
% resultsAllMeses(iMes).classificationTrain   (vector [numUsers x 1])
% resultsAllMeses(iMes).recognitionTrain
% resultsAllMeses(iMes).overlapTrain
% resultsAllMeses(iMes).processingTrain
% resultsAllMeses(iMes).classificationVal
% resultsAllMeses(iMes).recognitionVal
% resultsAllMeses(iMes).overlapVal
% resultsAllMeses(iMes).processingVal

resultsAllMeses = struct();

fprintf('\nIniciando Multi-Folder Evaluation...\n');

%% 3. Loop: for each "MesX" subfolder

for iMes = 1:numMeses
    mesName = subfolders{iMes};  % Ej. "Mes0", "Mes1"...
    fprintf('\n========================================\n');
    fprintf('Procesando carpeta: %s\n', mesName);
    fprintf('========================================\n');

    % Define dataDir where "trainingJSON" is located
    dataDirCurrent = fullfile(baseDataDir, mesName);
    trainingDir    = 'trainingJSON';  % Ajustar si tu carpeta se llama distinto

    % Retrieve users using Shared.getUsers
    % (Make sure the Shared class has the getUsers method, etc.)

    [users, trainingPath] = Shared.getUsers(dataDirCurrent, trainingDir);

    % Split users into train/val and test if needed
    if Shared.includeTesting
        limit        = length(users) - Shared.numTestUsers;
        usersTrainVal = users(1:limit, 1);
        usersTest     = users(limit+1:end, 1);
    else
        usersTrainVal = users;
    end

    % PREALLOCATE SPACE FOR RESULTS (TRAIN & VALIDATION)
    [classifications, recognitions, overlapings, procesingTimes] = preallocateResults(length(usersTrainVal));
    [classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal] = preallocateResults(length(usersTrainVal));

    %% EVALUATE EACH USER (TRAIN/VAL)

    for iUser = 1:length(usersTrainVal)
        % Obtain training/validation samples

        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTrainVal(iUser));

        % --- TRAIN ---
        transformedSamplesTraining = transformSamples(trainingSamples);
        userResultsTrain           = evaluateSamples(transformedSamplesTraining, model);

        % Save training results in row iUser
        [classifications(iUser, :), recognitions(iUser, :), ...
            overlapings(iUser, :), procesingTimes(iUser, :)] = ...
            deal(userResultsTrain.classifications, ...
            userResultsTrain.recognitions, ...
            userResultsTrain.overlapings, ...
            userResultsTrain.procesingTimes);

        % --- VALIDATION ---
        transformedSamplesValidation = transformSamples(validationSamples);
        userResultsVal               = evaluateSamples(transformedSamplesValidation, model);

        [classificationsVal(iUser, :), recognitionsVal(iUser, :), ...
            overlapingsVal(iUser, :), procesingTimesVal(iUser, :)] = ...
            deal(userResultsVal.classifications, ...
            userResultsVal.recognitions, ...
            userResultsVal.overlapings, ...
            userResultsVal.procesingTimes);
    end

    % Print TRAIN results to console
    fprintf('\n\n\tTraining data results (%s)\n\n', mesName);
    resultsTrain = calculateValidationResults(classifications, recognitions, ...
        overlapings, procesingTimes, length(usersTrainVal));

    % Print VALIDATION results to console
    fprintf('\n\n\tValidation data results (%s)\n\n', mesName);
    resultsVal = calculateValidationResults(classificationsVal, recognitionsVal, ...
        overlapingsVal, procesingTimesVal, length(usersTrainVal));

    % Save per-user results in the structure
    % (calculated using the same formula as calculateMeanUsers)

    % ================================================================
    % SAVE PER-USER RESULTS IN STRUCTURE (TRAIN, VAL, AVG)
    % ================================================================

    [classPerUserTrain, recogPerUserTrain, overlapPerUserTrain, pTimePerUserTrain] = ...
        calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, length(usersTrainVal));

    [classPerUserVal, recogPerUserVal, overlapPerUserVal, pTimePerUserVal] = ...
        calculateMeanUsers(classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal, length(usersTrainVal));

    % ---------- AVERAGE CALCULATION (TRAIN + VAL) ----------
    classPerUserAvg   = (classPerUserTrain + classPerUserVal) / 2;
    recogPerUserAvg   = (recogPerUserTrain + recogPerUserVal) / 2;
    overlapPerUserAvg = (overlapPerUserTrain + overlapPerUserVal) / 2;
    pTimePerUserAvg   = (pTimePerUserTrain + pTimePerUserVal) / 2;
    % -------------------------------------------------------

    resultsAllMeses(iMes).mesName   = mesName;
    resultsAllMeses(iMes).users     = usersTrainVal;

    % --- TRAIN ---
    resultsAllMeses(iMes).classificationTrain = classPerUserTrain;
    resultsAllMeses(iMes).recognitionTrain    = recogPerUserTrain;
    resultsAllMeses(iMes).overlapTrain        = overlapPerUserTrain;
    resultsAllMeses(iMes).processingTrain     = pTimePerUserTrain;

    % --- VALIDATION ---
    resultsAllMeses(iMes).classificationVal = classPerUserVal;
    resultsAllMeses(iMes).recognitionVal    = recogPerUserVal;
    resultsAllMeses(iMes).overlapVal        = overlapPerUserVal;
    resultsAllMeses(iMes).processingVal     = pTimePerUserVal;

    % --- NEW: AVG (TRAIN + VAL) ---
    resultsAllMeses(iMes).classificationAvg = classPerUserAvg;
    resultsAllMeses(iMes).recognitionAvg    = recogPerUserAvg;
    resultsAllMeses(iMes).overlapAvg        = overlapPerUserAvg;
    resultsAllMeses(iMes).processingAvg     = pTimePerUserAvg;



end

%% 4. PLOT ONLY AVERAGE (AVG) RESULTS PER USER vs. MONTH
% Custom color list (add more if needed)
colors = [
    0.85 0 0;    % red
    0 0.85 0;    % green
    0 0 0.85;    % blue
    0.85 0.85 0; % yellow
];

if ~isempty(resultsAllMeses)
    allUsers = resultsAllMeses(1).users;  % We asume the same order in each
    numUsers = length(allUsers);
    numMeses = length(resultsAllMeses);
    xVals    = 1:numMeses;  % Eje X

    for iUser = 1:numUsers
        %% --- CHANGE: plot only AVG ---
        % Reserve AVG arrays

        classificationAvgArray = zeros(1, numMeses);
        recognitionAvgArray    = zeros(1, numMeses);
        overlapAvgArray        = zeros(1, numMeses);
        processingAvgArray     = zeros(1, numMeses);

        % Fill arrays
        for iMes = 1:numMeses
            classificationAvgArray(iMes) = resultsAllMeses(iMes).classificationAvg(iUser);
            recognitionAvgArray(iMes)    = resultsAllMeses(iMes).recognitionAvg(iUser);
            overlapAvgArray(iMes)        = resultsAllMeses(iMes).overlapAvg(iUser);
            processingAvgArray(iMes)     = resultsAllMeses(iMes).processingAvg(iUser);
        end

        % Plot figure (AVG only)
        figure('Name', sprintf('Metrics (AVG) - %s', string(allUsers(iUser))), ...
            'NumberTitle','off');

        subplot(2,2,1);
        plot(xVals, classificationAvgArray,'-s','LineWidth',1.6,'Color', colors(1,:));
        grid on; xlabel('Mes'); ylabel('Classification'); title('Accuracy (AVG)');

        subplot(2,2,2);
        plot(xVals, recognitionAvgArray,'-s','LineWidth',1.6,'Color', colors(2,:));
        grid on; xlabel('Mes'); ylabel('Recognition'); title('Recognition (AVG)');

        subplot(2,2,3);
        plot(xVals, overlapAvgArray,'-s','LineWidth',1.6,'Color', colors(3,:));
        grid on; xlabel('Mes'); ylabel('Overlap'); title('Overlapping Factor (AVG)');

        subplot(2,2,4);
        plot(xVals, processingAvgArray,'-s','LineWidth',1.6,'Color', colors(4,:));
        grid on; xlabel('Mes'); ylabel('Time (s)'); title('Processing Time (AVG)');

        sgtitle(sprintf('Evolution of performance metrics (AVG) - User: %s', string(allUsers(iUser))));
    end
end

%% ========================================================================
% 5. SUMMARY TABLES (AVG) ──► Accuracy-Classification, Accuracy-Recognition,
%                             Overlapping, and Processing-Time
% ========================================================================


% ← Month names (column headers)
mesNames   = string({resultsAllMeses.mesName});        % Ej.: ["Mes1","Mes2","Mes3"]
% ← User names (row headers)
userNames  = string(resultsAllMeses(1).users);         % We asume the same order in each

numUsers   = numel(userNames);
numMeses   = numel(mesNames);

% ---------- Preallocate matrices ----------
classificationMat = zeros(numUsers, numMeses);
recognitionMat    = zeros(numUsers, numMeses);
overlapMat        = zeros(numUsers, numMeses);
processingMat     = zeros(numUsers, numMeses);

% ---------- Fill each metric ----------
for iMes = 1:numMeses
    classificationMat(:, iMes) = resultsAllMeses(iMes).classificationAvg;
    recognitionMat(:,    iMes) = resultsAllMeses(iMes).recognitionAvg;
    overlapMat(:,        iMes) = resultsAllMeses(iMes).overlapAvg;
    processingMat(:,     iMes) = resultsAllMeses(iMes).processingAvg;
end

% ---------- Create table objects ----------
Tclassification = array2table(classificationMat, 'VariableNames', mesNames, 'RowNames', userNames);

Trecognition    = array2table(recognitionMat,    'VariableNames', mesNames, 'RowNames', userNames);
Toverlap        = array2table(overlapMat,        'VariableNames', mesNames, 'RowNames', userNames);
Tprocessing     = array2table(processingMat,     'VariableNames', mesNames, 'RowNames', userNames);

% ---------- Display on console ----------
disp('===== Promedios por usuario y mes (Accuracy – Classification) =====');
% ======= Accuracy – Classification (AVG) =======
disp('===== Promedios por usuario y mes (Accuracy – Classification) =====');
disp(Tclassification);

figure('Name','Accuracy – Classification (AVG)','NumberTitle','off');
uitable( ...
    'Data',           Tclassification{:,:}, ...
    'ColumnName',     Tclassification.Properties.VariableNames, ...
    'RowName',        Tclassification.Properties.RowNames, ...
    'Units',          'normalized', ...
    'Position',       [0 0 1 1], ...
    'ColumnWidth',    'auto');

% ======= Accuracy – Recognition (AVG) =======
disp('===== Promedios por usuario y mes (Accuracy – Recognition) =====');
disp(Trecognition);

figure('Name','Accuracy – Recognition (AVG)','NumberTitle','off');
uitable( ...
    'Data',           Trecognition{:,:}, ...
    'ColumnName',     Trecognition.Properties.VariableNames, ...
    'RowName',        Trecognition.Properties.RowNames, ...
    'Units',          'normalized', ...
    'Position',       [0 0 1 1], ...
    'ColumnWidth',    'auto');

% ======= Overlapping (AVG) =======
disp('===== Promedios por usuario y mes (Overlapping) =====');
disp(Toverlap);

figure('Name','Overlapping (AVG)','NumberTitle','off');
uitable( ...
    'Data',           Toverlap{:,:}, ...
    'ColumnName',     Toverlap.Properties.VariableNames, ...
    'RowName',        Toverlap.Properties.RowNames, ...
    'Units',          'normalized', ...
    'Position',       [0 0 1 1], ...
    'ColumnWidth',    'auto');

% ======= Processing Time (AVG) =======
disp('===== Promedios por usuario y mes (Processing Time – s) =====');
disp(Tprocessing);

figure('Name','Processing Time (AVG)','NumberTitle','off');
uitable( ...
    'Data',           Tprocessing{:,:}, ...
    'ColumnName',     Tprocessing.Properties.VariableNames, ...
    'RowName',        Tprocessing.Properties.RowNames, ...asdd
    'Units',          'normalized', ...
    'Position',       [0 0 1 1], ...
    'ColumnWidth',    'auto');
% ---------- (Optional) Exportar to Excel ----------
exportFile = fullfile(pwd, 'Resumen-Metricas-AVG-CNN.xlsx');
writetable(Tclassification, exportFile, 'Sheet', 'Accuracy_Classification', 'WriteRowNames', true);
writetable(Trecognition,    exportFile, 'Sheet', 'Accuracy_Recognition',   'WriteRowNames', true);
writetable(Toverlap,        exportFile, 'Sheet', 'Overlapping',            'WriteRowNames', true);
writetable(Tprocessing,     exportFile, 'Sheet', 'Processing_Time',        'WriteRowNames', true);
fprintf('\nTablas exportadas a %s\n', exportFile);


%% 6. PLOT FINAL AVERAGE (AVG) OF ALL METRICS
% Compute average metrics (across all users and months)

classificationAvgFinal = zeros(1, numMeses);
recognitionAvgFinal    = zeros(1, numMeses);
overlapAvgFinal        = zeros(1, numMeses);
processingAvgFinal     = zeros(1, numMeses);

for iMes = 1:numMeses
    classificationAvgFinal(iMes) = mean(resultsAllMeses(iMes).classificationAvg);
    recognitionAvgFinal(iMes)    = mean(resultsAllMeses(iMes).recognitionAvg);
    overlapAvgFinal(iMes)        = mean(resultsAllMeses(iMes).overlapAvg);
    processingAvgFinal(iMes)     = mean(resultsAllMeses(iMes).processingAvg);
end

% Create figure to display final averages

figure('Name', 'Evolution of average performance metrics across all users', 'NumberTitle', 'off');

% Colors for metrics
colors = [
    0.85 0 0;    % red
    0 0.85 0;    % green
    0 0 0.85;    % blue
    0.85 0.85 0; % yellow
];

% Subplot for 'Classification' (Average)

subplot(2,2,1);
plot(1:numMeses, classificationAvgFinal, '-s', 'LineWidth', 1.6, 'Color', colors(1,:));
grid on; xlabel('Mes'); ylabel('Classification'); title('Promedio de Accuracy (AVG)');

% Subplot for 'Recognition' (Average)

subplot(2,2,2);
plot(1:numMeses, recognitionAvgFinal, '-s', 'LineWidth', 1.6, 'Color', colors(2,:));
grid on; xlabel('Mes'); ylabel('Recognition'); title('Promedio de Recognition (AVG)');

% Subplot for 'Overlap' (Average)

subplot(2,2,3);
plot(1:numMeses, overlapAvgFinal, '-s', 'LineWidth', 1.6, 'Color', colors(3,:));
grid on; xlabel('Mes'); ylabel('Overlap'); title('Promedio de Overlapping (AVG)');

% Subplot for 'Processing Time' (Average)
subplot(2,2,4);
plot(1:numMeses, processingAvgFinal, '-s', 'LineWidth', 1.6, 'Color', colors(4,:));
grid on; xlabel('Mes'); ylabel('Time (s)'); title('Promedio de Processing Time (AVG)');

sgtitle('Promedio Final de Métricas (AVG) por Mes');


fprintf('\n\nEvaluación completa. Se han generado gráficas con la evolución del PROMEDIO de las 4 métricas por usuario.\n');
end  % END of multiFolderEvaluation




%% ========================================================================
%  OTHER FUINCTIONS 
%% ========================================================================

function [clasifications, recognitions, overlapings, procesingTimes] = preallocateResults(numUsers)
% Matrices for store the results (numUsers x Shared.numSamplesUser).
clasifications = zeros(numUsers, Shared.numSamplesUser);
recognitions   = zeros(numUsers, Shared.numSamplesUser);
overlapings    = zeros(numUsers, Shared.numSamplesUser);
procesingTimes = zeros(numUsers, Shared.numSamplesUser);
end

function transformedSamples = transformSamples(samples)
samplesKeys = fieldnames(samples);
transformedSamples = cell(length(samplesKeys), 3);

for i = 1:length(samplesKeys)
    sample = samples.(samplesKeys{i});
    emg         = sample.emg;
    gestureName = sample.gestureName;
    signal      = Shared.getSignal(emg);

    transformedSamples{i,1} = signal;
    transformedSamples{i,2} = gestureName;
    if ~isequal(gestureName,'noGesture')
        groundTruth = sample.groundTruth;
        transformedSamples{i,3} = transpose(groundTruth);
    end
end
end

function [clasifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(numObservations)
% vector for each user sample
clasifications = zeros(numObservations, 1);
recognitions   = -1*ones(numObservations, 1);
overlapings    = -1*ones(numObservations, 1);
procesingTimes = zeros(numObservations, 1);
end

function userResults = evaluateSamples(samples, model)
[classifications, recognitions, overlapings, procesingTimes] = ...
    preallocateUserResults(length(samples));

for i = 1:length(samples)
    emg         = samples{i, 1};
    gesture     = samples{i, 2};
    groundTruth = samples{i, 3};

    if ~isequal(gesture,'noGesture')
        repInfo.groundTruth = logical(groundTruth);
    end
    repInfo.gestureName = categorical({gesture}, Shared.setNoGestureUse(true));

    [labels, timestamps, processingTimesFrames] = evaluateSampleFrames(emg, groundTruth, model);

    class  = Shared.classifyPredictions(labels);
    labels = Shared.postprocessSample(labels, char(class));

    response = struct('vectorOfLabels', labels, ...
        'vectorOfTimePoints', timestamps, ...
        'vectorOfProcessingTimes', processingTimesFrames, ...
        'class', class);

    result = evalRecognition(repInfo, response);

    classifications(i) = result.classResult;
    if ~isequal(gesture,'noGesture')
        recognitions(i) = result.recogResult;
        overlapings(i)  = result.overlappingFactor;
    end
    procesingTimes(i) = mean(processingTimesFrames);
end

userResults = struct('classifications', classifications, ...
    'recognitions',   recognitions, ...
    'overlapings',    overlapings, ...
    'procesingTimes', procesingTimes);
end

function [labels, timestamps, processingTimes] = evaluateSampleFrames(signal, groundTruth, model)
numPoints = length(signal);

if isequal(Shared.FILLING_TYPE_EVAL, 'before')
    numWindows = floor((numPoints - (Shared.FRAME_WINDOW / 2)) / Shared.WINDOW_STEP_RECOG) + 1;
    stepLimit  = numPoints - floor(Shared.FRAME_WINDOW / 2) + 1;
elseif isequal(Shared.FILLING_TYPE_EVAL, 'none')
    numWindows = floor((numPoints - Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_RECOG) + 1;
    stepLimit  = numPoints - Shared.FRAME_WINDOW + 1;
end

labels          = cell(1, numWindows);
timestamps      = zeros(1, numWindows);
processingTimes = zeros(1, numWindows);

if isequal(Shared.FILLING_TYPE_EVAL, 'before')
    if groundTruth
        noGestureInSignal = signal(~groundTruth, :);
        filling = noGestureInSignal(1: floor(Shared.FRAME_WINDOW / 2), :);
    else
        filling = signal(1: floor(Shared.FRAME_WINDOW / 2), :);
    end
    signal = [signal; filling];
end

idx = 1; inicio = 1;
while inicio <= stepLimit
    t0 = tic;
    finish    = inicio + Shared.FRAME_WINDOW - 1;
    timestamp = inicio + floor((Shared.FRAME_WINDOW - 1)/2);

    frameSignal = signal(inicio:finish, :);
    frameSignal = Shared.preprocessSignal(frameSignal);

    spectrograms = Shared.generateSpectrograms(frameSignal);
    [predicction, predictedScores] = classify(model, spectrograms);

    if max(predictedScores) < Shared.FRAME_CLASS_THRESHOLD
        predicction = 'noGesture';
    else
        predicction = char(predicction);
    end

    labels{idx}     = predicction;
    timestamps(idx) = timestamp;
    processingTimes(idx) = toc(t0);

    idx    = idx + 1;
    inicio = inicio + Shared.WINDOW_STEP_RECOG;
end
end

function results = calculateValidationResults(classifications, recognitions, overlapings, procesingTimes, numUsers)
[classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ...
    calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers);

% --- print each user result ---
disp('====== Individual results per user ======');
for iUser = 1:numUsers
    fprintf('User %d:\n', iUser);
    fprintf('\tClassification Accuracy: %.4f\n', classificationPerUser(iUser));
    fprintf('\tRecognition Accuracy:    %.4f\n', recognitionPerUser(iUser));
    fprintf('\tOverlapping Factor:      %.4f\n', overlapingPerUser(iUser));
    fprintf('\tProcessing Time (avg):   %.4f\n\n', processingTimePerUser(iUser));
end

% --- print stadistics ---
disp('Results (mean of user results)');
fprintf('Classification | acc: %f | std: %f  \n', ...
    mean(classificationPerUser), std(classificationPerUser));
fprintf('Recognition | acc: %f | std: %f  \n', ...
    mean(recognitionPerUser), std(recognitionPerUser));
fprintf('Overlaping | avg: %f | std: %f  \n', ...
    mean(overlapingPerUser), std(overlapingPerUser));
fprintf('Processing time | avg: %f | std: %f  \n\n', ...
    mean(processingTimePerUser), std(processingTimePerUser));

% Flatten samples
[classifications, recognitions, overlapings, procesingTimes] = ...
    deal(classifications(:), recognitions(:), overlapings(:), procesingTimes(:));

all = struct('classifications', classifications, 'recognitions', recognitions, ...
    'overlapings', overlapings, 'procesingTimes', procesingTimes);
perUser = struct('classifications', classificationPerUser, 'recognitions', recognitionPerUser, ...
    'overlapings', overlapingPerUser, 'procesingTimes', processingTimePerUser);

[globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers);

disp('Results (Global results)');
fprintf('Classification | acc: %f | std: %f  \n', ...
    globalResps.accClasification, globalStds.stdClassification);
fprintf('Recognition | acc: %f | std: %f  \n', ...
    globalResps.accRecognition, globalStds.stdRecognition);
fprintf('Overlaping | avg: %f | std: %f  \n', ...
    globalResps.avgOverlapingFactor, globalStds.stdOverlaping);
fprintf('Processing time | avg: %f | std: %f  \n\n', ...
    globalResps.avgProcesingTime, globalStds.stdProcessingTime);

results = struct('clasification',  globalResps.accClasification, ...
    'recognition',    globalResps.accRecognition, ...
    'overlapingFactor', globalResps.avgOverlapingFactor, ...
    'procesingTime',  globalResps.avgProcesingTime);
end

function [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ...
    calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers)

[classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ...
    deal(zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1));

for i = 1:numUsers
    classificationPerUser(i) = sum(classifications(i, :) == 1) / length(classifications(i, :));
    recognitionPerUser(i)    = sum(recognitions(i, :) == 1) / sum(recognitions(i, :) == 1 | recognitions(i, :) == 0);
    overlapingsUser          = overlapings(i, :);
    overlapingPerUser(i)     = mean(overlapingsUser(overlapingsUser ~= -1), 'omitnan');
    processingTimePerUser(i) = mean(procesingTimes(i, :));
end
end

function [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers)
accClasification     = sum(all.classifications==1) / numel(all.classifications);
accRecognition       = sum(all.recognitions==1) / sum(all.recognitions==1 | all.recognitions==0);
avgOverlapingFactor = mean(all.overlapings(all.overlapings ~= -1), 'omitnan');
avgProcesingTime    = mean(all.procesingTimes);

globalResps = struct('accClasification',     accClasification, ...
    'accRecognition',       accRecognition, ...
    'avgOverlapingFactor',  avgOverlapingFactor, ...
    'avgProcesingTime',     avgProcesingTime);

classificationPerUser = perUser.classifications;
recognitionPerUser    = perUser.recognitions;
overlapingPerUser     = perUser.overlapings;
processingTimePerUser = perUser.procesingTimes;

[stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);

for i = 1:numUsers
    stdClassification = stdClassification + (classificationPerUser(i) - accClasification)^2;
    stdRecognition    = stdRecognition    + (recognitionPerUser(i) - accRecognition)^2;
    stdOverlaping     = stdOverlaping     + (overlapingPerUser(i) - avgOverlapingFactor)^2;
    stdProcessingTime = stdProcessingTime + (processingTimePerUser(i) - avgProcesingTime)^2;
end

if numUsers > 1
    stdClassification = stdClassification / (numUsers - 1);
    stdRecognition    = stdRecognition    / (numUsers - 1);
    stdOverlaping     = stdOverlaping     / (numUsers - 1);
    stdProcessingTime = stdProcessingTime / (numUsers - 1);
end

globalStds = struct('stdClassification', stdClassification, ...
    'stdRecognition',    stdRecognition, ...
    'stdOverlaping',     stdOverlaping, ...
    'stdProcessingTime', stdProcessingTime);
end
