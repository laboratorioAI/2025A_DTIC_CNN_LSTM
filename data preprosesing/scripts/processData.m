% Define base paths for EMG data, indices, and output directories
emgBasePath = '.\data\Dataset-2024-2025\Mes2\Users\emgs\'; % Path to EMG data
indicesBasePath = '.\data\Dataset-2024-2025\Mes2\Users\idxs\'; % Path to indices data
outputBasePath = '.\data\Dataset-2024-2025\Mes2\OutputUsers\'; % Path to save processed output

%Resultados en math
resultBasePath = '.\data\Dataset-2024-2025\Mes2\resultadosFormatoMat\';

% Create a directory for processed data if it doesn't exist
processedDataBasePath = fullfile(outputBasePath, 'DataProcesada'); % Define path for 'DataProcesada'
if ~exist(processedDataBasePath, 'dir') % Check if directory exists
    mkdir(processedDataBasePath); % Create directory if it doesn't exist
end

% Load indices file
indicesTodos = load(fullfile(indicesBasePath, 'resultadoIndices.mat')); % Load indices file into a structure
indicesContent = indicesTodos; % Store indices for later use

% List all user folders in the EMG data directory
userFolders = dir(fullfile(emgBasePath, 'user*')); % Get a list of user directories

% Loop through each user folder to process data
for userIdx = 1:length(userFolders)
    % Get the current user's folder name
    userFolder = userFolders(userIdx).name; % Extract folder name

    % Load the user's data file
    userDataPath = fullfile(emgBasePath, userFolder, 'userData.mat'); % Path to the user's data
    userData = load(userDataPath); % Load data into a structure
    userDataContent = userData.userData; % Extract content from the loaded structure

    % Initialize a new structure for the processed user data
    newUserData = struct();

    % Copy the 'userInfo' field from the original data
    %newUserData.userInfo = userDataContent.userInfo;
    newUserData.userInfo = rmfield(userDataContent.userInfo, 'username'); 
    newUserData.userInfo.name = userDataContent.userInfo.username;

    %% Adding General info
    newUserData.generalInfo = struct( ...
    'deviceModel', 'Myo Armband', ...
    'samplingFrequencyInHertz', 200, ...
    'recordingTimeInSeconds', 5, ...
    'repetitionsForSynchronizationGesture', 5, ...
    'myoPredictionLabel', struct( ...
        'noGesture', 0, ...
        'fist', 1, ...
        'waveIn', 2, ...
        'waveOut', 3, ...
        'open', 4, ...
        'pinch', 5 ...
    ) ...
);
    %%Syncroooooo data

    if isfield(userDataContent.gestures, 'sync') && isfield(userDataContent.gestures.sync, 'data')
        %newUserData.synchronizationGesture = userDataContent.gestures.sync.data;

        

        % Crear la estructura newUserData.synchronizationGesture con el formato solicitado
        syncInstance = userDataContent.gestures.sync.data{1}; % Obtener el primer elemento de sync.data
        newUserData.synchronizationGesture.samples.idx_1 = struct();


        %Añadir el parametro "startPointforGestureExecution"
        newUserData.synchronizationGesture.samples.idx_1.startPointforGestureExecution = syncInstance.pointGestureBegins;
        
        % Renombrar y añadir gestureDevicePredicted como myoDetection
        newUserData.synchronizationGesture.samples.idx_1.myoDetection = syncInstance.gestureDevicePredicted;

        % Transformar y añadir los datos de quaternion
        quaternionData = syncInstance.quaternions;
        newUserData.synchronizationGesture.samples.idx_1.quaternion = struct(...
            'w', quaternionData(:, 1), 'x', quaternionData(:, 2), 'y', quaternionData(:, 3), 'z', quaternionData(:, 4));
        
        % Transformar y añadir los datos de EMG al formato solicitado
        emgData = syncInstance.emg;
        newUserData.synchronizationGesture.samples.idx_1.emg = struct(...
            'ch1', emgData(:, 1), 'ch2', emgData(:, 2), 'ch3', emgData(:, 3), 'ch4', emgData(:, 4), ...
            'ch5', emgData(:, 5), 'ch6', emgData(:, 6), 'ch7', emgData(:, 7), 'ch8', emgData(:, 8));
        
        
        % Transformar y añadir los datos de gyroscope
        gyroData = syncInstance.gyro;
        newUserData.synchronizationGesture.samples.idx_1.gyroscope = struct(...
            'x', gyroData(:, 1), 'y', gyroData(:, 2), 'z', gyroData(:, 3));
        
        % Transformar y añadir los datos de accelerometer
        accelData = syncInstance.accel;
        newUserData.synchronizationGesture.samples.idx_1.accelerometer = struct(...
            'x', accelData(:, 1), 'y', accelData(:, 2), 'z', accelData(:, 3));
        


        % copy of  syncro 3 times for normalization in code.
        copySync = newUserData.synchronizationGesture.samples.idx_1;
        newUserData.synchronizationGesture.samples.idx_2 =copySync;
        newUserData.synchronizationGesture.samples.idx_3 =copySync;
        newUserData.synchronizationGesture.samples.idx_4 =copySync;
        newUserData.synchronizationGesture.samples.idx_5 =copySync;

        %Comment if u want data with sync
        %the "continue" cant be commented in both lines.
        %continue;
    else
        fprintf('No sync data found for %s. Skipping sync processing.\n', userFolder);
        newUserData.synchronizationGesture = {};
        %Un-Comment for non sync data
        %Comment if u want just non sync data
        %continue;
    end

    % Initialize the 'trainingSamples' and 'testingSamples' fields
    trainingSamples = struct(); % For training data
    testingSamples = struct(); % For testing data

    gestureIndexTrain = 1; % Index for storing training gesture data
    gestureIndexTest = 1; % Index for storing testing gesture data

    gestureTypes = {"relax", "open", "fist", "waveOut", "waveIn", "pinch"}; % Define gesture types

    for g = 1:length(gestureTypes)
        gestureName = gestureTypes{g}; % Current gesture name
        if isfield(userDataContent.gestures, gestureName) % Check if the gesture exists in the data
            gestureData = userDataContent.gestures.(gestureName).data; % Get data for the gesture
            numInstances = numel(gestureData); % Count the number of instances for this gesture

            % Calculate split index for 50/50 division
            splitIndex = floor(numInstances / 2);

            for i = 1:numInstances
                instance = gestureData{i}; % Get the current instance
                gestureStruct = struct(); % Initialize a structure for this gesture instance

                % Populate fields with high priority first
                gestureStruct.startPointforGestureExecution = instance.pointGestureBegins; % Gesture start point

                % Rename "relax" to "noGesture" or keep gesture name
                if strcmp(gestureName, 'relax')
                    gestureStruct.gestureName = categorical(cellstr("noGesture"));
                else
                    gestureStruct.gestureName = categorical(cellstr(gestureName));
                end

                % Handle groundTruthIndex and groundTruth for gestures other than "relax"
                if ~strcmp(gestureName, 'relax') && isfield(indicesContent, userFolder) && isfield(indicesContent.(userFolder), gestureName)
                    gestureIndices = indicesContent.(userFolder).(gestureName); % Get indices for the gesture
                    if i <= size(gestureIndices, 1) % Ensure index exists
                        gestureStruct.groundTruthIndex = gestureIndices(i, :); % Store the ground truth index

                        % Generate groundTruth as a vector of 0s and 1s
                        groundTruthLength = size(instance.emg, 1); % Length of EMG data
                        groundTruth = zeros(groundTruthLength, 1); % Initialize ground truth as a column vector of 0s
                        startIdx = gestureIndices(i, 1); % Start index
                        endIdx = gestureIndices(i, 2); % End index
                        if startIdx <= groundTruthLength && endIdx <= groundTruthLength % Validate indices
                            groundTruth(startIdx:endIdx) = 1; % Mark 1 for the ground truth range
                        end
                        gestureStruct.groundTruth = groundTruth; % Store the ground truth
                    end
                end

                % Add remaining fields

                % Transform EMG data to structured format
                emgData = instance.emg; % EMG data
                gestureStruct.emg = struct('ch1', emgData(:, 1), 'ch2', emgData(:, 2), 'ch3', emgData(:, 3), 'ch4', emgData(:, 4), ...
                                           'ch5', emgData(:, 5), 'ch6', emgData(:, 6), 'ch7', emgData(:, 7), 'ch8', emgData(:, 8));

                gestureStruct.myoDetection = instance.gestureDevicePredicted; % Device-predicted gesture

                % Transform gyro data to structured format and rename to gyroscope
                gyroData = instance.gyro; % Gyroscope data
                gestureStruct.gyroscope = struct('x', gyroData(:, 1), 'y', gyroData(:, 2), 'z', gyroData(:, 3));

                % Transform accel data to structured format and rename to accelerometer
                accelData = instance.accel; % Accelerometer data
                gestureStruct.accelerometer = struct('x', accelData(:, 1), 'y', accelData(:, 2), 'z', accelData(:, 3));

                % Transform quaternions data to structured format and rename to quaternion
                quaternionData = instance.quaternions; % Quaternion data
                gestureStruct.quaternion = struct('w', quaternionData(:, 1), 'x', quaternionData(:, 2), 'y', quaternionData(:, 3), 'z', quaternionData(:, 4));

                % Divide gestures into trainingSamples and testingSamples
                

                if i <= splitIndex
                    trainingSamples.(sprintf('idx_%d', gestureIndexTrain)) = gestureStruct; % Add to training
                    gestureIndexTrain = gestureIndexTrain + 1;
                else
                    testingSamples.(sprintf('idx_%d', gestureIndexTest)) = gestureStruct; % Add to testing
                    gestureIndexTest = gestureIndexTest + 1;
                end
            end
        end
    end

    % Assign subsets to the new structure
    newUserData.trainingSamples = trainingSamples;
    newUserData.testingSamples = testingSamples;

    
    
    %%%% For .mat Files
    userOutputMatFolder = fullfile(resultBasePath, userFolder);

    % Crear la subcarpeta del usuario si no existe
    if ~exist(userOutputMatFolder, 'dir')
        mkdir(userOutputMatFolder);
    end
    
    % Guardar el archivo en la subcarpeta específica solo para mat
    outputFilePath = fullfile(userOutputMatFolder, 'newUserData.mat');
    save(outputFilePath, 'newUserData');



    % Create a subfolder for the current user in 'DataProcesada'
    userOutputFolder = fullfile(processedDataBasePath, userFolder); % Define user output folder
    if ~exist(userOutputFolder, 'dir') % Check if directory exists
        mkdir(userOutputFolder); % Create directory if it doesn't exist
    end

    % Save the processed data as a JSON file
    jsonFilePath = fullfile(userOutputFolder, [userFolder, '.json']); % Define the JSON file path
    fid = fopen(jsonFilePath, 'w'); % Open the file for writing
    if fid == -1 % Check if file opening failed
        error('Failed to open file for writing: %s', jsonFilePath); % Throw an error
    end
    jsonData = jsonencode(newUserData); % Convert structure to JSON format
    fprintf(fid, '%s', jsonData); % Write JSON data to file
    fclose(fid); % Close the file

    fprintf('New userData structure saved for %s to: %s\n', userFolder, jsonFilePath); % Display success message
end

fprintf('Processing complete for all users.\n'); % Indicate processing is complete