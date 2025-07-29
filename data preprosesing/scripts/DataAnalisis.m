% Ruta base donde están almacenados los archivos .mat
basePath = '.\data\Dataset-2024-2025\Mes1\resultadosFormatoMat\';

% Obtener las carpetas de usuarios
userFolders = dir(fullfile(basePath, 'user*'));

% Archivo de salida para las observaciones
outputFile = '.\data\Dataset-2024-2025\Mes1\observaciones_revisiones.txt';
fid = fopen(outputFile, 'w');
if fid == -1
    error('No se pudo abrir el archivo para escribir observaciones.');
end

% Recorrer cada carpeta de usuario
for i = 1:length(userFolders)
    userFolder = userFolders(i).name;
    userPath = fullfile(basePath, userFolder, 'newUserData.mat');

    % Cargar el archivo .mat del usuario
    if exist(userPath, 'file')
        userData = load(userPath);
        if isfield(userData, 'newUserData')
            userStruct = userData.newUserData;
            fprintf(fid, 'Carpeta Revisada **"%s"**\n', userFolder);
            if isfield(userStruct, 'userInfo')
                userName = userStruct.userInfo.name;
                fprintf(fid, 'Nombre del usuario revisado "%s"\n', userName);

                % Verificar si el nombre del usuario corresponde a la carpeta
                if ~strcmp(userName, userFolder)
                    fprintf(fid, '\t* El nombre del usuario no coincide con la carpeta.\n');
                end
            else
                fprintf(fid, '\t* El campo userInfo no existe.\n');
                continue;
            end

            % Verificar las estructuras training y testing
            for subset = {'trainingSamples', 'testingSamples'}
                subsetName = subset{1};
                if isfield(userStruct, subsetName)
                    samples = userStruct.(subsetName);
                    numGestures = numel(fieldnames(samples));
                    fprintf(fid, '\t- Numero de gestos para %s: %d\n', subsetName, numGestures);

                    % Validar cada gesto
                    for idx = 1:numGestures
                        fieldName = sprintf('idx_%d', idx);
                        if isfield(samples, fieldName)
                            sample = samples.(fieldName);

                            % Verificar gestureName
                            if idx <= 25 && ~strcmp(char(sample.gestureName), 'noGesture')
                                fprintf(fid, '\t\t* idx_%d no tiene gestureName correcto (debería ser "noGesture").\n', idx);
                            end

                            % Verificar EMG
                            if isfield(sample, 'emg')
                                emg = sample.emg;
                                channels = fieldnames(emg);
                                if length(channels) ~= 8
                                    fprintf(fid, '\t\t* idx_%d no tiene 8 canales EMG.\n', idx);
                                else
                                    n = numel(emg.ch1);
                                    if n <= 500
                                        fprintf(fid, '\t\t* idx_%d los canales EMG tienen tamaño menor o igual a 500.\n', idx);
                                    end
                                    for ch = 2:8
                                        if numel(emg.(sprintf('ch%d', ch))) ~= n
                                            fprintf(fid, '\t\t* idx_%d los canales EMG no tienen el mismo tamaño.\n', idx);
                                            break;
                                        end
                                    end
                                end
                            else
                                fprintf(fid, '\t\t* idx_%d no tiene datos EMG.\n', idx);
                            end

                            % Verificar groundTruthIndex y groundTruth
                            if ~strcmp(char(sample.gestureName), 'noGesture')
                                if isfield(sample, 'groundTruthIndex') && isfield(sample, 'groundTruth')
                                    groundTruthIdx = sample.groundTruthIndex;
                                    groundTruth = sample.groundTruth;
                                    if isempty(groundTruthIdx) || any(groundTruthIdx <= 0)
                                        fprintf(fid, '\t\t* idx_%d groundTruthIndex no es válido.\n', idx);
                                    else
                                        startIdx = groundTruthIdx(1);
                                        endIdx = groundTruthIdx(2);
                                        if endIdx > numel(groundTruth) || startIdx > numel(groundTruth)
                                            fprintf(fid, '\t\t* idx_%d groundTruthIndex está fuera de rango.\n', idx);
                                        else
                                            rangeValues = groundTruth(startIdx:endIdx);
                                            if any(rangeValues ~= 1) || any(groundTruth([1:startIdx-1, endIdx+1:end]) ~= 0)
                                                fprintf(fid, '\t\t* idx_%d groundTruth no corresponde a los valores de su índice.\n', idx);
                                            end
                                        end
                                    end
                                else
                                    fprintf(fid, '\t\t* idx_%d no tiene groundTruthIndex o groundTruth.\n', idx);
                                end
                            end
                        else
                            fprintf(fid, '\t\t* No existe idx_%d en %s.\n', idx, subsetName);
                        end
                    end
                else
                    fprintf(fid, '\t- No existe la estructura %s.\n', subsetName);
                end
            end
        else
            fprintf(fid, 'El archivo newUserData.mat no contiene la estructura newUserData.\n');
        end
    else
        fprintf(fid, 'No se encontró el archivo newUserData.mat en la carpeta %s.\n', userFolder);
    end
end

fclose(fid);

fprintf('Revisión completada. Observaciones guardadas en %s.\n', outputFile);
