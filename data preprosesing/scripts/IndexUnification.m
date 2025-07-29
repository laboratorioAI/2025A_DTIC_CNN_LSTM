%% Index Unification Script
%% Ahora solo Dios Sabe que esta escrito aqui

% Ruta base
indicesBasePath = '.\data\Dataset-2024-2025\Mes2\Users\idxs\';
resultadoIndices = struct(); % Estructura para consolidar datos

% Obtener todas las carpetas en la ruta base
carpetas = dir(indicesBasePath);
carpetas = carpetas([carpetas.isdir] & ~ismember({carpetas.name}, {'.', '..'})); % Filtrar carpetas válidas

% Recorrer cada carpeta, excepto matFilesUserCorrected
for i = 1:length(carpetas)
    carpetaActual = fullfile(indicesBasePath, carpetas(i).name);
    
    % Verificar si es la carpeta "matFilesUserCorrected"
    if strcmp(carpetas(i).name, 'matFilesUserCorrected')
        continue; % No procesamos esta carpeta todavía
    end
    
    archivoIndices = fullfile(carpetaActual, 'indicesTodos.mat');
    
    % Verificar si existe el archivo indicesTodos.mat
    if ~isfile(archivoIndices)
        % Buscar en subcarpeta "matFiles"
        subcarpeta = fullfile(carpetaActual, 'matFiles');
        archivoIndices = fullfile(subcarpeta, 'indicesTodos.mat');
    end
    
    % Si el archivo existe, cargarlo
    if isfile(archivoIndices)
        datos = load(archivoIndices);
        indices = datos.indices; % Extraer la estructura de índices
        
        % Recorrer los usuarios en el archivo
        campos = fieldnames(indices); % Obtener los nombres de los usuarios
        for j = 1:length(campos)
            usuario = campos{j};
            % Extraer los datos del usuario
            datosUsuario = indices.(usuario);
            
            % Añadir los datos a la estructura resultadoIndices
            if isfield(resultadoIndices, usuario)
                % Si ya existe, concatenar los datos
                for gesto = fieldnames(datosUsuario)'
                    resultadoIndices.(usuario).(gesto{1}) = [resultadoIndices.(usuario).(gesto{1}); datosUsuario.(gesto{1})];
                end
            else
                % Si no existe, agregar los datos
                resultadoIndices.(usuario) = datosUsuario;
            end
        end
    else
        fprintf('Archivo indicesTodos.mat no encontrado en: %s\n', carpetaActual);
    end
end

% Ahora procesamos la carpeta "matFilesUserCorrected" al final
carpetaCorrected = fullfile(indicesBasePath, 'matFilesUserCorrected', 'matFiles');
archivoIndicesCorrected = fullfile(carpetaCorrected, 'indicesTodos.mat');
if isfile(archivoIndicesCorrected)
    datos = load(archivoIndicesCorrected);
    indices = datos.indices; % Extraer la estructura de índices
    
    % Recorrer los usuarios en el archivo corregido
    campos = fieldnames(indices); % Obtener los nombres de los usuarios
    for j = 1:length(campos)
        usuario = campos{j};
        % Extraer los datos del usuario
        datosUsuario = indices.(usuario);
        
        % Eliminar el usuario anterior y agregar el nuevo
        resultadoIndices.(usuario) = datosUsuario;
    end
else
    fprintf('Archivo indicesTodos.mat no encontrado en: %s\n', carpetaCorrected);
end

% Guardar la estructura consolidada en un nuevo archivo
save(fullfile(indicesBasePath, 'resultadoIndices.mat'), '-struct', 'resultadoIndices');
disp('Consolidación completada. Archivo guardado como resultadoIndices.mat.');