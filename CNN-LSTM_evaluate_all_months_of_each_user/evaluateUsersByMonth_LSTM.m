function multiFolderEvaluation_LSTM()
%====================================================================
% Evaluate one CNN-LSTM model on EVERY  Mes*/trainingJSON folder.
% Prints per–month metrics (train & validation) and generates, for
% every user, two figures (TRAIN / VALID) showing the evolution of:
%   · Classification accuracy
%   · Recognition accuracy
%   · Overlap factor
%   · Avg. processing time per frame
%
% Folder layout expected:
%   EMG-EPN-612 dataset/
%       Mes1/trainingJSON/userX/*.json
%       Mes2/trainingJSON/...
%
% Requirements on the MATLAB path:
%   – Shared.m
%   – evalRecognition.m
%   – Trained LSTM model (.mat)
%====================================================================

%% ----------- CONFIG ------------------------------------------------
% (Set paths and model filename here)

baseDataDir = 'EMG-EPN-612 dataset';
modelFile   = 'ModelsLSTM\model_cnn-lstm_final.mat';
trainingDir = 'trainingJSON';
%--------------------------------------------------------------------

%% ----------- Discover Mes* folders --------------------------------
monthsInfo = dir(fullfile(baseDataDir,'Mes*'));
months     = {monthsInfo([monthsInfo.isdir]).name};
assert(~isempty(months), ...
       'No Mes* folders found inside "%s"', baseDataDir);

fprintf('\n=== CNN-LSTM multi-month evaluation  (%d months) ===\n', numel(months));

%% ----------- Load network once ------------------------------------
net = load(modelFile).net;

%% ----------- Results container ------------------------------------
resultsAll = struct;

for k = 1:numel(months)
    monthName = months{k};
    dataDir   = fullfile(baseDataDir, monthName);

    fprintf('\n========================================\n');
    fprintf('Processing  %s …\n', monthName);
    fprintf('========================================\n');

    % ---- user list for this month ----
    [users, trainPath] = Shared.getUsers(dataDir, trainingDir);
    nUsers             = numel(users);

    % ---- reserve result matrices ----
    [clsTR, recTR, ovlTR, ptTR] = preallocateResults(nUsers);
    [clsVL, recVL, ovlVL, ptVL] = preallocateResults(nUsers);

    % ---- evaluate every user ----
    for u = 1:nUsers
        [trainSamp,valSamp] = ...
            Shared.getTrainingTestingSamples(trainPath, users(u));

        resTrain = evaluateSamples_LSTM(transformSamples(trainSamp), net);
        resVal   = evaluateSamples_LSTM(transformSamples(valSamp),   net);

        [clsTR(u,:), recTR(u,:), ovlTR(u,:), ptTR(u,:)] = ...
            deal(resTrain.classifications, resTrain.recognitions, ...
                 resTrain.overlapings,      resTrain.procesingTimes);

        [clsVL(u,:), recVL(u,:), ovlVL(u,:), ptVL(u,:)] = ...
            deal(resVal.classifications,  resVal.recognitions, ...
                 resVal.overlapings,       resVal.procesingTimes);
    end

    % ---- print month metrics ----
    fprintf('\n--- %s  TRAIN ---\n', monthName);
    calculateValidationResults(clsTR, recTR, ovlTR, ptTR, nUsers);
    fprintf('\n--- %s  VALID ---\n', monthName);
    calculateValidationResults(clsVL, recVL, ovlVL, ptVL, nUsers);

    % ---- store per-user means ----
    [cUtr,rUtr,oUtr,tUtr] = calculateMeanUsers(clsTR, recTR, ovlTR, ptTR, nUsers);
    [cUvl,rUvl,oUvl,tUvl] = calculateMeanUsers(clsVL, recVL, ovlVL, ptVL, nUsers);

    resultsAll(k).month             = monthName;
    resultsAll(k).users             = users;
    resultsAll(k).classificationTR  = cUtr;
    resultsAll(k).recognitionTR     = rUtr;
    resultsAll(k).overlapTR         = oUtr;
    resultsAll(k).ptimeTR           = tUtr;
    resultsAll(k).classificationVAL = cUvl;
    resultsAll(k).recognitionVAL    = rUvl;
    resultsAll(k).overlapVAL        = oUvl;
    resultsAll(k).ptimeVAL          = tUvl;
end

%% ----------- Plot evolution per user ------------------------------
plotEvolutionPerUser(resultsAll);

%% ========== TABLAS RESUMEN (AVG Train+Val) ==========
% (Summary tables for each metric: average of train and validation)

mesNames  = string({resultsAll.month});
userNames = string(resultsAll(1).users);   % We assume the same order
nU = numel(userNames);  nM = numel(mesNames);

% Pre-alcate
clsMat = zeros(nU,nM); recMat = clsMat; ovlMat = clsMat; ptMat = clsMat;

for m = 1:nM
    clsMat(:,m) = (resultsAll(m).classificationTR + resultsAll(m).classificationVAL)/2;
    recMat(:,m) = (resultsAll(m).recognitionTR  + resultsAll(m).recognitionVAL )/2;
    ovlMat(:,m) = (resultsAll(m).overlapTR       + resultsAll(m).overlapVAL    )/2;
    ptMat(:,m)  = (resultsAll(m).ptimeTR         + resultsAll(m).ptimeVAL      )/2;
end

Tcls = array2table(clsMat,'VariableNames',mesNames,'RowNames',userNames);
Trec = array2table(recMat,'VariableNames',mesNames,'RowNames',userNames);
Tovl = array2table(ovlMat,'VariableNames',mesNames,'RowNames',userNames);
Tpt  = array2table(ptMat, 'VariableNames',mesNames,'RowNames',userNames);

disp('--- Accuracy-Classification (AVG) ---'); disp(Tcls);
disp('--- Accuracy-Recognition  (AVG) ---'); disp(Trec);
disp('--- Overlapping           (AVG) ---'); disp(Tovl);
disp('--- Processing Time       (AVG) ---'); disp(Tpt);

%% ========== EXPORT TO EXCEL (AVG Train+Val) ==========
% (Export summary tables to Excel file)
outFile = fullfile("CNN-LSTM",'Resultados_CNN-LSTM.xlsx');

opts = {'WriteVariableNames',true,'WriteRowNames',true};

writetable(Tcls,outFile,'Sheet','Classification',opts{:});
writetable(Trec,outFile,'Sheet','Recognition',  opts{:});
writetable(Tovl,outFile,'Sheet','Overlap',      opts{:});
writetable(Tpt ,outFile,'Sheet','ProcTime',    opts{:});

fprintf('Tablas exportadas a %s\n',outFile);


%% ========== GLOBAL PLOT(AVG ALL USERS) ==========
clsG = mean(clsMat,1);  recG = mean(recMat,1);
ovlG = mean(ovlMat,1);  ptG  = mean(ptMat ,1);

figure('Name','Promedio global por mes – CNN-LSTM','NumberTitle','off');
subplot(2,2,1); plot(1:nM,clsG ,'-s','LineWidth',1.6); grid on; title('Classification');
subplot(2,2,2); plot(1:nM,recG ,'-s','LineWidth',1.6); grid on; title('Recognition');
subplot(2,2,3); plot(1:nM,ovlG ,'-s','LineWidth',1.6); grid on; title('Overlap');
subplot(2,2,4); plot(1:nM,ptG  ,'-s','LineWidth',1.6); grid on; title('Proc. Time');
sgtitle('Evolución global – AVG (Train + Val)');






fprintf('\nEvaluation finished – evolution figures created.\n');
end  % =========================  end main  ============================



%====================================================================
%                          Helper functions
%====================================================================
function [M1,M2,M3,M4] = preallocateResults(n)
M1 = zeros(n, Shared.numSamplesUser);
M2 = zeros(n, Shared.numSamplesUser);
M3 = zeros(n, Shared.numSamplesUser);
M4 = zeros(n, Shared.numSamplesUser);
end

function tf = transformSamples(samples)
names = fieldnames(samples);
tf    = cell(numel(names),3);
for i = 1:numel(names)
    s        = samples.(names{i});
    tf{i,1}  = Shared.getSignal(s.emg);
    tf{i,2}  = s.gestureName;
    if ~isequal(s.gestureName,'noGesture')
        tf{i,3} = s.groundTruth.';
    end
end
end

function userRes = evaluateSamples_LSTM(samples, net)
nObs = size(samples,1);
[cls,rec,ovl,pt] = preallocateUserResults(nObs);

for i = 1:nObs
    emg   = samples{i,1};
    label = samples{i,2};
    gt    = samples{i,3};

    if ~isequal(label,'noGesture')
        repInfo.groundTruth = logical(gt);
    end
    repInfo.gestureName = categorical({label}, Shared.setNoGestureUse(true));

    [labs,ts,pT] = evaluateSampleFrames_LSTM(emg, gt, net);
    class  = Shared.classifyPredictions(labs);
    labs   = Shared.postprocessSample(labs, char(class));

    resp = struct('vectorOfLabels',labs,'vectorOfTimePoints',ts,...
                  'vectorOfProcessingTimes',pT,'class',class);
    r    = evalRecognition(repInfo, resp);

    cls(i) = r.classResult;
    if ~isequal(label,'noGesture')
        rec(i) = r.recogResult;
        ovl(i) = r.overlappingFactor;
    end
    pt(i) = mean(pT);
end

userRes = struct('classifications',cls,'recognitions',rec,...
                 'overlapings',ovl,'procesingTimes',pt);
end

function [cls,rec,ovl,pt] = preallocateUserResults(n)
cls = zeros(n,1);  rec = -1*ones(n,1);  ovl = -1*ones(n,1);  pt = zeros(n,1);
end

function [labs, stamps, pT] = evaluateSampleFrames_LSTM(signal, gt, net)
if isequal(Shared.FILLING_TYPE_EVAL,'before')
    signal = [signal; signal(1:floor(Shared.FRAME_WINDOW/2),:)];
end
nWin = floor((size(signal,1)-Shared.FRAME_WINDOW)/Shared.WINDOW_STEP_LSTM)+1;
labs = cell(1,nWin); stamps=zeros(1,nWin); pT=zeros(1,nWin);

net = resetState(net);
for w = 1:nWin
    t0 = tic;
    off   = (w-1)*Shared.WINDOW_STEP_LSTM;
    sIdx  = 1+off; eIdx = Shared.FRAME_WINDOW+off;
    stamps(w) = sIdx + floor(Shared.FRAME_WINDOW/2);

    frame  = Shared.preprocessSignal(signal(sIdx:eIdx,:));
    spec   = Shared.generateSpectrograms(frame);
    [net,~,scores] = classifyAndUpdateState(net,spec);

    [conf,lin] = max(scores(:));
    classIdx   = ind2sub(size(scores),lin);
    classes    = Shared.setNoGestureUse(true);
    lab        = char(classes(classIdx));
    if conf < Shared.FRAME_CLASS_THRESHOLD, lab='noGesture'; end

    labs{w} = lab; pT(w) = toc(t0);
end
end



%====================================================================
%        ---  Statistical helper functions  ---
%====================================================================
function [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ...
          calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers)
classificationPerUser = zeros(numUsers,1);
recognitionPerUser    = zeros(numUsers,1);
overlapingPerUser     = zeros(numUsers,1);
processingTimePerUser = zeros(numUsers,1);

for i = 1:numUsers
    classificationPerUser(i) = sum(classifications(i,:)==1)/numel(classifications(i,:));
    recognitionPerUser(i)    = sum(recognitions(i,:)==1) / ...
                               sum(recognitions(i,:)==1 | recognitions(i,:)==0);
    ov = overlapings(i,:);
    overlapingPerUser(i)     = mean(ov(ov~=-1),'omitnan');
    processingTimePerUser(i) = mean(procesingTimes(i,:));
end
end

function [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers)
accC  = sum(all.classifications==1)/numel(all.classifications);
accR  = sum(all.recognitions==1) / sum(all.recognitions==1 | all.recognitions==0);
avgOv = mean(all.overlapings(all.overlapings~=-1),'omitnan');
avgPt = mean(all.procesingTimes);

globalResps = struct('accClasification',accC,'accRecognition',accR,...
                     'avgOverlapingFactor',avgOv,'avgProcesingTime',avgPt);

cp = perUser.classifications; rp = perUser.recognitions;
op = perUser.overlapings;     tp = perUser.procesingTimes;
stdC = sum((cp-accC).^2)/(max(1,numUsers-1));
stdR = sum((rp-accR).^2)/(max(1,numUsers-1));
stdO = sum((op-avgOv).^2)/(max(1,numUsers-1));
stdT = sum((tp-avgPt).^2)/(max(1,numUsers-1));

globalStds = struct('stdClassification',stdC,'stdRecognition',stdR,...
                    'stdOverlaping',stdO,'stdProcessingTime',stdT);
end

function results = calculateValidationResults(cls,rec,ovl,pt,nUsers)
[classU,recU,ovlU,ptU] = calculateMeanUsers(cls,rec,ovl,pt,nUsers);

fprintf('Results (mean of user results)\n');
fprintf('Classification | acc: %.4f | std: %.4f\n',mean(classU),std(classU));
fprintf('Recognition    | acc: %.4f | std: %.4f\n',mean(recU),std(recU));
fprintf('Overlap factor | avg: %.4f | std: %.4f\n',mean(ovlU),std(ovlU));
fprintf('Processing time| avg: %.4f | std: %.4f\n',mean(ptU),std(ptU));

all = struct('classifications',cls(:), 'recognitions',rec(:), ...
             'overlapings',ovl(:),     'procesingTimes',pt(:));
perUser  = struct('classifications',classU,'recognitions',recU,...
                  'overlapings',ovlU,'procesingTimes',ptU);
[glob,sd] = calculateResultsGlobalMean(all,perUser,nUsers);

fprintf('\nResults (Global results)\n');
fprintf('Classification | acc: %.4f | std: %.4f\n',glob.accClasification,sd.stdClassification);
fprintf('Recognition    | acc: %.4f | std: %.4f\n',glob.accRecognition,sd.stdRecognition);
fprintf('Overlap factor | avg: %.4f | std: %.4f\n',glob.avgOverlapingFactor,sd.stdOverlaping);
fprintf('Processing time| avg: %.4f | std: %.4f\n\n',glob.avgProcesingTime,sd.stdProcessingTime);

results = glob; %#ok<NASGU>
end



%====================================================================
%        Plot AVG (Train+Val) per user
%====================================================================
function plotEvolutionPerUser(resAll)

users = resAll(1).users;          % same users in al months
nU    = numel(users);
nM    = numel(resAll);
X     = 1:nM;                    
yLim  = [0 1];                   

for u = 1:nU
    % ----- book arrays -----
    clsAVG = zeros(1,nM);  recAVG = clsAVG;
    ovlAVG = clsAVG;       ptAVG  = clsAVG;

    % ----- fill with  AVG (Train+Val)/2 -----
    for m = 1:nM
        clsAVG(m) = (resAll(m).classificationTR(u) + ...
                     resAll(m).classificationVAL(u)) / 2;
        recAVG(m) = (resAll(m).recognitionTR(u) + ...
                     resAll(m).recognitionVAL(u)) / 2;
        ovlAVG(m) = (resAll(m).overlapTR(u) + ...
                     resAll(m).overlapVAL(u)) / 2;
        ptAVG(m)  = (resAll(m).ptimeTR(u) + ...
                     resAll(m).ptimeVAL(u)) / 2;
    end

    % ----- unique plot  AVG -----
    figure('Name',sprintf('AVG  – %s',string(users(u))),...
           'NumberTitle','off');

    subplot(2,2,1); plot(X,clsAVG,'-o','LineWidth',1.4);
        grid on; title('Classification'); ylim(yLim);

    subplot(2,2,2); plot(X,recAVG,'-og','LineWidth',1.4);
        grid on; title('Recognition');   ylim(yLim);

    subplot(2,2,3); plot(X,ovlAVG,'-or','LineWidth',1.4);
        grid on; title('Overlap');       ylim(yLim);

    subplot(2,2,4); plot(X,ptAVG,'-om','LineWidth',1.4);
        grid on; title('Proc. Time');    

    sgtitle(sprintf('Evolución AVG – Usuario: %s',string(users(u))));
end
end  % END

