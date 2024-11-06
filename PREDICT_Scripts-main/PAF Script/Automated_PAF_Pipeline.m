%% Purpose:
%%This is the script for the automated component selection analysis pipeline. Data was loaded
%%at the point where bad epochs were rejected (see sensor_PAF_pipeline),
%%and then ICA was run followed by an automated component selection to
%%reduce subjectivity associated with the manual component selection 

%This script uses Ali Mazehari's alpha IC isolating toolbox function to determine an
%ICA component which displays an alpha peak and most closely follows a
%topography indicative of a dipole in the sensorimotor cortex, according to
%Ali's template.

%This version constrains the ICA in a PCA.

%Sarah_template_2 is a template that is very similar to the sensorimotor
%alpha component published in Furman et al, Neuroimage. 2018 (fig2)
restoredefaultpath
clear all
close all

%% Set-up Workspace %%
cwd = [pwd filesep]; % Store current working directory
wpms = []; % pre-allocate a workspace variables in struct

% Locate folders of interest - Change if necessary

wpms.DATAIN    = [cwd 'Output' filesep]; %The data directory which should be the Output of the Sensor_PAF_Pipeline script
wpms.DATAOUT    = [cwd 'Output_AutomatedICA' filesep]; %Where the data gets stored
wpms.FUNCTIONS    = [cwd 'eeglab_fieldtrip' filesep]; %Add EEGLAB and FIELDTRIP
wpms.TEMPLATE = [cwd 'channel_info' filesep] %store channel info folder
template_home = wpms.TEMPLATE

% Structure and store subject codes
subjlist = dir([wpms.DATAIN 'sub-*']);
exp.sessions  = {'ses-00','ses-02','ses-05'};

%Add functions
addpath([wpms.FUNCTIONS 'fieldtrip-20200215'])
addpath([wpms.FUNCTIONS 'fieldtrip-20200215\external\eeglab'])
addpath([wpms.FUNCTIONS 'fieldtrip-20200215\utilities'])

% Run script looping through each subject code. 'px' is used becuase it
% stands out more within the loop amoung the variable names

for px = 1:length(subjlist);
    clearvars -except subjlist wpms exp template_home b f px num_its
    fprintf(['\n Analysing participant: ' subjlist(px).name '\n\n']);
    this_subject = subjlist(px).name;
    %load sensor level data 
    all_eeg_files  = dir([wpms.DATAIN subjlist(px).name filesep '*_processed_eeglab_fieldtrip.mat']);

    for f = 1:1
        num_its = [1 2 3 4 5 6 7 8 9 10] % 10 ICA Iterations 
        for b = 1:length(num_its)

            clearvars -except subjlist all_eeg_files wpms exp template_home b f px num_its

            iter = num_its(b);
            iter = num2str(iter)

            run = all_eeg_files(f).name;
            
            %load fieldtrip data from the bad trials rejected stage
            load([all_eeg_files(f).folder filesep all_eeg_files(f).name], 'data_rejected'); %load EEGLAB + FIELDTRIP processed data from the Sensor PAF pipeline
            data = data_rejected;
            %%%%
            
            %load channels and exclude AFz as this was ground 
            cfg = [];
            cfg.channel = {'all' '-AFz'};
            data = ft_selectdata(cfg, data);

            %% Run ICA 
            cfg = [];
            cfg.method ='fastica';
            cfg.runica.pca = 15
            [IC] = ft_componentanalysis(cfg, data);


            %% Choose a component based on topography
            
            % Load Templates
            cd(template_home)
            cfg.layout='EEG1010.lay'; % the layout for the data
            cfg.foi=[8 12] %%% what frequency to chose
            cfg.template='SarahTemplate_2'; %%%  This is the topography you want it to match to the componenent.
            %%%It uses this as a template for testing components that have alpha peak

            % Frequency Decomposition :
            cfg2       = [];
            cfg2.method = 'mtmfft';
            cfg2.output = 'pow';
            cfg2.pad    = 'maxperlen';
            cfg2.foilim = [1 30];
            cfg2.taper  = 'hanning';
            cfg2.keeptrials='no'
            alpha_IC   = ft_freqanalysis(cfg2, IC);

            % get bandwith
            bandwidth=find(alpha_IC.freq<=cfg.foi(2)&alpha_IC.freq>=cfg.foi(1));
            f_spectra=(alpha_IC.powspctrm);

            % use the matlab find peaks to find local maxima (i.e a peak in a bandwidth
            for k=1:size(f_spectra,1),
                [peak{k},locs{k}]=findpeaks( f_spectra(k,:) ) ;
                [value ploc(k)]=max(peak{k});
                where_peak_happen(k)=locs{k}(ploc(k));
            end

            % get compoment
            alpha_components=find(ismember(where_peak_happen,bandwidth));

            % load the template file
            load(cfg.template);

            % make sure betwee template and current electrodes match;
            elec=template_electrodes{:};  % basically load electrode positions, compare with template, match order
            [common_electrodes index]=ismember(IC.topolabel,elec);
            size(template);
            template_electrodes=elec(index);
            template=template(index);

            % correlate the electrodes in the data with the electrodes in the template
            for k=1:length(alpha_components),
                [r(k) p(k)]  =corr(template, IC.topo(:,alpha_components(k)));
            end
            
            %Identify component with the highest correlation
            [max_r maximum_component_index]=(max(abs(r)));

            % Plot findings
            selected_component=alpha_components(maximum_component_index);
            cfg.zlim='absmax';
            cfg.component=selected_component;
            ft_topoplotIC(cfg,IC); title(strcat('correlation prob between IC and template= ',num2str(p(maximum_component_index))));;
            colormap('jet');
            savefig([wpms.DATAOUT subjlist(px).name '_iter_' char(string(b)) '_topoplot.fig'])

            dummy=IC;
            dummy.dimord='chan_time'
            dummy.time=1;
            dummy.temp=template;
            cfg.parameter='temp';
            cfg.zlim='absmax';
            cfg.comment            = 'no';
            figure
            ft_topoplotER(cfg,dummy);title('template');colormap('jet');

            %% Now for the PAF calculations of the winning component
            
            %Note that this section of the script calculates PAF PER trial
            %rather than across trials (as specified in keeptrials = yes). 
            %This is NOT the PAF that is subsequently used in the PREDICT
            %main analysis. We have presented the PAF analysis used in the paper
            %in the next for loop. 
            
            
            %Re run the frequency decomposition with keeptrials = yes
            cfg3       = [];
            cfg3.method = 'mtmfft';
            cfg3.output = 'pow';
            cfg3.pad    = 'maxperlen';
            cfg3.foilim = [1 30];
            cfg3.taper  = 'hanning';
            cfg3.keeptrials='yes' %% this allows us to get a value for each trial (epoch), IC, and frequency
            alpha_IC_trials   = ft_freqanalysis(cfg3, IC);

            %extract the correct component by going back to original IC and select data
            %from just the winner and run a frequency decomposition 
            %In alpha_IC_trials.powspctrm have trial*comp*frequency- we only care
            %about 1 comp.
            allfreq_IC_timeseries = alpha_IC_trials.powspctrm(:,selected_component, :);
            
            %this is now a 2D mat with a power value per trial/frquency bin.
            %lifted from existing preprocessing script          
            PAF = [];
            PAF_Timeseries = [];
            Alpha_Power = [];
            Power_Timeseries=[];
            freq = 8:.2:12;

            temp = [];
            temp = squeeze(allfreq_IC_timeseries(:,1,:));; %generates matrix of frequency bin power values over time
            for i = 1:size(allfreq_IC_timeseries,1); %for each epoch
                temp_paf(i) = sum(freq.*temp(i,36:56))/sum(temp(i,36:56)); %Cog method
                temp_power(i) = sum(temp(i,36:56));  %corresponds to 8-12
            end

            PAF = mean(temp_paf);
            PAF_Timeseries= temp_paf; %generates a time-series of 4 sec epochs
            Alpha_Power = mean(temp_power); %avg over the entire scan
            Power_Timeseries = temp_power; %generates a time-series of 4 sec epochs

            corr_to_template = (p(maximum_component_index))
            selected_component=alpha_components(maximum_component_index);

            save([wpms.DATAOUT subjlist(px).name '_iter_' char(string(b)) '_sarah_ICA.mat'],'IC', 'corr_to_template', 'alpha_components', 'selected_component','alpha_IC','PAF','PAF_Timeseries', 'p', 'Alpha_Power', 'cfg', 'cfg2', 'cfg3', 'Power_Timeseries','alpha_IC_trials', 'maximum_component_index', 'max_r')

            close all
        end
    end
end


%% DECIDE WHICH ICA ITERATION GENERATES THE BEST CORRELATION
%Note some of this code reruns what is already run in the 10 ICA iterations 
%however this code re-calculate PAF from the sensorimotor component
%with the best match, WITHOUT rerunning a frequency decomposition to specify
%keep trials = yes, As this is the way we calculated PAF in the rest of the paper 

subjlist = dir([wpms.DATAOUT '*_sarah_ICA.mat']); %store all the ICA outputs 
file_count = 1;
for sub = 1:length(subjlist)/10; %Loop through each group of 10 files
    clearvars m index
    for this_file = 1:10;
        clearvars max_r PAF alpha_components alpha_IC_trials IC selected_component
        load([subjlist(file_count).folder filesep subjlist(file_count).name], 'IC', 'alpha_components', 'alpha_IC_trials', 'selected_component'); %Load the selected component from the ICA iteration
        fprintf(['\n Analysing participant: ' subjlist(file_count).name '\n\n']);
        
        % Load the Template
        cd(template_home)
        cfg.template='SarahTemplate_2';
        load(cfg.template);
        
        % correlate        
        % make sure to match electrodes;
        elec=template_electrodes{:};  % basically load electrode positions, compare with template, match order.. blah blah
        [common_electrodes index]=ismember(IC.topolabel,elec);
        size(template);
        template_electrodes=elec(index);
        template=template(index);

        % correlate the electrodes in the data with the electrodes in the template
        for k=1:length(alpha_components),
            [r(k) p(k)]  =corr(template, IC.topo(:,alpha_components(k)));
        end
        [max_r maximum_component_index]=(max(abs(r)));

        %now in alpha_IC_trials.powspctrm have trial*comp*frequency- we only care
        %about 1 comp.
        allfreq_IC_timeseries = alpha_IC_trials.powspctrm(:,selected_component, :);

        %Calculate PAF for winning component
        PAF = [];
        PAF_Timeseries = [];
        Alpha_Power = [];
        Power_Timeseries=[];
        freq = 8:.2:12;
        temp = [];
        temp = squeeze(allfreq_IC_timeseries(:,1,:)); %generates matrix of frequency bin power values over time
        for i = 1:size(allfreq_IC_timeseries,1); %for each epoch
            temp_paf(i) = sum(freq.*temp(i,36:56))/sum(temp(i,36:56)); %Cog method
        end
        PAF = mean(temp_paf);

        fprintf(['\n PAF is: ' char(string(PAF)) '\n\n']);
        fprintf(['\n Corr is: ' char(string(max_r)) '\n\n']);

        PAFs_this_participant(this_file,:) = PAF;
        correlations_this_participant(this_file,:) = max_r;
        file_count = file_count + 1;
    end
    %Decide on iteration with strongest correlation to the template
    [m,index] = max(correlations_this_participant);
    PAF_best_match(sub,:) = PAFs_this_participant(index)
    Best_correlations(sub,:) = m;
end
PAF_best_match(PAF_best_match == 0) = NaN
save([wpms.DATAOUT 'winners.mat'],'PAF_best_match','Best_correlations')

