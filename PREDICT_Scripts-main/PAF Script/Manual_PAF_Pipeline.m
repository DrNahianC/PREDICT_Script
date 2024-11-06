%%This is the script for the main data analysis pipeline. Data was loaded
%%at the point where bad epochs were rejected (see sensor_PAF_pipeline) and then
%ICA was run to manually select the sensorimotor component. The chosen
%components in our study have been uploaded on OSF


clear all
close all

%% Set-up Workspace %%
cwd = [pwd filesep]; % Store current working directory
wpms = []; % pre-allocate a workspace variables in struct


wpms.DATAIN    = [cwd 'Output' filesep];  %The data directory which should be the Output of the Sensor_PAF_Pipeline script
wpms.DATAOUT    = [cwd 'Output_ManualICA' filesep]; %Where the data gets stored
wpms.FUNCTIONS    = [cwd 'eeglab_fieldtrip' filesep]; %Add EEGLAB and FIELDTRIP
wpms.CHANNELS = [cwd 'channel_info' filesep] %store channel info folder

% Structure and store subject codes
subjlist = dir([wpms.DATAIN 'sub-*']);
exp.sessions  = {'ses-00','ses-02','ses-05'};

%Add eeglab and fieldtrip functions
addpath([wpms.FUNCTIONS '\eeglab2019_1'])
addpath(genpath([wpms.FUNCTIONS '\eeglab2019_1\functions']));
addpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\xdfimport1.16'])
addpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\bva-io1.5.13'])
addpath(genpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\PrepPipeline0.55.3']));
addpath(genpath([wpms.FUNCTIONS '\eeglab2019_1\plugins\firfilt']));
addpath([wpms.FUNCTIONS '\fieldtrip-20200215'])
addpath([wpms.FUNCTIONS '\fieldtrip-20200215\external\eeglab'])

%Get channel info
load([wpms.CHANNELS '\chanlocs.mat'])
load([wpms.CHANNELS '\neighbour_template.mat'])
final_data = ""; %Create empty string variable where all PAF values will be stored



%% FIELDTRIP
clearvars all_eeg_files EEG n_triggers prompt rejected triggers px day
for px = 1:length(subjlist);
    clearvars all_eeg_files cfg data data_rejected data_freq data_pruned EEG freq paf power promt n_rejected rejected temp temp2 X y z
    fprintf(['\n Analysing participant: ' subjlist(px).name '\n\n']);
    this_subject = subjlist(px).name;
    all_fieldtrip_files  = dir([wpms.DATAIN subjlist(px).name filesep '*_processed_eeglab_fieldtrip.mat']); %store filenames of all the EEGLAB + FIELDTRIP processed data from the Sensor PAF pipeline
    all_eeglab_files  = dir([wpms.DATAIN subjlist(px).name filesep '*_processed_eeglab.mat']);%store filenames of all the EEGLAB processed data from the Sensor PAF pipeline

    for day = 1:length(all_fieldtrip_files)
        clearvars cfg data data_freq data_comp EEG freq paf power promt rejected temp temp2 X y z n_rejected;
        load([all_fieldtrip_files(day).folder filesep all_fieldtrip_files(day).name], 'data_rejected'); %Load the data at the point of the Sensor PAF pipeline where bad trials were rejected BEFORE THE ICA
        load([all_eeglab_files(day).folder filesep all_eeglab_files(day).name], 'EEG'); %Load the EEGLAB structure (used to access channel numbers for this px)

        %% ICA and Plot Components

        %Run ICA 
        cfg          = [];
        cfg.method = 'runica';
        data_comp = ft_componentanalysis(cfg, data_rejected);

        %Create topoplots of the identified components to visualize which
        %ones have a sensorimotor topography 
        cfg           = [];
        cfg.component = [1:20];       % specify the component(s) that should be plotted
        cfg.layout    = 'EEG1010.lay'; % specify the layout file that should be used for plotting
        cfg.viewmode = 'component';
        cfg.comment   = 'no';
        f = figure;
        movegui(f,[0 300]);
        ft_topoplotIC(cfg, data_comp);
        cfg.position = [700 300 500 500];

        % Create plots of each component for each trial to confirm that
        % alpha waves are present
        ft_databrowser(cfg, data_comp); 

        % Conduct Frequency Decomposition of the data in component space
        cfg              = [];
        cfg.output       = 'pow';
        cfg.channel      = 'all';%compute the power spectrum in all ICs
        cfg.method       = 'mtmfft';
        cfg.taper        = 'hanning';
        cfg.foi          = [2:.20:50];
        data_freq = ft_freqanalysis(cfg, data_comp);
        
        % Spectral Plots of each component to visualize which ones have
        % distinct alpha peaks
        nsubplots = 25;
        nbyn = sqrt(nsubplots);% sqrt(nsubplots) should not contain decimals,
        type doc subplot;
        Nfigs = ceil(size(data_comp.topo,1)/nsubplots);
        tot = Nfigs*nsubplots;
        rptvect = 1:size(data_comp.topo,1);
        rptvect = padarray(rptvect, [0 tot-size(data_comp.topo,1)], 0,'post');
        rptvect = reshape(rptvect,nsubplots,Nfigs)';
        for r=1:1;
            f = figure
            k=0;
            for j=1:20;
                if~(rptvect(r,j)==0);
                    k=k+1;
                    cfg=[];
                    cfg.channel = rptvect(r,j);
                    subplot(nbyn,nbyn,k);ft_singleplotER(cfg,data_freq);
                    windows = {[9:.2:11],[8:.2:12]};
                    window_indicies = {[36:46],[31:51]};
                    freq_window = 2; %8-12Hz window
                    temp = transpose(data_freq.powspctrm(j,:));
                    temp2 = zscore(temp);
                    spectral_data(:,j) = temp;
                    paf_this_component(:,j) = sum(windows{freq_window}'.*(((temp(window_indicies{freq_window})))))/sum(((((temp(window_indicies{freq_window}))))));
                end
            end
        end

        %% Choose Sensorimotor Component
        fprintf(['\n Waiting for component selection for: ' subjlist(px).name '\n\n']);
        prompt = 'Please identify the sensorimotor component';
        chosen_component = input(prompt);
        close all;
        %% Extract and store PAF for chosen component
        chosen_component_paf = paf_this_component(chosen_component);
        chosen_component_data = data_comp.topo(:,chosen_component);
        chosen_spectral_data = spectral_data(:,chosen_component);

        %save data
        save([wpms.DATAOUT subjlist(px).name  '_component_data.mat'],'data_comp','data_freq','chosen_component_data','chosen_component_paf', 'chosen_spectral_data','spectral_data');% Nahian - Adjust based on what you th
        %save topoplots of chosen component
        map = topoplot(chosen_component_data, EEG.chanlocs, 'headrad', 0.66, 'plotrad', 0.72);
        title(string(chosen_component_paf))
        saveas(map,[wpms.DATAOUT subjlist(px).name '_topoplot'], 'png');% Nahian - Adjust based on what you think data output should be called
        %save spectral plot of chosen component
        spectral = plot([2:.20:50],spectral_data(:,chosen_component));
        title(string(chosen_component_paf))
        saveas(spectral,[wpms.DATAOUT subjlist(px).name '_spectralplot'], 'png');% Nahian - Adjust based on what you think data output should be called

        %store chosen component data
        final_data(px,1) = subjlist(px).name;
        final_data(px,2) = string(chosen_component_paf);
        final_data(px,3) = string(chosen_component);

        %%his code will create matrices containing component level data for
        %%each channel while  accounting for missing channels, so that they can be
        %%collated across participants
        chosen_component_data(:,2) = chosen_component_data(:,1);
        for i = 1:63;
            channels(i,:) = string(chanlocs(i).labels);
        end
        channels_after_exclusion = string(data_rejected.label');

        missing_channels = channels(~all(ismember(channels,channels_after_exclusion),2),:);
        for j = 1:length(missing_channels);
            index_missing_channels(j) = find(channels==missing_channels(j));
        end

        for j = 1:length(missing_channels)
            chosen_component_data = [chosen_component_data(1:(index_missing_channels(j)-1), :); nan(1,size(chosen_component_data,2)); chosen_component_data(index_missing_channels(j):end, :)];
        end

        chosen_component_data = string(chosen_component_data);
        chosen_component_data(:,1) = channels;
        save([wpms.DATAOUT subjlist(px).name  '_components_all_channels.mat'],'chosen_component_data')


        close all;
        % end
    end
    fprintf(['\n Finished analysing participant: ' subjlist(px).name '\n\n']);
end







