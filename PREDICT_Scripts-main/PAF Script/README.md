This is the directory to analyse the EEG dataset. 

# Pre-requirements to Run Script

## EEGLAB AND FIELDTRIP
You will need MATLAB (including the signal processing toolbox), and EEGLAB and FIELDTRIP toolboxes to run these scripts. EEGLAB AND FIELDTRIP should be kept in the "eeglab_fieldtrip" folder (or alternatively you can set the wpms.FUNCTIONS path to wherever you keep EEGLAB and FIELDTRIP. The current scripts call on "eeglab2019_1" and "fieldtrip-20200215", so if you have another version, you will need to update the script on the "add function" lines of Sensor_PAF_Pipeline.mat, Manual_PAF_Pipeline.m and Automated_PAF_Pipeline.m to the version that you are using. e.g. if you are using "eeglab2023.0", then change all references of "eeglab2019_1" to "eeglab2023.0". The required EEGLAB plugins are also detailed, including "bva-io1.5.13" and "firfilt". 

## Raw Data
You can download all the EEG data from https://openneuro.org/datasets/ds005486/versions/1.0.0 and store this in the "PREDICT_EEG_Raw_Data" directory

## Channels and Templates
In the "channel_info" folder, we have already uploaded details of the 63 channels for the EEG recording (chanlocs.mat). We have also uploaded the "neighbours" template which contains the neighbouring channels of each channel needed to interpolate any excluded electrodes. Lastly, we have uploaded "SarahTemplate_2" which is the sensorimotor component template (obtained from Author Sarah Margerison) used to choose the "winning sensorimotor component" in the automated ICA pipeline. 

## Output Directories
We have already included the Output directories for each of the three pipelines

# Using the Three different Scripts

## Sensor based EEG Analysis
To analyse the EEG data, please start at "Sensor_PAF_Pipeline.m" which is the backbone for the Automated and Manual component selection pipleines. It contains the script for the raw data loading (i.e the brain vision files), downsampling, re-referencing, filtering, bad channel selection, bad epoch selection, ICA to remove ocular artefacts and interpolation of channels. The output of this pipeline is used to calculate the sensorimotor ROI PAF in the main paper. All data is saved in the Output Folder directory

## Component based EEG Analysis
The Automated and manual component selection scripts start at the point after bad epochs are removed in fieldtrip from the Sensor_PAF_Pipeline script. Specifically, it calls on data from the "Output" data directory, and loads the "data_rejected" variable stored in the .mat file of a subject's folder. "data_rejected" is the fieldtrip EEG data structure post-bad epoch removal. Thus, these scripts will only work once the Sensor_PAF_Pipeline processing is complete. The job of these scripts is to isolate the sensorimotor alpha component, which is done manually (Manual_PAF_Pipeline) or is automated (Automated_PAF_Pipeline) - see comments for specific details and steps.   

