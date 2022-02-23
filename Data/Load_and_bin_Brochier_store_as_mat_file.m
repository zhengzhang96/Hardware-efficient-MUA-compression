% Script to extract MUA from Thomas Brochier lab
% % Author: Oscar Savolainen
clearvars

% Iterate through files.
path_to_raw_Brochier_data = 'D:\Dropbox (Imperial NGNI)\NGNI Share\Storage\Neural Data - Community\2018 Thomas Brochier\Main content\datasets_matlab';
    
% Where we save Sabes data to. Should be:
% [project_root_directory]\Data\Sabes_data\
path_to_save_binned_data_to = 'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Upload_code\Data\Brochier_data\';


for dataset = ['L','N'] % Options: 'N', 'L'

    if contains(dataset, 'L')
        load([path_to_raw_Brochier_data,'\l101210-001_lfp-spikes.mat'])
    elseif contains(dataset, 'N')
        load([path_to_raw_Brochier_data,'\i140703-001_lfp-spikes.mat'])   
    end

    x = block.segments{1,1}.spiketrains;

    MUA = [];
    for i = 1:length(x) % Iterate through each SUA unit
        fprintf([x{1,i}.description,'\n'])

        temp = x{1,i}.times / 30000 * 1000; % The spike times of that unit in ms
        for channel_counter = 1:96 % iterate through all channels
            if contains([x{1,i}.description,'\n'],['channel: ',num2str(channel_counter),',']) % Check SUA is on that channel
                MUA = [MUA; temp' ones(length(temp),1)*channel_counter];
            end
        end
    end

    %% Bin data
    for BP = [1, 5, 10, 20, 50, 100]
        time_bins = 0:BP:max(MUA(:,1));
        channel_bins = 0.5:1:96.5;
        h = histogram2(MUA(:,1),MUA(:,2),time_bins,channel_bins);
        data = uint8(h.Values);

        binned_MUA = data;
        save([path_to_save_binned_data_to,'Monkey_',dataset,'_BP_',num2str(BP),'_ms.mat'],'binned_MUA')

    end
end