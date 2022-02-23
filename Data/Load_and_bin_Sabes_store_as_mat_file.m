% Author: Oscar Savolainen

% Iterate through files.
% NOTE: make sure the Indy and Loco files are both in this folder.
path_to_raw_Sabes_data = 'D:\Dropbox (Imperial NGNI)\NGNI Share\Storage\Neural Data - Community\2016-2017 Sabes Lab\Indy';

% Where we save Sabes data to. Should be:
% [project_root_directory]\Data\Sabes_data\
path_to_save_binned_data_to = 'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\Sabes_lab_data\MAT_data_correct\';

d = dir(path_to_raw_Sabes_data);
for dataset_ind = 1:length(d)
    dataset = d(dataset_ind).name;
    if contains(dataset,'indy') || contains(dataset,'loco') % only if actually an indy or loco file (anmes of subjects)
        fprintf([dataset, '\n'])
        
        clearvars -except d dataset dataset_ind 
        
        % Load data
        load([d(dataset_ind).folder,'\',d(dataset_ind).name])

        % Combine SUA into MUA
        nb_channels = length(spikes);
        up_to_x_SUA = min(size(spikes));

        % Get first and last time steps (initialise)
        first = 1000;
        last = 0;

        MUA = cell(nb_channels,1);
        MUA_vec = [];
        for chan = 1:nb_channels
            for SUA = 1:up_to_x_SUA
                MUA{chan,1} = [MUA{chan,1}; spikes{chan,SUA}];

                % Update first and last time steps 
                first = min([first, min(spikes{chan,SUA})]);
            end
            MUA_vec = [MUA_vec; sort(MUA{chan,1}) chan * ones(length(MUA{chan,1}),1)];
        end
        
        % Offset time
        if t(1) < first
            input('This should not be the case, verify')
        end
        MUA_vec(:,1) =  MUA_vec(:,1) - first;
        t = t - first; 

        %% Bin data at 1 ms
        for BP = [1 5 10 20 50 100]
            time_bins = min(t):BP/1000:max(t);
            channel_bins = 0.5:1:nb_channels+0.5;
            h = histogram2(MUA_vec(:,1),MUA_vec(:,2),time_bins,channel_bins);
            binned_MUA = uint8(h.Values);

            % We downsmaple the behavioral data
            trimmed_time_bins = time_bins(1:end-1);
            cursor_pos_resamp = interp1(t,cursor_pos,trimmed_time_bins);


            %% Save binned data, cursor pos, and time vector
            file_name = erase(dataset,'.mat');
            save([path_to_save_binned_data_to,file_name,'_BP_',num2str(BP),'ms.mat'],'binned_MUA','trimmed_time_bins','cursor_pos_resamp')
        end

    end
end
