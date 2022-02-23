%% Script for binning Flint MUA data
% Author: Oscar Savolainen
clearvars 
close all

Flint_raw_data = 'D:\Dropbox (Imperial NGNI)\NGNI Share\Storage\Neural Data - Community\2012 Flint & Slutzky';
path_to_save_binned_data_to = 'D:\Dropbox (Imperial NGNI)\NGNI Share\Workspace\Oscar\Work\MUA compression\End days\MAT_data\';

% Transform SUA to MUA
for rec = 1:5
    % Load data
    try
        MUA_data_file = [Flint_raw_data,'\Flint_2012_e',num2str(rec),'.mat'];
        load(MUA_data_file)
    catch
    end
            
    for subject = 1:5
        [rec, subject]
        
        for binning_res = [1 5 10 20 50 100]
             try
                save_path = [path_to_save_binned_data_to,'Rec_',num2str(rec),'_Subject_',num2str(subject),'_BP_',num2str(binning_res),'_ms.mat'];

                % Bin MUA data
                [binned_MUA,collated_hand_vel,norm_collated_hand_vel, time_vector] = Flint_behavioral_data_extract(Subject,rec,subject, binning_res);

                save(save_path,'binned_MUA','collated_hand_vel','norm_collated_hand_vel')
            catch
            end
        end
    end
end