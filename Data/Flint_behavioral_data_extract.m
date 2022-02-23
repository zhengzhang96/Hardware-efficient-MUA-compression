function [binned_MUA,interp_collated_hand_vel,interp_norm_collated_hand_vel, time_vector] = Flint_behavioral_data_extract(Subject,rec,subject, bin_period)

    % Iterate through trials, get hand velocity data, collate tasks 
    % together into a single recording
    norm_collated_hand_vel = []; % each session z-standardised b4 collation
    collated_hand_vel = [];
    orig_time_vector = [];
    for task = 1:length(Subject(subject).Trial(:,1))
        norm_collated_hand_vel = [norm_collated_hand_vel; normalize(Subject(subject).Trial(task,1).HandVel)];
        collated_hand_vel = [collated_hand_vel; Subject(subject).Trial(task,1).HandVel];
        data_neuron{task,1} = Subject(subject).Trial(task,1).Neuron; % renaming for simplicity
        orig_time_vector = [orig_time_vector; Subject(subject).Trial(task,1).Time];
    end
    
    %% Turn SUA into MUA
    neuron_mapping = Subject(subject).Special.NeuronMapping;
    N = max(Subject(subject).Special.NeuronMapping(:,1));

    temp = [];
    for channel = 1:N
        neuron_indices = find(neuron_mapping(:,1) == channel);
        for task = 1:length(data_neuron)
            for i = 1:length(neuron_indices)
                temp2 =  data_neuron{task, 1}(neuron_indices(i)).Spike;
                temp = [temp; temp2 channel * ones(length(temp2),1)];
            end
        end
    end
    
    % Transform into binned MUA (at 1 ms)
    time_vector = min(Subject(subject).Trial(1).Time):bin_period*1e-3:max(Subject(subject).Trial(end).Time);
    channel_vector = 0.5:1:N+0.5;
    h = histogram2(temp(:,1), temp(:,2), time_vector, channel_vector);
    binned_MUA = uint8(h.Values);

    % Interpolate down to BP
    interp_collated_hand_vel = interp1(orig_time_vector,collated_hand_vel,time_vector(1:end-1));
    interp_norm_collated_hand_vel = interp1(orig_time_vector,norm_collated_hand_vel,time_vector(1:end-1));

end
