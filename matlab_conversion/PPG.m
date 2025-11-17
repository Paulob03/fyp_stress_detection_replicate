clear all
close all

signal = csvread('BVP.csv');


fs = 64;  % Sampling frequency
total_duration_minutes = length(signal) / fs / 60;  

% Calculate sample indices for the segments
first_segment_end_index = round(27 * 60 * fs); 
last_segment_start_index = round((total_duration_minutes - 5) * 60 * fs);

%segments
first_segment = signal(1:first_segment_end_index);
last_segment = signal(last_segment_start_index:end);
middle_segment = signal(first_segment_end_index+1:last_segment_start_index-1); %because the math task is limitless

% Number of middle segments
num_whole_minutes = floor(length(middle_segment) / fs / 60);


first_matrix = [];
middle_matrix = [];
last_matrix = [];

%allocate the segments in the right matrix
for i = 1:27

   
    start_index = (i-1)*fs*60 + 1;
    end_index = i*fs*60;
    if end_index <= length(first_segment)
        first_matrix(i, :) = first_segment(start_index:end_index);
    end
end 
for i=1:15
   if i <= num_whole_minutes
       
        segment_duration = 60*64;
        start_index = (i-1) * segment_duration + 1;
        end_index = i * segment_duration;
        if end_index <= length(middle_segment)
            middle_matrix(i, :) = middle_segment(start_index:end_index);
        end
    end
end
for i=1:5
    start_index = (i-1)*fs*60 + 1;
    end_index = i*fs*60;
    if end_index <= length(last_segment)
        last_matrix(i, :) = last_segment(start_index:end_index);
    end
end

%%
%adding the labels 0 for rest and 1 for stress
all_segments=[first_matrix; middle_matrix;last_matrix];

matrix_length = size(all_segments, 1);

pattern = ones(1, matrix_length);

pattern([1:3, 14:15, 21:22, 26:27]) = 0;
pattern([4:13, 16:20, 23:25]) = 1;
pattern(end-4:end) = [0 0 1 0 0];

remaining_elements = matrix_length - length(pattern);
pattern(end-remaining_elements+1:end) = 1;

tag_matrix = pattern;



%%
%removing the segments outside the range
num_segments = size(all_segments, 1);
percentRemoved_all = zeros(num_segments, 1);

for j = 1:num_segments
    current_segment = all_segments(j, :);


    fcutlow = 0.5;   % low cut frequency in Hz
    fcuthigh = 5;    % high cut frequency in Hz

    [A, B, C, D] = cheby2(2, 20, [fcutlow fcuthigh] / (64 / 2));
    [filter_SOS, g] = ss2sos(A, B, C, D);
    filtered_PPG = filtfilt(filter_SOS, g, current_segment);
    % 
    % figure
    % subplot(4, 1, 1)
    % plot(current_segment)
    % title('Raw signal')
    % 
    % subplot(4, 1, 2)
    % plot(current_segment)
    % hold on
    % plot(filtered_PPG)
    % title('Raw signal + filtered')
    % N = length(filtered_PPG);
    % Fs = 64;
    % Tsample = 1 / Fs;
    % T = (0:N - 1) * Tsample;
    % subplot(4, 1, 3)
    % plot(filtered_PPG)
    % title('Filtered signal')

    [pks,locs]=findpeaks(filtered_PPG,64,'MinPeakDistance',0.4, 'MinPeakHeight',0);
    % subplot(4, 1, 4)
    % plot(T, filtered_PPG, locs, pks, "*")
    % title('Filtered with Peaks detection')

    % Getting RRI values
    locs = locs * 64;
    peaks = locs;

    RRI = ones(1, numel(peaks) - 1);

    for k = 1:numel(peaks) - 1
        RRI(k) = (peaks(k + 1) - peaks(k)) / 64;
    end 

    original = numel(RRI);

    for k = 1:numel(RRI)
        if RRI(k) > 1.2 || RRI(k) < 0.5
            RRI(k) = NaN;
        end
    end

    RRI = rmmissing(RRI);
    modified = numel(RRI);
    
    % percentage removed
    percentRemoved = (original - modified) / original * 100;

    % Storing percentage removed
    percentRemoved_all(j, :) = percentRemoved;
end

%%
segments_percentage_tag=[all_segments,tag_matrix,percentRemoved_all];

%rows to keep
rows_to_keep = segments_percentage_tag(:, end) < 15;
segments_percentage_tag = segments_percentage_tag(rows_to_keep, :);

%%
segments_after_rejecting = segments_percentage_tag(:, 1:3840); %accepted segments each row one minute segment
tags_after_rejecting = segments_percentage_tag(:, 3841); %tags for accepted segments 0 rest, 1 stress

num_segments_after = size(segments_after_rejecting, 1);

%filtering and plotting ONLY accepted segments
for j = 1:num_segments_after
    current_segment = segments_after_rejecting(j, :);
    fcutlow = 0.5;   % low cut frequency
    fcuthigh = 5;    % high cut frequency

    [A, B, C, D] = cheby2(2, 20, [fcutlow fcuthigh] / (64 / 2));
    [filter_SOS, g] = ss2sos(A, B, C, D);
    filtered_PPG = filtfilt(filter_SOS, g, current_segment);

    figure
    subplot(4, 1, 1)
    plot(current_segment)
    title('Raw signal')

    subplot(4, 1, 2)
    plot(current_segment)
    hold on
    plot(filtered_PPG)
    title('Raw signal + filtered')
    N = length(filtered_PPG);
    Fs = 64;
    Tsample = 1 / Fs;
    T = (0:N - 1) * Tsample;
    subplot(4, 1, 3)
    plot(filtered_PPG)
    title('Filtered signal')

    [pks,locs]=findpeaks(filtered_PPG,64,'MinPeakDistance',0.4, 'MinPeakHeight',0);
    subplot(4, 1, 4)
    plot(T, filtered_PPG, locs, pks, "*")
    title('Filtered with Peaks detection')

end
