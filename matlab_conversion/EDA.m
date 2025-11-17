clear all
close all 

signal = csvread('EDA.csv');
signal=resample(signal,64,4);

fs = 64;  % Sampling frequency
total_duration_minutes = length(signal) / fs / 60;  

% Calculate sample indices for the segments
first_segment_end_index = round(27 * 60 * fs);
last_segment_start_index = round((total_duration_minutes - 5) * 60 * fs); 

%segments
first_segment = signal(1:first_segment_end_index);
last_segment = signal(last_segment_start_index:end);
middle_segment = signal(first_segment_end_index+1:last_segment_start_index-1);%because the math task is limitless

% Number of middle segments
num_whole_minutes = floor(length(middle_segment) / fs / 60);


first_matrix = [];
middle_matrix = [];
last_matrix = [];


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
num_segments = size(all_segments, 1);
percentRemoved_all = zeros(num_segments, 1);

for j = 1:num_segments
    current_segment = all_segments(j, :);

N=40;
alpha=0.4;
gaussFilter = gausswin(N,alpha);
gaussFilter = gaussFilter / sum(gaussFilter);
sz=size(current_segment);
y=zeros(sz);
for i=1:sz(1)
    filtered(i,:) = filtfilt(gaussFilter,1,current_segment(i,:));
end

N = length(filtered);
fs = 64;
Tsample = 1/fs;
T = (0:N-1)*Tsample;

figure
plot(T,current_segment,'LineWidth',1.2)
hold on
plot(T,filtered,'LineWidth',1.2)
xlabel('Time (S)','FontSize',20,'FontWeight','bold')
ylabel('Amplitude (µS)','FontSize',20,'FontWeight','bold')
legend({'Raw signal','Filtered signal'},'Location','southeast','FontSize', 20)
 
end



















