% Separate positive and negative samples in the dataset

clc;
clear;

% Load file lists
filelist1 = dir('data\*.mat');  % Radar data files
filelist2 = dir('layer\*.mat'); % Layer data files
LAKE_excel = readtable('training.xlsx'); % Label data

label = [];
[m_1, n_1] = size(LAKE_excel);

% Record file length
len_file = length(filelist1); 

for i=1:len_file
    disp(i)
    file_name1 = filelist1(i).name;    % Current radar data filename
    a = ['C:\Users\ma_97\Desktop\减均值\training\data\' file_name1]; % Radar data path
    load(a);                           % Load radar data
    file_name2 = filelist2(i).name;    % Current layer data filename
    b = ['C:\Users\ma_97\Desktop\减均值\training\layer\' file_name2]; % Layer data path 
    load(b);                           % Load layer data
    
    Radar_data = lp(Data);             % Process radar data
    [m, n] = size(Radar_data);         % Data dimensions (m=depth, n=along track)
    
    % Extract ice-bed interface (bottom)
    Bottom = interp1(Time,1:length(Time),layerData{2}.value{2}.data);
    if all(isnan(Bottom))
        break;
    else
        bottom = fillmissing(Bottom,'nearest'); % Fill missing bottom values
    end
    Bottom = interp1(GPS_time,bottom,GPS_time); % Ice-bed interface
    bottom = Bottom;
    
    % Extract ice surface
    Surface = interp1(Time,1:length(Time),layerData{1}.value{2}.data);
    Surface = interp1(GPS_time,Surface,GPS_time); % Ice surface
    if all(isnan(Surface))
        surface = 300*ones(size(Surface)); % Default surface if missing
    else
        surface = fillmissing(Surface,'nearest'); % Fill missing surface
    end

    dt = Time(2)-Time(1);              % Sampling interval
    [m,n] = size(Radar_data);          % Radar data dimensions
    w = 150;                           % Interface width
    bottom2 = zeros(2*w+1,n);          % Interface matrix size
    fs = 1 / dt;                       % Sampling frequency
    window = 20;                       % Smoothing window size 1
    movwindow = 15;                    % Smoothing window size 2
    fontsize = 12;                     % Font size
    
    %% Flatten ice-bed interface to prevent picking errors
    clear bottommax;
    clear S_Bed;
    bottommax = [];
    S_Bed = [];
    for i = 1:n
        bottom2(:,i) = Radar_data(ceil(bottom(i))-150:ceil(bottom(i)+150),i);
        bottomtemp(:,i) = Radar_data(ceil(bottom(i))-70:ceil(bottom(i)+70),i);
        [S_Bed(i), bottommax(i)] = max(bottomtemp(:,i)); % Find bed reflection peak
    end

    bottom = bottommax + bottom -51;
    bottom_aver = movmean(bottom,window); % Moving average of bottom
    s = movmean(Radar_data,window,2);    % Smoothed radar data
    
    % Physical constants
    c = 299792458;                       % Speed of light
    er = 3.17;                           % Dielectric constant of ice
    
    % Calculate ice parameters
    H = (bottom-surface) * dt * c / sqrt(er) / 2 / 1000; % Ice thickness (km)
    h = surface * dt * c / 2 / 1000;     % Aircraft altitude (km)
    G = 10 * log10(2 * (h*1000 + H*1000 / sqrt(er))); % Geometric correction
    P_Bed = S_Bed + 2 * G + 2 * 11.7 * H; % Corrected bed reflectivity
    P_Bed_aver = movmean(P_Bed,window);   % Smoothed reflectivity

    mean_P_Bed = mean(P_Bed, 'omitnan');  % Mean P_Bed for entire file
    P_Bed_centered = P_Bed - mean_P_Bed;  % Center P_Bed by subtracting mean

    dz = dt * c / (2 * sqrt(er));        % Vertical resolution
    
    % Calculate horizontal distance between points
    clear dx;
    for i = 1:length(bottom)-1
        dx(i) = distance(Latitude(i),Longitude(i),Latitude(i+1),Longitude(i+1),referenceEllipsoid('GRS80'));
    end
    averare_result = mean(dx);
    A_scopre_dis = round(averare_result);

    bottom2 = bottom2-mean(mean(bottom2));
    
    % Hydraulic potential calculation
    rou_i = 917;                         % Ice density
    rou_w = 1000;                        % Water density
    hydraulic = (rou_i/rou_w) * Elevation + (1-rou_i/rou_w) * (Elevation - H * 1000); % Hydraulic potential
    y = [];
    y = diff(hydraulic);
    T_hydraulic = (y) ./dx;              % Hydraulic potential gradient
    T_hydraulic = [T_hydraulic T_hydraulic(end)];

    clear bottom_aver_max;
    for k = 1:n
        bottomtemp(:,k) = s(ceil(bottom_aver(k))- 70 : ceil(bottom_aver(k) + 70),k);
        [S_temp, bottom_aver_max(k)] = max(bottomtemp(:,k));
    end
    bottom_aver = bottom_aver_max + bottom_aver - 51;

    % Flatten interface for better visualization
    clear bottom2_aver;
    clear fig_bottom2;
    for k = 1:n
        bottom2_aver(:,k) = s(ceil(bottom_aver(k))-w : ceil(bottom_aver(k)+w),k);
        fig_bottom2(:,k) = Radar_data(ceil(bottom_aver(k))-w : ceil(bottom_aver(k)+w),k);
    end
    bottom2_aver = bottom2_aver - mean(mean(bottom2_aver,"omitnan"),"omitnan");

    % Load labels
    current_name = file_name1(1:end-4);

    label_lake_idx_begin = [];
    label_lake_idx_end = [];
    label_begin_0 = zeros(n,1);

    % Match labels with current data frame
    for j=1:m_1
        if strcmp(current_name, LAKE_excel.Cresis_Frame(j))
            label_lake_idx_begin = LAKE_excel.Creis_Begin(j);
            label_lake_idx_end = LAKE_excel.Cresis_End(j);
            label_begin_0(label_lake_idx_begin:label_lake_idx_end, 1) = 1; % 1 indicates subglacial lake
        end
    end
    
    final_label = label_begin_0;

    % Calculate TFF (Time-Frequency Feature)
    optimalWindows_list = [];
    window_size = 30;  % Window size range

    stft_window_results = [];
    clear maxS;
    clear maxF;
    clear maxT;
    detectionfreq = [];
    detection = [];
    detectionfreq_new = [];
    
    % Perform STFT analysis
    for kk = 1:n
        current_window = window_size;
        ccc = bottom2_aver(:,kk);
        [S,F,T] = mySTFT(rickerize2(bottom2_aver(:,kk))',dt,current_window);

        maxS(kk) = max(max(S));          % Maximum amplitude
        [t1,t2] = find(S == maxS(kk));  % Find peak frequency
        maxF(kk) = F(t1(1));            % Frequency at peak
        maxT(kk) = T(t2(1));            % Time at peak

        % Filter edge effects
        if t2(1) <= 120 || t2(end) >= 180
            maxS(kk) = 0;
        end
    end

    detectionfreq = maxF .* maxS /fs;    % Detection frequency feature

    % Calculate roughness
    roughness = [];
    r = [];

    if A_scopre_dis == 18 
        % Roughness calculation for 18m spacing
        for q=1:n
            if q-n<=-5
                e{q} = [bottom(q), bottom(q+1), bottom(q+2),bottom(q+3),bottom(q+4),bottom(q+5)];
                ave_e = mean(e{q});
                l = e{q};
                f = (l-ave_e).^2;
                f = 1/5*(sum(f));
                f = sqrt(f); % RMS roughness
            else
               r(q)=r(q-1);
            end
           r(q) = f; 
        end
     else
         % Roughness calculation for other spacings
         for q=1:n
            if q-n<=-3
                e{q} = [bottom(q), bottom(q+1), bottom(q+2),bottom(q+3)];
                ave_e = mean(e{q});
                l = e{q};
                f = (l-ave_e).^2;
                f = 1/3*(sum(f));
                f = sqrt(f); % RMS roughness
            else
               r(q)=r(q-1);
            end
           r(q) = f; 
        end
     end

    rock_height = Elevation - H*1000;    % Bedrock elevation

    % Create output table with all features
    signal_table = table(Latitude', Longitude', label_begin_0, S_Bed', P_Bed', H', r', hydraulic', T_hydraulic', detectionfreq', rock_height', ...
        'VariableNames', {'Lat', 'Lon','Label' ,'BRP', 'CBRP', 'Ice_thickness', 'Roughness', 'Hydraulic', 'Hydraulic_gradient', 'TFF', 'bed elevation'});
    
    % Save results to CSV
    output_folder_2 = 'C:\Users\ma_97\Desktop\减均值\training\csv';
    output_filename_signal = [current_name '.csv'];
    output_path_signal_csv = fullfile(output_folder_2, output_filename_signal);
    writetable(signal_table, output_path_signal_csv);
end