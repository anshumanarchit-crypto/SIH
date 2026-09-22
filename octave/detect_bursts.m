function bursts = detect_bursts(iq)

    % SPECTRAQ - Burst detection
    %
    % Input:
    %   iq - structure returned by load_iq.m
    %
    % Output:
    %   bursts - structure containing detected burst information

    if ~isstruct(iq) || ~isfield(iq, 'samples')
        error('detect_bursts: input must be an IQ structure from load_iq.');
    end

    x = iq.samples(:);
    N = length(x);

    if N == 0
        error('detect_bursts: IQ sample array is empty.');
    end

    % Sampling frequency
    if isfield(iq, 'fs_hz') && ~isempty(iq.fs_hz)
        Fs = iq.fs_hz;
    else
        error('detect_bursts: sampling frequency fs_hz is required.');
    end

    % ---------------------------------------------------------
    % Use first 500,000 samples for analysis
    % ---------------------------------------------------------

    N_analysis = min(N, 500000);
    x = x(1:N_analysis);

    magnitude = abs(x);
    power = magnitude.^2;

    % ---------------------------------------------------------
    % Smooth power envelope
    % ---------------------------------------------------------

    window = max(1, round(0.001 * Fs));

    kernel = ones(window, 1) / window;

    power_smooth = conv(power, kernel, 'same');

    % ---------------------------------------------------------
    % Detection threshold
    % ---------------------------------------------------------

    threshold = 0.20 * max(power_smooth);

    active = power_smooth > threshold;

    % Remove very short gaps / detections
    min_burst_samples = max(1, round(0.005 * Fs));

    % Find transitions
    transitions = diff([0; active; 0]);

    start_idx = find(transitions == 1);
    end_idx   = find(transitions == -1) - 1;

    % ---------------------------------------------------------
    % Keep bursts longer than minimum duration
    % ---------------------------------------------------------

    valid = (end_idx - start_idx + 1) >= min_burst_samples;

    start_idx = start_idx(valid);
    end_idx   = end_idx(valid);

    num_bursts = length(start_idx);

    % ---------------------------------------------------------
    % Create burst structure
    % ---------------------------------------------------------

    bursts = struct();

    bursts.count = num_bursts;
    bursts.threshold = threshold;
    bursts.analysis_samples = N_analysis;
    bursts.analysis_duration_s = N_analysis / Fs;

    bursts.start_sample = start_idx;
    bursts.end_sample = end_idx;

    bursts.start_time_s = (start_idx - 1) / Fs;
    bursts.end_time_s = (end_idx - 1) / Fs;

    bursts.duration_s = ...
        (end_idx - start_idx + 1) / Fs;

    % ---------------------------------------------------------
    % Burst statistics
    % ---------------------------------------------------------

    if num_bursts > 0

        burst_power = zeros(num_bursts, 1);

        for k = 1:num_bursts
            segment = power(start_idx(k):end_idx(k));
            burst_power(k) = mean(segment);
        end

        bursts.mean_power = burst_power;
        bursts.mean_burst_power = mean(burst_power);

        bursts.activity_ratio = ...
            sum(end_idx - start_idx + 1) / N_analysis;

        % Time between consecutive bursts
        if num_bursts > 1
            bursts.inter_burst_interval_s = ...
                bursts.start_time_s(2:end) - ...
                bursts.start_time_s(1:end-1);

            bursts.mean_inter_burst_interval_s = ...
                mean(bursts.inter_burst_interval_s);
        else
            bursts.inter_burst_interval_s = [];
            bursts.mean_inter_burst_interval_s = [];
        end

    else

        bursts.mean_power = [];
        bursts.mean_burst_power = [];
        bursts.activity_ratio = 0;
        bursts.inter_burst_interval_s = [];
        bursts.mean_inter_burst_interval_s = [];

    end

end