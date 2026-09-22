function features = extract_features(iq)

    % SPECTRAQ - Basic IQ feature extraction
    %
    % Input:
    %   iq - structure returned by load_iq.m
    %
    % Output:
    %   features - structure containing basic signal features

    if ~isstruct(iq) || ~isfield(iq, 'samples')
        error('extract_features: input must be an IQ structure from load_iq.');
    end

    x = iq.samples(:);
    N = length(x);

    if N == 0
        error('extract_features: IQ sample array is empty.');
    end

    % ---------------------------------------------------------
    % Basic amplitude statistics
    % ---------------------------------------------------------

    magnitude = abs(x);

    features.num_samples = N;
    features.rms_magnitude = sqrt(mean(magnitude.^2));
    features.peak_magnitude = max(magnitude);
    features.mean_magnitude = mean(magnitude);
    features.std_magnitude = std(magnitude);

    % ---------------------------------------------------------
    % Sampling frequency
    % ---------------------------------------------------------

    if isfield(iq, 'fs_hz') && ~isempty(iq.fs_hz)
        Fs = iq.fs_hz;
        features.fs_hz = Fs;
    else
        Fs = [];
        features.fs_hz = [];
    end

    % ---------------------------------------------------------
    % Frequency-domain features
    % ---------------------------------------------------------

    if isempty(Fs)
        features.dominant_frequency_hz = [];
        features.spectral_centroid_hz = [];
        features.spectral_bandwidth_hz = [];
        features.occupied_bandwidth_90_hz = [];
    else

        % Use a manageable analysis segment
        max_samples = min(N, 500000);
        x_seg = x(1:max_samples);

        L = length(x_seg);

        % FFT
        X = fftshift(fft(x_seg));

        P = abs(X).^2;
        P = P / sum(P);

        f = (-L/2:L/2-1)' * (Fs/L);

        % Dominant frequency
        [~, idx] = max(P);
        features.dominant_frequency_hz = f(idx);

        % Spectral centroid
        features.spectral_centroid_hz = sum(abs(f) .* P);

        % Spectral bandwidth
        centroid = features.spectral_centroid_hz;

        features.spectral_bandwidth_hz = ...
            sqrt(sum((abs(f) - centroid).^2 .* P));

        % 90% occupied bandwidth
        [f_sorted, sort_idx] = sort(f);
        P_sorted = P(sort_idx);

        cumulative_power = cumsum(P_sorted);

        low_idx = find(cumulative_power >= 0.05, 1);
        high_idx = find(cumulative_power >= 0.95, 1);

        if ~isempty(low_idx) && ~isempty(high_idx)
            features.occupied_bandwidth_90_hz = ...
                f_sorted(high_idx) - f_sorted(low_idx);
        else
            features.occupied_bandwidth_90_hz = [];
        end

    end

    % ---------------------------------------------------------
    % Activity estimate
    % ---------------------------------------------------------

    threshold = 0.20 * max(magnitude);

    active_samples = magnitude > threshold;

    features.activity_ratio = ...
        sum(active_samples) / N;

end