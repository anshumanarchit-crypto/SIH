function iq = load_iq(filename, fs_hz, fc_hz)

    % SPECTRAQ - IQ loader
    % Loads a MATLAB .mat IQ capture and returns a standard structure.
    %
    % filename : path to .mat file
    % fs_hz    : sampling frequency in Hz (optional)
    % fc_hz    : centre frequency in Hz (optional)

    if nargin < 1
        error('load_iq: filename is required.');
    end

    if ~isfile(filename)
        error('load_iq: file not found: %s', filename);
    end

    % Load MAT file
    S = load(filename);

    % Expected variable from the real dataset
    if ~isfield(S, 'IQ_samples')
        error('load_iq: variable IQ_samples was not found in the MAT file.');
    end

    x = S.IQ_samples;

    % Convert row vector to column vector
    if isvector(x)
        x = x(:);
    end

    % If IQ is stored as two columns [I Q], convert to complex IQ
    if ~isreal(x)
        iq_samples = x;
    elseif size(x, 2) == 2
        iq_samples = complex(x(:,1), x(:,2));
    else
        error('load_iq: IQ_samples must be complex or contain [I Q] columns.');
    end

    % Create output structure
    iq = struct();

    iq.samples = iq_samples;
    iq.num_samples = length(iq_samples);

    iq.fs_hz = [];
    if nargin >= 2 && ~isempty(fs_hz)
        iq.fs_hz = fs_hz;
    end

    iq.fc_hz = [];
    if nargin >= 3 && ~isempty(fc_hz)
        iq.fc_hz = fc_hz;
    end

    iq.filename = filename;
    iq.variable = 'IQ_samples';
    iq.data_type = class(x);

end