function plot_curves(run_id)
%PLOT_CURVES Plot BER/BLER curves for a given run_id or all runs.
%   plot_curves(run_id) loads results/<run_id>_curves.mat and saves plots
%   to figures/. If run_id is empty or omitted, all runs are scanned.

    if nargin < 1
        run_id = "";
    end

    results_dir = "results";
    figures_dir = "figures";
    if ~exist(figures_dir, 'dir')
        mkdir(figures_dir);
    end

    if strlength(run_id) == 0
        files = dir(fullfile(results_dir, "*_curves.mat"));
        for k = 1:numel(files)
            [~, base_name] = fileparts(files(k).name);
            run_id = erase(base_name, "_curves");
            plot_single_run(results_dir, figures_dir, run_id);
        end
    else
        plot_single_run(results_dir, figures_dir, run_id);
    end
end

function plot_single_run(results_dir, figures_dir, run_id)
    mat_path = fullfile(results_dir, run_id + "_curves.mat");
    if ~exist(mat_path, 'file')
        warning("No curves file found for run_id: %s", run_id);
        return;
    end

    data = load(mat_path);
    if ~isfield(data, 'snr_db')
        warning("snr_db missing in %s", mat_path);
        return;
    end

    snr_db = data.snr_db;

    if isfield(data, 'ber')
        fig = figure('Visible', 'off');
        semilogy(snr_db, data.ber, '-o', 'LineWidth', 1.5);
        grid on;
        xlabel('SNR (dB)');
        ylabel('BER');
        title(run_id + " BER");
        saveas(fig, fullfile(figures_dir, run_id + "_ber.png"));
        close(fig);
    end

    if isfield(data, 'bler')
        fig = figure('Visible', 'off');
        semilogy(snr_db, data.bler, '-s', 'LineWidth', 1.5);
        grid on;
        xlabel('SNR (dB)');
        ylabel('BLER');
        title(run_id + " BLER");
        saveas(fig, fullfile(figures_dir, run_id + "_bler.png"));
        close(fig);
    end
end
