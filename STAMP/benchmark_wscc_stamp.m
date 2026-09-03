%% Repeatable single-process benchmark for the 88-state WSCC STAMP model.
close all; clearvars; clc;
path(pathdef); addpath(genpath(pwd));

caseName = 'WSCC_SG_GFOR_GFOL';
path_results = '02_results\';
fanals = 2;
number_runs = 20;

setup_start = tic;
run set_file_names.m
run read_data.m
run preprocess_data.m
run get_parameters.m
set_breaker_state('line', 1, 'close')
run PF_results.m
run update_OP.m
run delta_slack_acdc.m
run generate_NET_with_Qneg.m
run generate_elements.m
input = {'NET.Rld1'};
select_all_outputs;
ss_sys = connect(l_blocks{:}, input, output);
setup_seconds = toc(setup_start);
A = ss_sys.A;

% Warm both paths before collecting samples.
eig(A);
FEIG(ss_sys, [0.25 0.25 0.25], 'o', false);

eig_seconds = zeros(number_runs, 1);
ssa_seconds = zeros(number_runs, 1);
for run_index = 1:number_runs
    timer = tic;
    eigenvalues = eig(A); %#ok<NASGU>
    eig_seconds(run_index) = toc(timer);

    timer = tic;
    eigenvalue_table = FEIG(ss_sys, [0.25 0.25 0.25], 'o', false); %#ok<NASGU>
    ssa_seconds(run_index) = toc(timer);
end

implementation = repmat("STAMP", number_runs, 1);
run = (1:number_runs)';
results = table(implementation, run, eig_seconds, ssa_seconds);
output_dir = fullfile('02_results', 'comparison');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end
writetable(results, fullfile(output_dir, 'benchmark_stamp_multivac.csv'));

fprintf('STAMP states: %d\n', size(A, 1));
fprintf('STAMP setup: %.9f s\n', setup_seconds);
fprintf('STAMP eig median: %.9f s (min %.9f s)\n', median(eig_seconds), min(eig_seconds));
fprintf('STAMP SSA median: %.9f s (min %.9f s)\n', median(ssa_seconds), min(ssa_seconds));
