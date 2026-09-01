%% Export the exact STAMP/MATPOWER operating point used for SSA (no plots)
close all;
clearvars;
clc;
path(pathdef);
addpath(genpath(pwd));

caseName = 'WSCC_SG_GFOR_GFOL';
path_results = '02_results/';
fanals = 2;  % MATPOWER, matching SSA.m

run set_file_names.m;
run read_data.m;
run preprocess_data.m;
run get_parameters.m;
set_breaker_state('line', 1, 'close');
run PF_results.m;
run update_OP.m;
run delta_slack_acdc.m;

% These are the exact linearization-point builders used by
% generate_elements.m/SSA.m (including STAMP's base and frame conversions).
lp_SG = generate_linearization_point_SG(T_SG, T_global, delta_slk);
lp_VSC_peak = generate_linearization_point_VSC_PEAK(T_VSC, T_global, delta_slk);

output_dir = fullfile(path_results, 'multivac');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

writetable(results.bus, fullfile(output_dir, ...
    'WSCC_SG_GFOR_GFOL_power_flow.csv'));
writetable(T_SG, fullfile(output_dir, ...
    'WSCC_SG_GFOR_GFOL_sg_operating_point.csv'));
writetable(T_VSC, fullfile(output_dir, ...
    'WSCC_SG_GFOR_GFOL_vsc_operating_point.csv'));

write_lp_csv(lp_SG, fullfile(output_dir, ...
    'WSCC_SG_GFOR_GFOL_sg_linearization_point.csv'));
write_lp_csv(lp_VSC_peak, fullfile(output_dir, ...
    'WSCC_SG_GFOR_GFOL_vsc_linearization_point.csv'));

fprintf('Exported STAMP operating point to %s\n', output_dir);

function write_lp_csv(points, filename)
    device = strings(0, 1);
    field = strings(0, 1);
    value = zeros(0, 1);
    for ii = 1:numel(points)
        names = fieldnames(points{ii});
        for jj = 1:numel(names)
            item = points{ii}.(names{jj});
            if isnumeric(item) && isscalar(item)
                device(end + 1, 1) = string(ii); %#ok<AGROW>
                field(end + 1, 1) = string(names{jj}); %#ok<AGROW>
                value(end + 1, 1) = double(item); %#ok<AGROW>
            end
        end
    end
    writetable(table(device, field, value), filename);
end
