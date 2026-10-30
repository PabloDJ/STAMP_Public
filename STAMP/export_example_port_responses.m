function export_example_port_responses(output_file)
% Export numeric port responses from STAMP's two saved MATLAB examples.
% The saved ss objects require MATLAB; the output is a plain v7 MAT file.

if nargin < 1
    output_file = fullfile('02_results', 'comparison', 'stamp_example_ports.mat');
end

example_dir = fullfile('00_tool', 'State Space Modelling', 'Analysis', ...
                       'Frequency Domain Analysis');
frequencies_hz = [1, 10, 100, 800, 1000];
angular_frequencies = 2 * pi * frequencies_hz;
frequency_count = numel(frequencies_hz);

gfor_data = load(fullfile(example_dir, 'example_data2.mat'), ...
                 'factor', 'l_blocks_list', 'caseName', 'lp_VSC');
gfor_factors = gfor_data.factor;
gfor_port_voltage_qd = [gfor_data.lp_VSC{1}.vg_q0; ...
                        gfor_data.lp_VSC{1}.vg_d0];
gfor_admittance = complex(zeros(2, 2, frequency_count, numel(gfor_factors)));
gfor_grid_impedance = complex(zeros(2, 2, frequency_count, numel(gfor_factors)));
for factor_index = 1:numel(gfor_factors)
    blocks = gfor_data.l_blocks_list{factor_index};
    converter = connect(blocks{6}, ...
        {'NET.vn4q'; 'NET.vn4d'}, {'VSC1.iq'; 'VSC1.id'});
    gfor_admittance(:, :, :, factor_index) = ...
        freqresp(converter, angular_frequencies);
    grid = connect(blocks{[1:5, 7]}, ...
        {'VSC1.iq'; 'VSC1.id'}, {'NET.vn4q'; 'NET.vn4d'});
    gfor_grid_impedance(:, :, :, factor_index) = ...
        freqresp(grid, angular_frequencies);
end
gfor_case = gfor_data.caseName;

gfl_data = load(fullfile(example_dir, 'example_data1.mat'), ...
                'factor', 'l_blocks_list', 'caseName');
gfl_factors = gfl_data.factor;
first_blocks = gfl_data.l_blocks_list{1};
gfl_converter = connect(first_blocks{7}, ...
    {'NET.vn1q'; 'NET.vn1d'}, {'VSC1.iq'; 'VSC1.id'});
gfl_admittance = freqresp(gfl_converter, angular_frequencies);
gfl_grid_impedance = complex(zeros(2, 2, frequency_count, numel(gfl_factors)));
for factor_index = 1:numel(gfl_factors)
    blocks = gfl_data.l_blocks_list{factor_index};
    grid = connect(blocks{1:6}, ...
        {'VSC1.iq'; 'VSC1.id'}, {'NET.vn1q'; 'NET.vn1d'});
    gfl_grid_impedance(:, :, :, factor_index) = ...
        freqresp(grid, angular_frequencies);
end
gfl_case = gfl_data.caseName;

output_dir = fileparts(output_file);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
save(output_file, 'frequencies_hz', 'gfor_case', 'gfor_factors', ...
    'gfor_port_voltage_qd', ...
    'gfor_admittance', 'gfor_grid_impedance', 'gfl_case', 'gfl_factors', ...
    'gfl_admittance', 'gfl_grid_impedance', '-v7');
fprintf('Exported both STAMP examples to %s\n', output_file);
end
