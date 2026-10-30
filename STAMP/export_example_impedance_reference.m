function export_example_impedance_reference()
% Export the raw bus-4 GFOR admittance from the saved STAMP example.
% The example plots -Y to use a load-oriented current convention. This
% export keeps the original generator-oriented current for comparison with
% VeraGrid's positive injected current.

stamp_dir = fileparts(mfilename('fullpath'));
data_file = fullfile(stamp_dir, '00_tool', 'State Space Modelling', ...
    'Analysis', 'Frequency Domain Analysis', 'example_data2.mat');
result_dir = fullfile(stamp_dir, '02_results', 'comparison');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
result_file = fullfile(result_dir, 'example_data2_gfor_admittance.csv');

saved = load(data_file, 'factor', 'l_blocks_list');
frequencies_hz = [1, 10, 100, 1000];
input_names = {'NET.vn4q'; 'NET.vn4d'};
output_names = {'VSC1.iq'; 'VSC1.id'};

row_count = numel(saved.factor) * numel(frequencies_hz) * 4;
rows = zeros(row_count, 6);
next_row = 1;
for factor_index = 1:numel(saved.factor)
    blocks = saved.l_blocks_list{1, factor_index};
    converter = connect(blocks{6}, input_names, output_names);
    response = freqresp(converter, 2*pi*frequencies_hz);
    for frequency_index = 1:numel(frequencies_hz)
        for output_index = 1:2
            for input_index = 1:2
                value = response(output_index, input_index, frequency_index);
                rows(next_row, :) = [saved.factor(factor_index), ...
                    frequencies_hz(frequency_index), output_index, input_index, ...
                    real(value), imag(value)];
                next_row = next_row + 1;
            end
        end
    end
end

reference = array2table(rows, 'VariableNames', ...
    {'factor', 'frequency_hz', 'output_index', 'input_index', 'real', 'imag'});
writetable(reference, result_file);
fprintf('Exported %d admittance entries to %s\n', height(reference), result_file);
end
