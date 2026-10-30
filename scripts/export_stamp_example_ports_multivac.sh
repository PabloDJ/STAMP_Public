#!/bin/bash
#SBATCH --job-name=stamp-port-export
#SBATCH --output=stamp_port_export_%j.out
#SBATCH --error=stamp_port_export_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
export MLM_LICENSE_FILE=/opt/mathworks/matlab2025b/licenses/network.lic:/opt/mathworks/matlab2025b/licenses/parallel.lic
cd "$HOME/STAMP_Public/STAMP"
/opt/mathworks/matlab2025b/bin/matlab -singleCompThread -batch "export_example_port_responses"
