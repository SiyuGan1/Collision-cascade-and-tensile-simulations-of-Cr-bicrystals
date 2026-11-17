#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ==================== Parameter configuration ====================
LAMMPS_CMD = "pirun -np $SLURM_NTASKS --map-by $MAP_OPT lmp -k on g $SLURM_GPUS_ON_NODE -sf kk -pk kokkos cuda/aware on neigh full comm device -in {0} -log {1}"

# Potential configuration
POTENTIAL = "Fe-Cr-Al.eam.fs"
ELEMENTS = "Cr"

# Physical parameters
STRAIN_RATE = 1e9
TOTAL_STRAIN = 0.20
TEMPERATURE = 300.0
TIMESTEP = 0.001

# Ensemble damping coefficients
DAMPING_T = 0.1
DAMPING_P = 1.0

# Post-irradiation handling parameters
PREBALANCE_STEPS = 20000
DO_PREBALANCE = True

# Initial velocity setup
SET_INITIAL_VELOCITY = True          # whether to assign initial velocities
VELOCITY_SEED = 12345                # random seed, fixed for reproducibility

# Output control
THERMO_INTERVAL = 500
DUMP_INTERVAL = 2000
RESTART_INTERVAL = 200000

# Post-processing parameters
YIELD_OFFSET = 0.002
SMOOTH_WINDOW = 5
USE_TRUE_STRESS = True
# ========================================================

def compute_simulation_params():
    erate_per_ps = STRAIN_RATE * 1e-12
    total_time_ps = TOTAL_STRAIN / erate_per_ps
    total_steps = int(total_time_ps / TIMESTEP)
    return total_steps, erate_per_ps

def generate_lammps_input(data_file, total_steps, erate_per_ps):
    prebalance_block = ""
    if DO_PREBALANCE:
        prebalance_block = f"""# ==================== Pre-balance after irradiation ====================
fix pre all npt temp {TEMPERATURE} {TEMPERATURE} {DAMPING_T} \
    x 0.0 0.0 {DAMPING_P} y 0.0 0.0 {DAMPING_P} z 0.0 0.0 {DAMPING_P} couple none
run {PREBALANCE_STEPS}
unfix pre
write_data prebalanced.data
print "Pre-balance completed: {PREBALANCE_STEPS} steps"
"""

    # Initial velocity setup
    velocity_block = ""
    if SET_INITIAL_VELOCITY:
        velocity_block = f"""# ==================== Initial velocity assignment ====================
velocity all create {TEMPERATURE} {VELOCITY_SEED} mom yes rot yes dist gaussian
print "Initial velocities assigned at {TEMPERATURE} K (seed={VELOCITY_SEED})"
"""

    # Only single-frame .data files are supported
    read_cmd = f'read_data "{data_file}"'
    data_source_info = data_file

    script = f"""# Cr Uniaxial Tensile Simulation
# Data: {data_source_info}
# Strain Rate: {STRAIN_RATE:.0e} s^-1
# Temperature: {TEMPERATURE} K

clear
units metal
atom_style atomic
atom_modify map array
boundary p p p

{read_cmd}

pair_style eam/fs
pair_coeff * * {POTENTIAL} {ELEMENTS}

neighbor 2.0 bin
neigh_modify delay 10 every 1 check yes

timestep {TIMESTEP}

{velocity_block}

{prebalance_block}

# ==================== Uniaxial tensile ====================
reset_timestep 0
variable x0 equal $(lx)
variable y0 equal $(ly)
variable z0 equal $(lz)
print "L0_X = ${{x0}}"
print "L0_Y = ${{y0}}"
print "L0_Z = ${{z0}}"

fix 1 all npt temp {TEMPERATURE} {TEMPERATURE} {DAMPING_T} \
    x 0.0 0.0 {DAMPING_P} y 0.0 0.0 {DAMPING_P} \
    couple none

fix 2 all deform 1 z erate {erate_per_ps:.6e} remap x

compute stress all stress/atom NULL
compute pe_atom all pe/atom

thermo {THERMO_INTERVAL}
thermo_style custom step temp pe ke etotal press pxx pyy pzz lx ly lz vol density

dump 1 all custom {DUMP_INTERVAL} dump.tensile.lammpstrj \
    id type x y z c_stress[1] c_stress[2] c_stress[3] c_stress[4] c_stress[5] c_stress[6] c_pe_atom
dump_modify 1 sort id

restart {RESTART_INTERVAL} restart.tensile.*

run {total_steps}

unfix 1
unfix 2
undump 1

write_data final_tensile.data
print "Tensile simulation completed successfully!"
"""
    return script

def run_lammps(input_file, log_file):
    cmd = LAMMPS_CMD.format(input_file, log_file)
    print(f">>> Run command: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        print(">>> LAMMPS run completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] LAMMPS run failed: {e}")
        return False

def extract_stress_strain(log_file):
    if not os.path.exists(log_file):
        print(f"[X] Log file not found: {log_file}")
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract initial lengths (take the last occurrence)
    x0 = y0 = z0 = None
    for m in re.finditer(r'L0_X\s*=\s*([0-9.eE+-]+)', content):
        x0 = float(m.group(1))
    for m in re.finditer(r'L0_Y\s*=\s*([0-9.eE+-]+)', content):
        y0 = float(m.group(1))
    for m in re.finditer(r'L0_Z\s*=\s*([0-9.eE+-]+)', content):
        z0 = float(m.group(1))

    if z0 is None:
        print("[X] L0_Z not found in log")
        return None
    if x0 is None: x0 = z0
    if y0 is None: y0 = z0

    # Extract data blocks: only match headers containing Pzz (tensile stage)
    blocks = []
    lines = content.splitlines()
    current_block = []
    in_data = False
    headers = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Step') and 'Pzz' in line:
            if current_block:
                blocks.append((headers, current_block))
            headers = line.split()
            current_block = []
            in_data = True
            continue
        if 'Loop' in line or 'Tensile simulation completed' in line:
            in_data = False
            if current_block:
                blocks.append((headers, current_block))
                current_block = []
            continue
        if in_data and not line.startswith('Step'):
            parts = line.split()
            if len(parts) >= len(headers):
                current_block.append(parts)

    if not blocks:
        print("[X] No data block found")
        # Debug: print all Step lines to help troubleshooting
        for line in lines:
            if line.strip().startswith('Step'):
                print(f"    Found Step line: {line.strip()[:80]}...")
        return None

    headers, data_rows = blocks[-1]

    try:
        pzz_idx = headers.index('Pzz')
        lz_idx = headers.index('Lz')
        lx_idx = headers.index('Lx')
        ly_idx = headers.index('Ly')
        temp_idx = headers.index('Temp')
        step_idx = headers.index('Step')
    except ValueError as e:
        print(f"[X] Failed to parse log column names: {e}")
        print(f"    Available columns: {headers}")
        return None

    steps, strains_z, stresses, strains_x, strains_y, temps = [], [], [], [], [], []

    for parts in data_rows:
        try:
            step = int(parts[step_idx])
            pzz = float(parts[pzz_idx])
            lz = float(parts[lz_idx])
            lx = float(parts[lx_idx])
            ly = float(parts[ly_idx])
            temp = float(parts[temp_idx])

            strain_z = (lz - z0) / z0
            strain_x = (lx - x0) / x0
            strain_y = (ly - y0) / y0
            stress = -pzz * 1e-4  # bar -> GPa

            steps.append(step)
            strains_z.append(strain_z)
            stresses.append(stress)
            strains_x.append(strain_x)
            strains_y.append(strain_y)
            temps.append(temp)
        except (ValueError, IndexError):
            continue

    if not steps:
        print("[X] Data block is empty")
        return None

    return (np.array(steps), np.array(strains_z), np.array(stresses),
            np.array(strains_x), np.array(strains_y), np.array(temps))

def compute_poisson_ratio(strains_z, strains_x, strains_y, elastic_limit=0.005):
    mask = strains_z < elastic_limit
    if np.sum(mask) < 2:
        mask = np.arange(min(10, len(strains_z)))
    nu_x = -np.polyfit(strains_z[mask], strains_x[mask], 1)[0]
    nu_y = -np.polyfit(strains_z[mask], strains_y[mask], 1)[0]
    nu_avg = (nu_x + nu_y) / 2
    return nu_avg, nu_x, nu_y

def compute_yield_stress(strains, stresses, E):
    offset_strains = strains - YIELD_OFFSET
    offset_stresses = E * offset_strains
    for i in range(1, len(strains)):
        if stresses[i] < offset_stresses[i] and stresses[i-1] >= offset_stresses[i-1]:
            diff_prev = stresses[i-1] - offset_stresses[i-1]
            diff_curr = offset_stresses[i] - stresses[i]
            if diff_prev + diff_curr == 0:
                continue
            frac = diff_prev / (diff_prev + diff_curr)
            yield_strain = strains[i-1] + frac * (strains[i] - strains[i-1])
            yield_stress = stresses[i-1] + frac * (stresses[i] - stresses[i-1])
            return yield_strain, yield_stress
    return None, None

def compute_proportional_limit(strains, stresses, threshold=0.05):
    elastic_limit = 0.005
    mask = strains < elastic_limit
    if np.sum(mask) < 2:
        mask = np.arange(min(10, len(strains)))
    E, intercept = np.polyfit(strains[mask], stresses[mask], 1)
    predicted = E * strains + intercept
    deviation = np.abs(stresses - predicted) / (np.abs(predicted) + 1e-6)
    for i in range(len(strains)):
        if strains[i] > 0.001 and deviation[i] > threshold:
            return strains[i], stresses[i], E
    return None, None, E

def plot_results(strains, stresses, temps, output_path,
                 E=None, yield_point=None, uts=None,
                 nu=None, true_stress=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Main plot: stress-strain
    ax1 = axes[0, 0]
    ax1.plot(strains * 100, stresses, 'b-', alpha=0.4, linewidth=1.0, label='Raw')
    if len(stresses) >= SMOOTH_WINDOW:
        smoothed = uniform_filter1d(stresses, size=SMOOTH_WINDOW)
        ax1.plot(strains * 100, smoothed, 'b-', linewidth=2.0, label='Smoothed')
    else:
        smoothed = stresses

    if E is not None and yield_point is not None:
        y_strain, y_stress = yield_point
        offset_strains = np.linspace(0, max(strains), 100)
        offset_stresses = E * (offset_strains - YIELD_OFFSET)
        valid = offset_stresses <= max(stresses) * 1.2
        ax1.plot(offset_strains[valid] * 100, offset_stresses[valid], 'r--',
                linewidth=1.2, alpha=0.8, label=f'0.2% Offset (E={E:.1f} GPa)')
        ax1.plot(y_strain * 100, y_stress, 'ro', markersize=8,
                label=f'Yield: {y_stress:.2f} GPa')

    if uts is not None:
        ax1.plot(uts[0] * 100, uts[1], 'g^', markersize=8,
                label=f'UTS: {uts[1]:.2f} GPa')

    ax1.set_xlabel("Strain (%)", fontsize=12)
    ax1.set_ylabel("Stress (GPa)", fontsize=12)
    ax1.set_title("Stress-Strain Curve", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(left=-0.5)
    ax1.set_ylim(bottom=-0.2)

    # Temperature-strain
    ax2 = axes[0, 1]
    ax2.plot(strains * 100, temps, 'm-', linewidth=1.5)
    ax2.axhline(TEMPERATURE, color='gray', linestyle='--', alpha=0.5, label='Target T')
    ax2.set_xlabel("Strain (%)", fontsize=12)
    ax2.set_ylabel("Temperature (K)", fontsize=12)
    ax2.set_title("Temperature Evolution", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Poisson effect
    ax3 = axes[1, 0]
    nu_text = f'Poisson Ratio (estimated):\n{nu:.3f}' if nu is not None else 'N/A'
    ax3.text(0.5, 0.5, nu_text,
            transform=ax3.transAxes, fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title("Poisson Effect", fontsize=13, fontweight='bold')
    ax3.axis('off')
    ax4 = axes[1, 1]
    if len(smoothed) > 5:
        d_sigma = np.diff(smoothed)
        d_epsilon = np.diff(strains)
        mask = d_epsilon > 1e-8
        hardening = np.zeros_like(d_epsilon)
        hardening[mask] = d_sigma[mask] / d_epsilon[mask]
        if len(hardening) >= 5:
            hardening = uniform_filter1d(hardening, size=5)
        ax4.plot(strains[:-1] * 100, hardening, 'c-', linewidth=1.5)
        ax4.set_xlabel("Strain (%)", fontsize=12)
        ax4.set_ylabel("Hardening Rate (GPa)", fontsize=12)
        ax4.set_title("Work Hardening Rate", fontsize=13, fontweight='bold')
        ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Chart saved: {output_path}")

def save_csv_data(strains, stresses_eng, stresses_true, temps, strains_x, strains_y, output_path):
    data = np.column_stack([
        strains, strains*100, stresses_eng, stresses_true, temps, strains_x, strains_y
    ])
    header = 'strain,strain_percent,engineering_stress_GPa,true_stress_GPa,temp_K,strain_x,strain_y'
    np.savetxt(output_path, data, fmt='%.6f', delimiter=',', header=header, comments='')
    print(f"CSV data saved: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python tensile.py <data_file> [options]")
        print("")
        print("  data_file          : initial structure file (.data format)")
        print("                       e.g. relaxed.data, frame_0.data, etc.")
        print("                       for multi-frame dumps, extract a single frame with extract_frame.py first")
        print("")
        print("  Options:")
        print("    --run            : run the LAMMPS simulation")
        print("    --plot           : post-process only (requires an existing log.tensile)")
        print("")
        print("  Examples:")
        print("    python tensile.py relaxed.data --run")
        print("    python tensile.py frame_10.data --run")
        print("    python tensile.py frame_10.data --plot")
        sys.exit(1)

    data_file = sys.argv[1]
    
    # Parse command-line arguments
    do_run = '--run' in sys.argv
    do_plot = '--plot' in sys.argv
    
    # Check file extension
    file_ext = os.path.splitext(data_file)[1].lower()
    if file_ext != '.data':
        print(f"[X] Error: only .data files are supported, current file: {data_file}")
        print("  for multi-frame dump files, extract a single frame with extract_frame.py first")
        sys.exit(1)

    if not os.path.exists(data_file):
        print(f"[X] Error: file not found {data_file}")
        sys.exit(1)

    total_steps, erate_per_ps = compute_simulation_params()

    print(f"\n{'='*60}")
    print(f"  Cr Uniaxial Tensile Simulation (Irradiation Mechanical Analysis) v4")
    print(f"{'='*60}")
    print(f"  Input structure : {data_file}")
    print(f"  Temperature     : {TEMPERATURE} K")
    print(f"  Strain rate     : {STRAIN_RATE:.0e} s^-1")
    print(f"  Target strain   : {TOTAL_STRAIN*100:.0f}%")
    print(f"  Total steps     : {total_steps:,} ({total_steps*TIMESTEP:.1f} ps)")
    print(f"  Pre-balance     : {'on' if DO_PREBALANCE else 'off'} ({PREBALANCE_STEPS} steps)")
    print(f"  Initial velocity: {'on (seed=' + str(VELOCITY_SEED) + ')' if SET_INITIAL_VELOCITY else 'off'}")
    print(f"  Stress type     : {'True stress' if USE_TRUE_STRESS else 'Engineering stress'}")
    print(f"{'='*60}\n")

    script = generate_lammps_input(data_file, total_steps, erate_per_ps)
    with open("in.tensile", "w") as f:
        f.write(script)

    if do_run or not do_plot:
        if not run_lammps("in.tensile", "log.tensile"):
            sys.exit(1)

    if do_plot or do_run or os.path.exists("log.tensile"):
        print(f"\nExtracting mechanical data...")
        result = extract_stress_strain("log.tensile")

        if result is None:
            print("[X] Data extraction failed, please check the log file")
            sys.exit(1)

        steps, strains, stresses, strains_x, strains_y, temps = result

        if len(strains) == 0:
            print("[X] No valid data extracted")
            sys.exit(1)

        # Poisson ratio
        nu, nu_x, nu_y = compute_poisson_ratio(strains, strains_x, strains_y)

        # Elastic modulus
        elastic_limit = 0.005
        mask = strains < elastic_limit
        if np.sum(mask) < 3:
            mask = np.arange(min(10, len(strains)))
        E, intercept = np.polyfit(strains[mask], stresses[mask], 1)

        # Smoothing
        if len(stresses) >= SMOOTH_WINDOW:
            stresses_smooth = uniform_filter1d(stresses, size=SMOOTH_WINDOW)
        else:
            stresses_smooth = stresses

        # True stress
        stresses_true = stresses * (1 + strains)
        stresses_smooth_true = stresses_smooth * (1 + strains)

        if USE_TRUE_STRESS:
            stresses_plot = stresses_true
            stresses_smooth_plot = stresses_smooth_true
        else:
            stresses_plot = stresses
            stresses_smooth_plot = stresses_smooth

        # Yield point
        yield_strain, yield_stress = compute_yield_stress(strains, stresses_plot, E)
        yield_method = "0.2% offset"
        if yield_stress is None:
            yield_strain, yield_stress, E = compute_proportional_limit(strains, stresses_plot)
            yield_method = "proportional limit (5% deviation)"

        # UTS
        max_idx = np.argmax(stresses_smooth_plot)
        uts_strain = strains[max_idx]
        uts_stress = stresses_smooth_plot[max_idx]

        fracture_strain = strains[-1]
        fracture_stress = stresses_plot[-1]

        # [Note] E is already in GPa, the /1e3 conversion was removed
        print(f"\n{'='*60}")
        print(f"  [Mechanical property extraction results]")
        print(f"{'='*60}")
        print(f"  Elastic modulus E   : {E:.2f} GPa")
        print(f"  Poisson ratio nu    : {nu:.3f} (nu_x={nu_x:.3f}, nu_y={nu_y:.3f})")
        if yield_stress is not None:
            print(f"  Yield stress sigma_y: {yield_stress:.2f} GPa  ({yield_method})")
            print(f"  Yield strain eps_y  : {yield_strain*100:.2f}%")
        else:
            print(f"  Yield stress sigma_y: not found")
        print(f"  UTS                 : {uts_stress:.2f} GPa @ {uts_strain*100:.2f}%")
        print(f"  Final strain/stress : {fracture_strain*100:.2f}% / {fracture_stress:.2f} GPa")
        print(f"  Mean temperature    : {np.mean(temps):.1f} K (max={np.max(temps):.1f} K)")
        print(f"{'='*60}\n")

        save_csv_data(strains, stresses, stresses_true, temps, strains_x, strains_y, "stress_strain.csv")

        yield_point = (yield_strain, yield_stress) if yield_stress is not None else None
        uts_point = (uts_strain, uts_stress)
        plot_results(strains, stresses_plot, temps, "stress_strain.png",
                    E=E, yield_point=yield_point, uts=uts_point, nu=nu,
                    true_stress=USE_TRUE_STRESS)

        # [Note] E unit in summary.txt
        with open("summary.txt", "w") as f:
            f.write(f"Cr Tensile Simulation Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Input: {data_file}\n")
            f.write(f"Temperature: {TEMPERATURE} K\n")
            f.write(f"Strain Rate: {STRAIN_RATE:.0e} s^-1\n")
            f.write(f"Stress Type: {'True' if USE_TRUE_STRESS else 'Engineering'}\n\n")
            f.write(f"Elastic Modulus E: {E:.2f} GPa\n")
            f.write(f"Poisson Ratio nu: {nu:.3f}\n")
            if yield_stress:
                f.write(f"Yield Stress sigma_y: {yield_stress:.2f} GPa ({yield_method})\n")
                f.write(f"Yield Strain eps_y: {yield_strain*100:.2f}%\n")
            f.write(f"UTS: {uts_stress:.2f} GPa\n")
            f.write(f"Max Temperature: {np.max(temps):.1f} K\n")
        print("Summary saved: summary.txt")

if __name__ == "__main__":
    main()
