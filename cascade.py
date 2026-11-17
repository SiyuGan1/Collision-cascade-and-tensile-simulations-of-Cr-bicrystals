import os, subprocess, math, random
import re
import numpy as np

# =========================================================================
# 1. Configuration parameters
# =========================================================================
LAMMPS_CMD = "mpirun -np $SLURM_NTASKS --map-by $MAP_OPT lmp -k on g $SLURM_GPUS_ON_NODE -sf kk -pk kokkos cuda/aware on neigh full comm device -in {0} -log {1}"

INIT_DATA = "init.data"
POTENTIAL_FILE = "Fe-Cr-Al.eam.fs"
ELEMENTS = "Cr"
MASS_PKA, LATTICE_CONST = 51.996, 2.88

# Cascade control parameters
PKA_ENERGY_KEV = 10                    # PKA energy (keV)
PKA_Z_OFFSET = 20.0                   # vertical distance of PKA from the grain boundary (A)
THERMO_THICKNESS = 9.0                # boundary thermostat layer thickness (A)

# === PKA direction control parameters ===
N_CASCADES = 50                        # number of cascade repetitions
REF_DIR = [0.0, 0.0, -1.0]            # reference direction
TILT_MIN_DEG = 0.0                    # minimum tilt angle (deg)
TILT_MAX_DEG = 10.0                    # maximum tilt angle (deg)
DIRECTION_MODE = "random_cone"        # "uniform_azimuth" | "sweep_angle" | "random_cone"

# === Channeling control parameters ===
CHANNEL_MAX_Z_DIFF = 100.0             # threshold for the max Z-coordinate difference of displaced atoms (A)
CHANNEL_TIME_CUTOFF = 1             # channeling detection time window (ps), auto-disabled afterwards
DISP_THRESHOLD = 1                 # displacement threshold to flag an atom as displaced (A)

# Electronic stopping control parameters
USE_ELECTRONIC_STOPPING = True
STOPPING_FILE = "Cr_stopping.txt"
STOPPING_ECUT = 10.0

# =========================================================================
# 2. Helper functions
# =========================================================================
def run_lmp(inp, log):
    cmd = LAMMPS_CMD.format(inp, log)
    print(f">>> Run: {cmd}")
    subprocess.check_call(cmd, shell=True)

def get_pka_id(data_file, tz_offset):
    with open(data_file, 'r') as f: lines = f.readlines()

    box = {}
    for line in lines[:30]:
        for ax in ('x', 'y', 'z'):
            if f'{ax}lo {ax}hi' in line:
                box[ax] = [float(x) for x in line.split()[:2]]

    lx, ly, lz = [b[1]-b[0] for b in box.values()]
    tx, ty, tz = box['x'][0]+lx/2, box['y'][0]+ly/2, box['z'][0]+lz/2 + tz_offset

    pka_id, min_dist = None, float('inf')
    start = next(i for i, l in enumerate(lines) if l.startswith('Atoms')) + 2

    for line in lines[start:]:
        p = line.split()
        if not p: break
        try:
            dx, dy, dz = float(p[2])-tx, float(p[3])-ty, float(p[4])-tz
            dx -= round(dx/lx)*lx; dy -= round(dy/ly)*ly; dz -= round(dz/lz)*lz
            dist = dx**2 + dy**2 + dz**2
            if dist < min_dist: min_dist, pka_id = dist, int(p[0])
        except ValueError: continue

    print(f">>> PKA located: ID={pka_id}, near X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f}")
    return pka_id

def generate_pka_directions(n_cascades, theta_min_deg, theta_max_deg, ref_dir, mode="uniform_azimuth"):
    directions = []
    ref = np.array(ref_dir, dtype=float)
    ref = ref / np.linalg.norm(ref)

    if abs(ref[2]) < 0.9:
        perp1 = np.cross(ref, [0, 0, 1])
    else:
        perp1 = np.cross(ref, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(ref, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    for i in range(n_cascades):
        if mode == "uniform_azimuth":
            theta_deg = (theta_min_deg + theta_max_deg) / 2.0
            phi = 2 * math.pi * i / n_cascades
        elif mode == "sweep_angle":
            theta_deg = theta_min_deg + (theta_max_deg - theta_min_deg) * i / max(1, n_cascades - 1)
            phi = 2 * math.pi * i / n_cascades
        elif mode == "random_cone":
            cos_tmax = math.cos(math.radians(theta_max_deg))
            cos_tmin = math.cos(math.radians(theta_min_deg))
            cos_theta = cos_tmax + (cos_tmin - cos_tmax) * random.random()
            theta_deg = math.degrees(math.acos(cos_theta))
            phi = 2 * math.pi * random.random()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        theta_rad = math.radians(theta_deg)
        dir_vec = (math.sin(theta_rad) * math.cos(phi) * perp1 +
                   math.sin(theta_rad) * math.sin(phi) * perp2 +
                   math.cos(theta_rad) * ref)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        directions.append(dir_vec.tolist())

    return directions

def check_channeling_from_log(log_file, channel_time_cutoff, channel_max_z_diff):
    """
    Determine whether channeling occurred, based on the LAMMPS log file.
    """
    if not os.path.exists(log_file):
        return "unknown"

    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
    except Exception as e:
        print(f"    Warning: error reading log file: {e}")
        return "unknown"

    # Check whether halt_channeling was triggered
    channeling_halt_pattern = r'Fix halt condition for fix-id halt_channeling met on step (\d+) with value ([\d.]+)'
    channeling_match = re.search(channeling_halt_pattern, log_content)

    # Check whether halt_cascade was triggered (time reached 30 ps)
    cascade_halt_pattern = r'Fix halt condition for fix-id halt_cascade met on step (\d+) with value ([\d.]+)'
    cascade_match = re.search(cascade_halt_pattern, log_content)

    if channeling_match:
        step = int(channeling_match.group(1))
        value = float(channeling_match.group(2))

        # Extract the time corresponding to that step from the log
        time_pattern = rf'Step\s+Time\s+Dt.*?^\s*{step}\s+([\d.]+)'
        time_match = re.search(time_pattern, log_content, re.MULTILINE | re.DOTALL)

        if time_match:
            halt_time = float(time_match.group(1))
        else:
            halt_time = None

        if halt_time is not None:
            if halt_time < channel_time_cutoff:
                print(f"    Channeling triggered: step={step}, time={halt_time:.4f}ps, z_spread={value:.2f}A")
                return "yes"
            else:
                print(f"    Warning: channeling triggered at {halt_time:.4f}ps (> {channel_time_cutoff}ps)")
                return "unknown"
        else:
            print(f"    Channeling triggered: step={step}, z_spread={value:.2f}A (time unknown)")
            return "yes"

    elif cascade_match:
        step = int(cascade_match.group(1))
        print(f"    Cascade ended normally: step={step}, no channeling detected")
        return "no"

    else:
        print(f"    Warning: simulation ended abnormally, cannot determine channeling")
        return "unknown"

# =========================================================================
# 3. Main workflow and LAMMPS script generation
# =========================================================================
if __name__ == "__main__":
    os.makedirs('relax', exist_ok=True)
    os.makedirs('irradiation', exist_ok=True)

    # --- 1. Relaxation stage ---
    if not os.path.exists("relax/relaxed.data"):
        with open("in.relax", "w") as f: f.write(f"""
units metal
dimension 3
boundary p p p
atom_style atomic
read_data {INIT_DATA}
pair_style eam/fs
pair_coeff * * {POTENTIAL_FILE} {ELEMENTS}
thermo 1000
minimize 1e-10 1e-10 10000 10000
reset_timestep 0
timestep 0.001
velocity all create 300.0 12345 dist gaussian
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
variable atom_time atom time
dump 1 all custom 10000 relax/dump_relax.lammpstrj id type x y z v_atom_time
run 20000
write_data relax/relaxed.data
""")
        run_lmp("in.relax", "relax/relax.log")

    # --- 2. Generate PKA direction list ---
    pka_directions = generate_pka_directions(
        N_CASCADES, TILT_MIN_DEG, TILT_MAX_DEG, REF_DIR, mode=DIRECTION_MODE
    )

    print(f"\n>>> Generated {N_CASCADES} PKA directions (angle with {REF_DIR}: {TILT_MIN_DEG}-{TILT_MAX_DEG} deg):")
    for idx, d in enumerate(pka_directions, 1):
        angle = math.degrees(math.acos(np.dot(d, np.array(REF_DIR)/np.linalg.norm(REF_DIR))))
        print(f"    Cascade {idx:02d}: [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]  (angle: {angle:.2f} deg)")

    # --- 3. Repeated cascade stage ---
    pka_id = get_pka_id("relax/relaxed.data", PKA_Z_OFFSET)

    # Create the summary file up front and write the header
    # Include dir_x, dir_y, dir_z as incident direction columns
    summary_file = "irradiation/cascade_summary.txt"
    with open(summary_file, "w") as f:
        f.write("# cascade_no, pka_id, z_offset(A), dir_x, dir_y, dir_z, pka_vx, pka_vy, pka_vz, tilt_angle(deg), output_dir, e_stopping, channeling_detected\n")

    for i in range(1, N_CASCADES + 1):
        pka_dir = pka_directions[i-1]

        # Incident direction (unit vector)
        dir_x, dir_y, dir_z = pka_dir[0], pka_dir[1], pka_dir[2]

        # Velocity magnitude
        v_mag = 98.2269 * math.sqrt(2 * PKA_ENERGY_KEV * 1000 / MASS_PKA)
        # Velocity components
        vx, vy, vz = v_mag * dir_x, v_mag * dir_y, v_mag * dir_z

        actual_angle = math.degrees(math.acos(np.dot(pka_dir, np.array(REF_DIR)/np.linalg.norm(REF_DIR))))

        print(f"\n{'='*60}")
        print(f">>> Starting cascade {i}/{N_CASCADES}")
        print(f">>> PKA incident direction: [{dir_x:+.6f}, {dir_y:+.6f}, {dir_z:+.6f}]")
        print(f">>> PKA velocity: [{vx:.2f}, {vy:.2f}, {vz:.2f}] (angle with reference {actual_angle:.2f} deg)")
        if USE_ELECTRONIC_STOPPING:
            print(f">>> Electronic stopping: ON (file={STOPPING_FILE}, Ecut={STOPPING_ECUT} eV)")
        else:
            print(f">>> Electronic stopping: OFF")
        print(f">>> Channeling detection: ON (threshold={CHANNEL_MAX_Z_DIFF}A, time window={CHANNEL_TIME_CUTOFF}ps)")
        print(f"{'='*60}")

        cascade_dir = f"irradiation/cascade_{i:02d}"
        os.makedirs(cascade_dir, exist_ok=True)

        langevin_seed = 98765 + i

        if USE_ELECTRONIC_STOPPING:
            if not os.path.exists(STOPPING_FILE):
                print(f"Warning: electronic stopping data file {STOPPING_FILE} not found, disabling electronic stopping")
                e_stopping_fix = ""
                e_stopping_flag = "off"
                thermo_style = "thermo_style custom step time dt temp pe ke etotal press"
            else:
                e_stopping_fix = f"fix e_stop all electron/stopping {STOPPING_ECUT} {STOPPING_FILE}"
                e_stopping_flag = f"on(Ecut={STOPPING_ECUT})"
                thermo_style = "thermo_style custom step time dt temp pe ke etotal press f_e_stop"
        else:
            e_stopping_fix = ""
            e_stopping_flag = "off"
            thermo_style = "thermo_style custom step time dt temp pe ke etotal press"

        inp_file = f"in.cascade_{i:02d}"
        log_file = f"{cascade_dir}/cascade.log"

        with open(inp_file, "w") as f: f.write(f"""
units metal
dimension 3
boundary p p p
atom_style atomic
atom_modify map array
read_data relax/relaxed.data
pair_style eam/fs
pair_coeff * * {POTENTIAL_FILE} {ELEMENTS}

variable pka_id equal {pka_id}
region core_zone block $(xlo+{THERMO_THICKNESS}) $(xhi-{THERMO_THICKNESS}) $(ylo+{THERMO_THICKNESS}) $(yhi-{THERMO_THICKNESS}) $(zlo+{THERMO_THICKNESS}) $(zhi-{THERMO_THICKNESS}) units box side out
group thermo_atoms dynamic all region core_zone every 1
group pka_atom id {pka_id}

velocity pka_atom set {vx:.4f} {vy:.4f} {vz:.4f} units box

reset_timestep 0
fix g_nve all nve
fix langevin_thermo thermo_atoms langevin 300.0 300.0 1.0 {langevin_seed}
fix dt_var all dt/reset 1 1e-9 1e-3 {0.005 * LATTICE_CONST} units box

# Electronic stopping setup
{e_stopping_fix}

# =========================================================================
# Channeling dynamic detection module (only active for t < {CHANNEL_TIME_CUTOFF} ps)
# =========================================================================
compute disp all displace/atom
variable is_disp atom "c_disp[4] > {DISP_THRESHOLD}"
compute zu_coord all property/atom zu

variable z_max_calc atom "v_is_disp * c_zu_coord - (1 - v_is_disp) * 1000000.0"
variable z_min_calc atom "v_is_disp * c_zu_coord + (1 - v_is_disp) * 1000000.0"

compute max_z all reduce max v_z_max_calc
compute min_z all reduce min v_z_min_calc
variable z_spread equal "c_max_z - c_min_z"

# Time-window control: enable detection only while sim_time < {CHANNEL_TIME_CUTOFF}
variable channel_active equal "(time < {CHANNEL_TIME_CUTOFF})"
variable z_spread_active equal "v_channel_active * v_z_spread + (1 - v_channel_active) * 0.0"

# Channeling halt: only judged during the active window; after the window z_spread_active stays 0 and never triggers
fix halt_channeling all halt 100 v_z_spread_active > {CHANNEL_MAX_Z_DIFF} error continue

# Print channeling log for post-analysis
fix print_channel all print 500 "$(step) $(time) $(c_max_z) $(c_min_z) $(v_z_spread) $(v_channel_active) $(v_z_spread_active)" file {cascade_dir}/channeling.log screen no title "# step time max_z min_z z_spread active z_spread_active"
# =========================================================================

thermo 1000
{thermo_style}
variable atom_time atom time

variable next_dump equal "step + (time < 1.0) * 1000 + ((time >= 1.0) && (time < 5.0)) * 2000 + ((time >= 5.0) && (time < 25.0)) * 3000 + (time >= 25.0) * 1000"

dump 2 all custom 1000 {cascade_dir}/dump_cascade.lammpstrj id type x y z v_atom_time
dump_modify 2 every v_next_dump first yes

variable sim_time equal time
variable time_diff equal "v_sim_time - 30.0"
fix halt_cascade all halt 10 v_time_diff > 0.0

run 1000000

write_data {cascade_dir}/final_cascade.data
""")
        # Run the current cascade
        run_lmp(inp_file, log_file)

        # =====================================================================
        # Judge channeling
        # =====================================================================
        channeling_detected = check_channeling_from_log(
            log_file, CHANNEL_TIME_CUTOFF, CHANNEL_MAX_Z_DIFF
        )

        # After each cascade, append the result (including incident direction) to the summary file
        with open(summary_file, "a") as f:
            f.write(f"{i}, {pka_id}, {PKA_Z_OFFSET:.1f}, {dir_x:.6f}, {dir_y:.6f}, {dir_z:.6f}, {vx:.4f}, {vy:.4f}, {vz:.4f}, {actual_angle:.2f}, {cascade_dir}, {e_stopping_flag}, {channeling_detected}\n")

        print(f">>> Cascade {i} done, channeling: {channeling_detected}")

    print(f"\n{'='*60}")
    print(">>> All cascade simulations finished!")
    print(f">>> Summary saved to: {summary_file}")
