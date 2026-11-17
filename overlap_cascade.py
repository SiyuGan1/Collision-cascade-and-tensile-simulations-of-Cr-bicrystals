#!/usr/bin/env python3
import os, subprocess, math, random, glob
import numpy as np

# ============================================================================
# Configuration parameters
# ============================================================================
LAMMPS_CMD = (
    "mpirun -np $SLURM_NTASKS --map-by $MAP_OPT "
    "lmp -k on g $SLURM_GPUS_ON_NODE "
    "-sf kk -pk kokkos cuda/aware on neigh full comm device "
    "-in {0} -log {1}"
)

INIT_DATA      = "init.data"
POTENTIAL      = "Fe-Cr-Al.eam.fs"
ELEMENTS       = "Cr"
MASS_PKA       = 51.996
LAT_CONST      = 2.88

PKA_E          = 10
PKA_ZOFF       = 20.0
THICK          = 9.0

REF_DIR        = [0.0, 0.0, -1.0]
TILT_MIN_DEG   = 0.0
TILT_MAX_DEG   = 10.0
DIRECTION_MODE = "random_cone"

N_CASC         = 200        # total target count
START_CASC     = 1          # start from the 1st run
SHIFT_SEED     = 42

RELAX_DIR      = "relax"
IRRAD_DIR      = "irradiation"
RELAXED_DATA   = os.path.join(RELAX_DIR, "relaxed.data")
DUMP_OUT       = os.path.join(IRRAD_DIR, "overlap_cascade.lammpstrj")

USE_ELECTRONIC_STOPPING = True
STOPPING_FILE           = "Cr_stopping.txt"
STOPPING_ECUT           = 10.0

# ============================================================================
# Core functions (kept unchanged)
# ============================================================================
def run(inp, log):
    subprocess.check_call(LAMMPS_CMD.format(inp, log), shell=True)

def read_box(path):
    box = {}
    with open(path) as f:
        for line in f:
            for ax in 'xyz':
                if f'{ax}lo {ax}hi' in line:
                    p = line.split()
                    box[ax] = (float(p[0]), float(p[1]))
    return box

def pick_pka_and_shift(path, zoff, seed_offset):
    box = read_box(path)
    Lx, Ly, Lz = box['x'][1]-box['x'][0], box['y'][1]-box['y'][0], box['z'][1]-box['z'][0]
    cx, cy, cz = box['x'][0]+Lx/2, box['y'][0]+Ly/2, box['z'][0]+Lz/2

    rng = random.Random(seed_offset)
    target_x = box['x'][0] + rng.uniform(0, Lx)
    target_y = box['y'][0] + rng.uniform(0, Ly)
    target_z = cz + zoff

    with open(path) as f: lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('Atoms')) + 2

    pka, mind = None, float('inf')
    pka_orig_x = pka_orig_y = 0.0

    for line in lines[start:]:
        p = line.split()
        if not p or not p[0].isdigit(): break
        x, y, z = float(p[2]), float(p[3]), float(p[4])

        dx, dy, dz = x-target_x, y-target_y, z-target_z
        dx -= round(dx/Lx)*Lx; dy -= round(dy/Ly)*Ly; dz -= round(dz/Lz)*Lz
        d = dx*dx + dy*dy + dz*dz
        if d < mind:
            mind, pka = d, int(p[0])
            pka_orig_x, pka_orig_y = x, y

    return pka, cx - pka_orig_x, cy - pka_orig_y

def pka_v(E, mass, d):
    v = 98.2269 * math.sqrt(2*E*1000/mass)
    n = math.sqrt(sum(x*x for x in d))
    return [v*x/n for x in d]

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

def make_inp(out, data, pid, shift_dx, shift_dy, vx, vy, vz, seed, box):
    xlo, xhi = box['x']; ylo, yhi = box['y']; zlo, zhi = box['z']
    final_path = os.path.join(os.path.dirname(out), "final.data")

    if USE_ELECTRONIC_STOPPING and os.path.exists(STOPPING_FILE):
        e_stop_fix = f"fix e_stop all electron/stopping {STOPPING_ECUT} {STOPPING_FILE}"
        e_stop_unfix = "unfix e_stop"
        thermo_style_phase1 = "thermo_style custom step time dt temp pe ke etotal press f_e_stop"
    else:
        e_stop_fix = "# electronic stopping disabled or file not found"
        e_stop_unfix = ""
        thermo_style_phase1 = "thermo_style custom step time dt temp pe ke etotal press"

    script = f"""units metal
dimension 3
boundary p p p
atom_style atomic
atom_modify map array
read_data {data}

pair_style eam/fs
pair_coeff * * {POTENTIAL} {ELEMENTS}

displace_atoms all move {shift_dx:.6f} {shift_dy:.6f} 0.0 units box

variable pka_id equal {pid}
group pka_atom id ${{pka_id}}
velocity pka_atom set {vx:.4f} {vy:.4f} {vz:.4f} units box

fix remove_drift all momentum 10 linear 1 1 1

region core_zone block {xlo+THICK:.6f} {xhi-THICK:.6f} {ylo+THICK:.6f} {yhi-THICK:.6f} {zlo+THICK:.6f} {zhi-THICK:.6f} units box side out
group thermo_atoms dynamic all region core_zone every 1

reset_timestep 0
fix g_nve all nve
fix langevin_thermo thermo_atoms langevin 300.0 300.0 1.0 {seed}
fix dt_var all dt/reset 1 1e-9 1e-3 {0.005*LAT_CONST} units box

{e_stop_fix}

thermo 1000
{thermo_style_phase1}
variable sim_time equal time
variable time_diff equal "v_sim_time - 20.0"

fix halt_cascade all halt 10 v_time_diff > 0.0 error continue
run 100000000

unfix g_nve
unfix langevin_thermo
unfix dt_var
unfix halt_cascade
unfix remove_drift
{e_stop_unfix}

reset_timestep 0
timestep 0.001
thermo_style custom step time dt temp pe ke etotal press
fix g_npt all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run 10000

displace_atoms all move {-shift_dx:.6f} {-shift_dy:.6f} 0.0 units box

write_dump all custom {DUMP_OUT} id type xu yu zu modify append yes
write_data {final_path}
"""
    with open(out, 'w') as f: f.write(script)

# ============================================================================
# Main workflow
# ============================================================================
if __name__ == "__main__":
    random.seed(SHIFT_SEED)

    os.makedirs(RELAX_DIR, exist_ok=True)
    os.makedirs(IRRAD_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 0. Initial relaxation (if not done yet)
    # ------------------------------------------------------------------------
    if not os.path.exists(RELAXED_DATA):
        print(f">>> {RELAXED_DATA} not found, starting initial relaxation...")
        if not os.path.exists(INIT_DATA):
            raise FileNotFoundError(f"{INIT_DATA} not found, cannot run initial relaxation!")

        relax_inp = os.path.join(RELAX_DIR, "in.relax")
        relax_log = os.path.join(RELAX_DIR, "relax.log")
        with open(relax_inp, "w") as f:
            f.write(f"""units metal
dimension 3
boundary p p p
atom_style atomic
read_data {INIT_DATA}
pair_style eam/fs
pair_coeff * * {POTENTIAL} {ELEMENTS}

thermo 1000
minimize 1e-10 1e-10 10000 10000

reset_timestep 0
timestep 0.001
velocity all create 300.0 12345 dist gaussian
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
run 20000
write_data {RELAXED_DATA}
""")
        run(relax_inp, relax_log)
        print(f">>> Initial relaxation done -> {RELAXED_DATA}\n")

    # ------------------------------------------------------------------------
    # 1. Detect existing progress and determine the starting point
    # ------------------------------------------------------------------------
    prev_data_path = os.path.join(IRRAD_DIR, "prev.data")
    final_data_path = os.path.join(IRRAD_DIR, "final.data")

    if START_CASC > 1:
        # Continue mode: look for the output of run START_CASC-1
        if os.path.exists(prev_data_path):
            prev = prev_data_path
            print(f">>> prev.data detected, continuing from run {START_CASC}")
        elif os.path.exists(final_data_path):
            # Previous run may have been interrupted, final.data still exists, rename to prev.data
            os.rename(final_data_path, prev_data_path)
            prev = prev_data_path
            print(f">>> final.data detected, renamed to prev.data, continuing from run {START_CASC}")
        else:
            raise FileNotFoundError(
                f"START_CASC={START_CASC}, but prev.data or final.data not found!\n"
                f"Please make sure the output file of run {START_CASC-1} exists."
            )
    else:
        # Fresh start
        prev = RELAXED_DATA

        # Write frame 0
        init_inp = os.path.join(IRRAD_DIR, "in.init_dump")
        with open(init_inp, "w") as f:
            f.write(f"units metal\natom_style atomic\nread_data {RELAXED_DATA}\n"
                    f"write_dump all custom {DUMP_OUT} id type xu yu zu modify append no\n")
        run(init_inp, os.path.join(IRRAD_DIR, "init_dump.log"))
        print(f">>> Frame 0 -> {DUMP_OUT}")

    box = read_box(prev)

    e_stop_status = "off"
    if USE_ELECTRONIC_STOPPING:
        if os.path.exists(STOPPING_FILE):
            e_stop_status = f"on (Ecut={STOPPING_ECUT}eV, file={STOPPING_FILE})"
        else:
            print(f"!!! Warning: {STOPPING_FILE} not found, electronic stopping will be forcibly disabled for this run !!!")

    # ------------------------------------------------------------------------
    # 2. Generate direction set (generate all of them, use only the needed range)
    # ------------------------------------------------------------------------
    all_directions = generate_pka_directions(N_CASC, TILT_MIN_DEG, TILT_MAX_DEG, REF_DIR, mode=DIRECTION_MODE)

    # Check whether the first directions are consistent with previous runs (optional, for verification)
    print(f">>> Direction set generated, total {len(all_directions)}")
    print(f">>> Will use directions from {START_CASC} to {N_CASC}")

    # ------------------------------------------------------------------------
    # 3. Cascade main loop
    # ------------------------------------------------------------------------
    summary = ["cascade_no,pka_id,shift_dx,shift_dy,dir_x,dir_y,dir_z,vx,vy,vz,tilt_angle(deg),e_stopping,seed\n"]

    for i in range(START_CASC, N_CASC+1):
        current_dir = all_directions[i-1]
        dir_x, dir_y, dir_z = current_dir
        vx, vy, vz = pka_v(PKA_E, MASS_PKA, current_dir)
        actual_angle = math.degrees(math.acos(np.dot(current_dir, np.array(REF_DIR)/np.linalg.norm(REF_DIR))))

        print(f"\n>>> Cascade {i}/{N_CASC}")
        print(f">>> Incident direction: [{dir_x:+.4f}, {dir_y:+.4f}, {dir_z:+.4f}] (offset from reference: {actual_angle:.2f} deg)")

        seed = 98765 + i*111
        seed_offset = SHIFT_SEED + i*137

        pid, shift_dx, shift_dy = pick_pka_and_shift(prev, PKA_ZOFF, seed_offset)

        cascade_inp = os.path.join(IRRAD_DIR, f"in.cascade_{i:02d}")
        cascade_log = os.path.join(IRRAD_DIR, f"cascade_{i:02d}.log")
        make_inp(cascade_inp, prev, pid, shift_dx, shift_dy, vx, vy, vz, seed, box)
        run(cascade_inp, cascade_log)
        print(f">>> Frame {i} -> {DUMP_OUT}")

        # File handoff
        final_data = os.path.join(IRRAD_DIR, "final.data")
        os.replace(final_data, prev_data_path)
        prev = prev_data_path

        summary.append(f"{i},{pid},{shift_dx:.4f},{shift_dy:.4f},{dir_x:.6f},{dir_y:.6f},{dir_z:.6f},{vx:.4f},{vy:.4f},{vz:.4f},{actual_angle:.2f},{e_stop_status},{seed}\n")

    # ------------------------------------------------------------------------
    # 4. Result summary
    # ------------------------------------------------------------------------
    summary_path = os.path.join(IRRAD_DIR, "cascade_summary.csv")

    # If a summary from previous runs already exists, merge it
    if os.path.exists(summary_path) and START_CASC > 1:
        with open(summary_path, "r") as f:
            existing_summary = f.readlines()
        # Drop the header of the new summary, then merge
        summary = existing_summary + summary[1:]

    with open(summary_path, "w") as f:
        f.writelines(summary)

    print(f"\n>>> Done! Unified output: {DUMP_OUT}")
    print(f">>> Summary: {summary_path}")
