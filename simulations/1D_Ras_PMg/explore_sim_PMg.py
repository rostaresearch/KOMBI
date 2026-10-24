#KOMBI driver for the Ras Mg2+-Pi unbinding (1D_PMg).
#
# workflow:
#    propagate the system for a number of steps under the current bias (propagation 0: tiny
#    random bias around the start), record the Mg-P distance s at every step,
#    DHAM over all propagations so far -> working transition matrix,
#    choose the next bias by minimising the model MFPT from the current state to the state
#    closest to 8 A, repeat until s first reaches 8 A.
#
# bias and bookkeeping:
#    - the bias acts through ONE Mg-P bond;
#    - the optimiser and DHAM use the applied Gaussian a*exp(-(s-b)^2/(2c^2)) (util.gaussian);
#    - DHAM receives per-Gaussian (a, b, c) triples;
#    - all parameters are kept in kcal/mol and A and converted once for OpenMM;
#    - first passage = first step with s >= target; total = (n_props - 1) * L + index + 1;
#    - the optimiser starts from DHAM's own bin of the current position.
# checks and outputs:
#    - a start-up guard: the energy and force of the bias force group (OpenMM) are compared with
#      the Python bias used by DHAM and the optimiser at several distances and parameter sets,
#      and the force-group layout is checked; the run stops if anything differs;
#    - the applied bias (OpenMM, own force group) is written for every step next to the Python
#      value and checked at every step;
#    - per-run seeds, checkpoint/resume (exit 42 at the wall-time budget), --pbc (PME), outputs
#      per propagation (CV, bias, DHAM model, decision), DCD of the solute and of the full system,
#      a log of the Mg2+ first shell. No plotting.
#
# usage: python explore_sim_PMg.py --run-index 0 --outdir runs/run_000 --system-dir ../system
#        python explore_sim_PMg.py --check-only ...   (guard + a short timed MD test, then exit)

import argparse
import json
import os
import sys
import time
import numpy as np
import openmm as omm
import openmm.app as omm_app
import openmm.unit as unit
from openmm.app import CharmmPsfFile, CharmmParameterSet, PDBFile, CutoffNonPeriodic, PME, HBonds

import config
import mg_shell
from util import *
from dham import DHAM, count_transitions_one

#force groups. CharmmPsfFile.createSystem already spreads the CHARMM terms over groups 0-6
#(angles 1, torsions 2, bonds 3, ..., nonbonded 6), so the KOMBI forces use the top two groups
#and the integrator uses every group except CV_GROUP.
BIAS_GROUP = 30           #the KOMBI bias, integrated
CV_GROUP = 31             #CustomBondForce "r": the Mg-P distance read back as an energy, NOT integrated
INTEGRATION_GROUPS = set(range(32)) - {CV_GROUP}
KJ_PER_KCAL = 4.184
EXIT_OK, EXIT_GUARD, EXIT_BIAS_MISMATCH, EXIT_WALL = 0, 3, 4, 42

def log(*args):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

###############################################
# bias: units, layout, OpenMM force
###############################################

def to_openmm_units(gaussian_params):
    """[a(N) kcal/mol, b(N) A, c(N) A] -> a (kJ/mol), b (nm), c (nm)"""
    n = config.num_gaussian
    p = np.asarray(gaussian_params, dtype=float)
    return p[:n] * KJ_PER_KCAL, p[n:2*n] / 10.0, p[2*n:3*n] / 10.0

def params_to_triples(gaussian_params):
    """[a(N), b(N), c(N)] -> (N, 3) per-Gaussian (a, b, c), the layout DHAM reads"""
    return np.asarray(gaussian_params, dtype=float).reshape(3, config.num_gaussian).T

def bias_kcal(s, gaussian_params):
    """the bias the optimiser and DHAM assume, kcal/mol, s in A"""
    return get_total_bias(np.asarray(s, dtype=float), np.asarray(gaussian_params, dtype=float), num_gaussians=config.num_gaussian)

def random_initial_bias(rng, initial_position=config.initial_position, num_gaussians=config.num_gaussian):
    #returns random a,b,c for the gaussian functions around the initial position, as run:
    #0.01 kcal/mol each, centres within 0.2 A of 3.7 A, widths 1-5 A. kcal/mol and A.
    a = np.ones(num_gaussians) * config.initial_amp
    b = rng.uniform(initial_position-0.2, initial_position+0.2, num_gaussians)
    c = rng.uniform(1, 5.0, num_gaussians)
    return np.concatenate((a,b,c), axis=None)

def add_bias(system, gaussian_params, pbc=False):
    """one CustomBondForce holding the sum of the Gaussians and ONE Mg-P bond, in its own force group"""
    a, b, c = to_openmm_units(gaussian_params)
    n = config.num_gaussian
    potential = ' + '.join(f'a{i}*exp(-(r-b{i})^2/(2*c{i}^2))' for i in range(n))
    custom_bias = omm.CustomBondForce(potential)
    for i in range(n):
        custom_bias.addGlobalParameter(f'a{i}', a[i])
        custom_bias.addGlobalParameter(f'b{i}', b[i])
        custom_bias.addGlobalParameter(f'c{i}', c[i])
    custom_bias.addBond(config.mg_index, config.p_index)  #once: the expression already sums all Gaussians
    custom_bias.setUsesPeriodicBoundaryConditions(pbc)
    custom_bias.setForceGroup(BIAS_GROUP)
    custom_bias.setName('KOMBI_bias')
    system.addForce(custom_bias)
    return custom_bias

def add_cv_force(system, pbc=False):
    """Mg-P distance as an energy (kJ/mol == nm) in a group the integrator does not use.
    cv_on is set to 0 only while OpenMM draws velocities: setVelocitiesToTemperature evaluates
    every force group (checked with OpenMM 8.2), including excluded ones."""
    cv = omm.CustomBondForce('cv_on*r')
    cv.addGlobalParameter('cv_on', 1.0)
    cv.addBond(config.mg_index, config.p_index)
    cv.setUsesPeriodicBoundaryConditions(pbc)
    cv.setForceGroup(CV_GROUP)
    cv.setName('CV_MgP_distance')
    system.addForce(cv)
    return cv

def set_bias(context, gaussian_params):
    """the only place where parameters reach OpenMM after start-up (kcal/mol, A -> kJ/mol, nm)"""
    a, b, c = to_openmm_units(gaussian_params)
    for j in range(config.num_gaussian):
        context.setParameter(f'a{j}', a[j])
        context.setParameter(f'b{j}', b[j])
        context.setParameter(f'c{j}', c[j])

def read_cv_and_bias(context):
    """s (A) and the applied bias (kcal/mol) from the two force groups"""
    r_nm = context.getState(getEnergy=True, groups=1 << CV_GROUP).getPotentialEnergy()._value
    e_kj = context.getState(getEnergy=True, groups=1 << BIAS_GROUP).getPotentialEnergy()._value
    return r_nm * 10.0, e_kj / KJ_PER_KCAL

def bias_tolerance(v):
    return config.bias_check_tol + config.bias_check_rtol * np.abs(v)

###############################################
# start-up guard
###############################################

GUARD_DISTANCES = (2.8, 3.0, 3.3, 3.7, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0)  #A

def guard_param_sets(rng):
    """parameter sets for the guard: a fixed one with large, distinct amplitudes and widths
    (a bias applied ten times, a sqrt(2)-wide Gaussian or mis-ordered (a, b, c) each change it
    by far more than the tolerance) and a
    random one within the optimiser's bounds"""
    n = config.num_gaussian
    fixed = np.concatenate((np.linspace(1.0, 0.1, n), np.linspace(3.0, 8.0, n), np.linspace(0.7, 2.0, n)))
    rand = np.concatenate((rng.uniform(*config.amp_bounds, n), rng.uniform(3.0, 8.0, n), rng.uniform(*config.width_bounds, n)))
    return [("fixed", fixed), ("random", rand)]

def check_force_layout(simulation):
    """bias and CV forces: one bond each between Mg and P, each alone in its group; the integrator
    uses the bias group and every group holding a physical force, and not the CV group"""
    problems = []
    system = simulation.system
    groups = simulation.integrator.getIntegrationForceGroups()
    n_bias = n_cv = 0
    for k, f in enumerate(system.getForces()):
        g = f.getForceGroup()
        name = f.getName()
        if name == 'KOMBI_bias':
            n_bias += 1
            if g != BIAS_GROUP: problems.append(f"bias force in group {g}")
            if f.getNumBonds() != 1: problems.append(f"bias force has {f.getNumBonds()} bonds")
            if tuple(f.getBondParameters(0)[:2]) != (config.mg_index, config.p_index): problems.append("bias bond atoms")
        elif name == 'CV_MgP_distance':
            n_cv += 1
            if g != CV_GROUP: problems.append(f"CV force in group {g}")
            if f.getNumBonds() != 1: problems.append(f"CV force has {f.getNumBonds()} bonds")
        elif g in (BIAS_GROUP, CV_GROUP):
            problems.append(f"force {k} ({name}) shares group {g} with a KOMBI force")
        elif not groups & (1 << g):
            problems.append(f"force {k} ({name}) in group {g} is not integrated")
    if (n_bias, n_cv) != (1, 1): problems.append(f"{n_bias} bias forces and {n_cv} CV forces")
    if groups & (1 << CV_GROUP): problems.append("integrator integrates the CV group")
    if not groups & (1 << BIAS_GROUP): problems.append("integrator does not integrate the bias group")
    return problems

def bias_guard(simulation, param_sets, restore_params):
    """move P along the Mg->P direction to each guard distance and compare OpenMM's bias energy and
    force (bias group only) and the CV readback with the Python bias. Positions and the bias
    parameters are restored afterwards."""
    ctx = simulation.context
    pos0 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer).copy()
    mg = pos0[config.mg_index]
    uvec = pos0[config.p_index] - mg
    uvec = uvec / np.linalg.norm(uvec)
    others = np.ones(len(pos0), dtype=bool); others[[config.mg_index, config.p_index]] = False
    records, ok = [], True
    for name, params in param_sets:
        set_bias(ctx, params)
        #the bias as the two consumers evaluate it: DHAM (util.gaussian on per-Gaussian triples from
        #params_to_triples) and the optimiser (util.optimiser_total_bias on the [a, b, c] layout)
        d = DHAM(params_to_triples(params)[None], num_bins=len(GUARD_DISTANCES))
        d.qspace = np.array(GUARD_DISTANCES, dtype=float)
        v_dham_all = d.bias_on_grid()[0]
        v_opt_all = optimiser_total_bias(np.array(GUARD_DISTANCES, dtype=float), np.asarray(params, dtype=float), config.num_gaussian)
        for ir, r in enumerate(GUARD_DISTANCES):
            pos = pos0.copy()
            pos[config.p_index] = mg + uvec * (r / 10.0)
            ctx.setPositions(pos)
            st = ctx.getState(getEnergy=True, getForces=True, groups=1 << BIAS_GROUP)
            e_omm = st.getPotentialEnergy()._value / KJ_PER_KCAL
            F = st.getForces(asNumpy=True)._value
            f_p = float(F[config.p_index] @ uvec) / (KJ_PER_KCAL * 10.0)    #kcal/mol/A, = -dV/dr
            f_mg = float(F[config.mg_index] @ uvec) / (KJ_PER_KCAL * 10.0)  #= +dV/dr
            f_other = float(np.abs(F[others]).max())
            r_cv = ctx.getState(getEnergy=True, groups=1 << CV_GROUP).getPotentialEnergy()._value * 10.0
            v_py = float(bias_kcal(np.array([r]), params)[0])
            dv_py = float(get_total_bias_derivative(np.array([r]), np.asarray(params), num_gaussians=config.num_gaussian)[0])
            v_dham, v_opt = float(v_dham_all[ir]), float(v_opt_all[ir])
            rec = {"set": name, "r_A": r, "r_cv_A": r_cv, "V_openmm": e_omm, "V_python": v_py,
                   "V_dham": v_dham, "V_optimiser": v_opt,
                   "F_P_openmm": f_p, "minus_dVdr_python": -dv_py, "F_Mg_openmm": f_mg, "max_F_other": f_other}
            rec["pass"] = bool(abs(e_omm - v_py) <= bias_tolerance(v_py)
                               and abs(e_omm - v_dham) <= bias_tolerance(v_py)
                               and abs(e_omm - v_opt) <= bias_tolerance(v_py)
                               and abs(f_p + dv_py) <= bias_tolerance(dv_py)
                               and abs(f_mg - dv_py) <= bias_tolerance(dv_py)
                               and f_other <= 1e-6
                               and abs(r_cv - r) <= 1e-4)
            ok &= rec["pass"]
            records.append(rec)
    ctx.setPositions(pos0)
    set_bias(ctx, restore_params)
    return ok, records

def run_guard(simulation, rng, restore_params, outdir, tag):
    problems = check_force_layout(simulation)
    ok, records = bias_guard(simulation, guard_param_sets(rng) + [("current", restore_params)], restore_params)
    worst = max(max(abs(r["V_openmm"] - r[k]) for k in ("V_python", "V_dham", "V_optimiser")) for r in records)
    res = {"tag": tag, "pass": bool(ok and not problems), "layout_problems": problems,
           "max_abs_dV_kcal": worst, "n_points": len(records), "records": records,
           "openmm_version": omm.__version__,
           "platform": simulation.context.getPlatform().getName()}
    with open(os.path.join(outdir, f"guard_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"guard ({tag}): {'PASS' if res['pass'] else 'FAIL'}; {len(records)} points, max |V_openmm - V_python, V_dham, V_optimiser| = {worst:.2e} kcal/mol; layout problems: {problems}")
    return res["pass"]

###############################################
# system
###############################################

def read_params(filename):
    #toppar.str lists parameter files relative to its own folder
    here = os.path.dirname(os.path.abspath(filename))
    parFiles = ()
    for line in open(filename, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0: parFiles += ( parfile if os.path.isabs(parfile) else os.path.join(here, parfile), )

    params = CharmmParameterSet( *parFiles )
    return params

def check_atoms(psf):
    mg = psf.atom_list[config.mg_index]
    p = psf.atom_list[config.p_index]
    assert (mg.name, mg.residue.resname) == ("MG", "MG"), f"atom {config.mg_index} is {mg.residue.resname} {mg.name}, expected MG MG"
    assert (p.name, p.residue.resname) == ("P", "H2PO4"), f"atom {config.p_index} is {p.residue.resname} {p.name}, expected H2PO4 P"

def build_system(psf, params, pbc):
    if pbc:
        return psf.createSystem(params, nonbondedMethod=PME, nonbondedCutoff=config.pme_cutoff*unit.nanometer,
                                switchDistance=config.pme_switch*unit.nanometer, constraints=HBonds)
    #as run
    return psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=config.nonbonded_cutoff*unit.nanometer, constraints=HBonds)

def solute_indices(psf):
    """everything except the bulk water segment SOLV (crystal waters WATA/WATB and ions kept)"""
    return np.array([i for i, a in enumerate(psf.atom_list) if a.system != "SOLV"])

def subset_topology(topology, keep):
    keep = set(int(i) for i in keep)
    mod = omm_app.Modeller(topology, [omm.Vec3(0, 0, 0)] * topology.getNumAtoms())
    mod.delete([a for a in mod.topology.atoms() if a.index not in keep])
    return mod.topology

###############################################
# MD propagation
###############################################

def propagate(simulation, prop_index, gaussian_params, outdir, steps, io, tag="prop", first_step=None):
    """one propagation; s and the applied bias every step, frames and the Mg shell at the set strides.
    tag "prop" for KOMBI propagations, "post" for the unbiased continuation after the first passage.
    returns s (A), V_openmm and V_python (kcal/mol), and the number of steps where they differ."""
    ctx = simulation.context
    integ = simulation.integrator
    s = np.empty(steps)
    v_omm = np.empty(steps)
    dt_ps = config.stepsize.value_in_unit(unit.picosecond)
    full_stride = max(1, int(round(config.dcd_full_ps / dt_ps)))
    solute_stride = max(1, int(round(config.dcd_solute_ps / dt_ps)))
    shell_stride = max(1, int(round(config.shell_log_ps / dt_ps)))
    fh_full = open(os.path.join(outdir, "traj", f"full_{tag}_{prop_index:04d}.dcd"), "wb")
    fh_sol = open(os.path.join(outdir, "traj", f"solute_{tag}_{prop_index:04d}.dcd"), "wb")
    if first_step is None:
        first_step = prop_index * steps
    dcd_full = omm_app.DCDFile(fh_full, simulation.topology, config.stepsize, first_step, full_stride)
    dcd_sol = omm_app.DCDFile(fh_sol, io["solute_top"], config.stepsize, first_step, solute_stride)
    shell_rows = []
    pbc = io["pbc"]
    for i in range(steps):
        integ.step(config.dcdfreq_mfpt)
        s[i], v_omm[i] = read_cv_and_bias(ctx)
        n = i + 1
        if n % full_stride == 0 or n % solute_stride == 0 or n % shell_stride == 0:
            #continuous (unwrapped) coordinates: wrapping molecule by molecule would split Ras from GAP;
            #the shell distances below use the minimum image
            st = ctx.getState(getPositions=True)
            pos = st.getPositions(asNumpy=True)
            box = st.getPeriodicBoxVectors() if pbc else None
            if n % full_stride == 0:
                dcd_full.writeModel(pos, periodicBoxVectors=box)
            if n % solute_stride == 0:
                dcd_sol.writeModel(pos[io["solute_idx"]], periodicBoxVectors=box)
            if n % shell_stride == 0:
                L = np.array([box[k][k].value_in_unit(unit.angstrom) for k in range(3)]) if pbc else None
                shell_rows.append((first_step + n, mg_shell.shell_row(pos.value_in_unit(unit.angstrom), io["shell"], L)))
    fh_full.close(); fh_sol.close()
    v_py = bias_kcal(s, gaussian_params)
    bad = ~(np.abs(v_omm - v_py) <= bias_tolerance(v_py)) | ~np.isfinite(s)   #NaN counts as a mismatch
    np.savez(os.path.join(outdir, "cv", f"{tag}_{prop_index:04d}.npz"), s_A=s, V_openmm_kcal=v_omm, V_python_kcal=v_py,
             first_step=first_step, dt_ps=dt_ps, params=np.asarray(gaussian_params))
    #one shell file per propagation, so a propagation that is rerun after a crash overwrites its rows
    with open(os.path.join(outdir, "shell", f"{tag}_{prop_index:04d}.csv"), "w") as fh:
        fh.write(mg_shell.HEADER + "\n")
        for step, row in shell_rows:
            fh.write(mg_shell.format_row(step, row))
    return s, v_omm, v_py, int(bad.sum())

def DHAM_it(Ntr_list, global_gaussian_params, T=300, lagtime=1, num_bins=150, prop_index=0, time_tag=None):
    """
    DHAM over all propagations so far, from their per-propagation transition counts.
    output: DHAM qspace, F_M (free energy, kcal/mol), the Markov matrix M (column-stochastic), diagnostics
    """
    d = DHAM(global_gaussian_params, num_bins=num_bins)
    d.setup(None, T, prop_index=prop_index, time_tag=time_tag, Ntr=np.stack(Ntr_list))
    d.lagtime = lagtime
    results = d.run(biased = True, plot=False, use_dynamic_bins=config.use_dynamic_bins)
    return results + (d.diagnostics,)

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

###############################################
# bookkeeping
###############################################

def first_passage_steps(n_props, reach, propagation_step):
    """steps to the first visit of the target: all earlier propagations plus the per-step
    index of the first visit in the hitting propagation, plus one (index 0 is after step 1)"""
    return (n_props - 1)*propagation_step + reach + 1

def write_first_passage(outdir, n_props, reach, L):
    """first_passage.json: written when the target is first reached (or, censored, when the cap is hit)"""
    dt_ps = config.stepsize.value_in_unit(unit.picosecond)
    total_steps = first_passage_steps(n_props, reach, L) if reach is not None else None
    fp = {"reached": reach is not None, "target_A": config.target_distance, "n_props": n_props, "prop_steps": L,
          "reach_index_in_last_prop": reach, "total_steps": total_steps,
          "first_passage_ps": None if total_steps is None else total_steps * dt_ps,
          "simulated_ps": n_props * L * dt_ps}
    write_json_atomic(os.path.join(outdir, "first_passage.json"), fp)
    log("first passage:", fp)

def zero_amplitude(gaussian_params):
    """the same Gaussians with every amplitude 0 (bias switched off, parameters kept valid)"""
    p = np.array(gaussian_params, dtype=float)
    p[:config.num_gaussian] = 0.0
    return p

def derived_seed(*keys):
    """a nonzero 31-bit seed for OpenMM (0 would mean 'random')"""
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0] % (2**31 - 2)) + 1

def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)

def params_file(outdir, k):
    return os.path.join(outdir, "params", f"gaussian_param_prop_{k}.txt")

def save_params(outdir, k, gaussian_params):
    np.savetxt(params_file(outdir, k), gaussian_params,
               header=f"propagation {k}: [a({config.num_gaussian}) kcal/mol, b({config.num_gaussian}) A, c({config.num_gaussian}) A]; V(s) = sum a exp(-(s-b)^2/(2c^2)), applied once")

def load_params(outdir, k):
    return np.loadtxt(params_file(outdir, k))

def parse_args():
    ap = argparse.ArgumentParser(description="Ras KOMBI run (one run per process)")
    ap.add_argument("--run-index", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None, help="default 20261011 + 1000 * run_index")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--system-dir", default=".", help="folder with step3_input.psf, traj_0.restart.pdb, toppar.str, toppar/")
    ap.add_argument("--start-state", default=None, help="--pbc: equilibrated OpenMM state XML (positions + box)")
    ap.add_argument("--pbc", action="store_true")
    ap.add_argument("--pme-cutoff", type=float, default=config.pme_cutoff, help="--pbc: real-space cutoff, nm")
    ap.add_argument("--pme-switch", type=float, default=config.pme_switch, help="--pbc: LJ switch distance, nm")
    ap.add_argument("--platform", default=config.platform_name)
    ap.add_argument("--precision", default=config.precision)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prop-steps", type=int, default=config.propagation_step)
    ap.add_argument("--max-prop", type=int, default=config.num_propagation, help="last propagation index allowed")
    ap.add_argument("--amp-max", type=float, default=config.amp_bounds[1], help="upper bound of each Gaussian amplitude, kcal/mol")
    ap.add_argument("--target", type=float, default=config.target_distance, help="first-passage target, A (as run: 8)")
    ap.add_argument("--post-ns", type=float, default=0.0,
                    help="after the first passage, continue this long with the bias switched off (Mg2+ shell relaxation, "
                         "e.g. Thr35 release, KOMBI SI.5); 0 = stop at the first passage, as run")
    ap.add_argument("--wall-h", type=float, default=1e9, help="checkpoint and exit 42 before this many hours")
    ap.add_argument("--check-only", action="store_true", help="guard + a short timed MD test, then exit")
    ap.add_argument("--check-steps", type=int, default=2000)
    return ap.parse_args()

###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

def main():
    t_start = time.time()
    args = parse_args()
    config.propagation_step = args.prop_steps
    config.num_propagation = args.max_prop
    config.amp_bounds = (config.amp_bounds[0], args.amp_max)
    config.pbc = args.pbc
    config.pme_cutoff, config.pme_switch = args.pme_cutoff, args.pme_switch
    config.target_distance = args.target
    if config.dham_lag_steps != 1:
        sys.exit("dham_lag_steps != 1: the optimiser's direct_bias tilts only nearest neighbours; a longer lag needs an all-pairs tilt")
    seed = args.seed if args.seed is not None else 20261011 + 1000 * args.run_index
    outdir = os.path.abspath(args.outdir)
    for sub in ("params", "cv", "decisions", "traj", "shell"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    sysdir = os.path.abspath(args.system_dir)
    L = config.propagation_step

    used = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v))
            for k, v in vars(config).items() if not k.startswith("_") and k not in ("np", "unit", "time")}
    used.update({"run_index": args.run_index, "seed": seed, "args": vars(args), "openmm_version": omm.__version__,
                 "qspace": list(map(float, config.qspace)), "post_ns": args.post_ns})
    progress_path = os.path.join(outdir, "progress.json")
    progress = json.load(open(progress_path)) if os.path.exists(progress_path) else None
    if args.check_only and progress is not None:
        sys.exit("--check-only needs a fresh output directory (this one holds a run)")
    if progress is not None and progress.get("done"):
        log("run already finished:", progress); return EXIT_OK
    if progress is None:
        write_json_atomic(os.path.join(outdir, "config_used.json"), used)
    else:
        old = json.load(open(os.path.join(outdir, "config_used.json")))
        for key in ("seed", "propagation_step", "amp_bounds", "pbc", "pme_cutoff", "pme_switch", "num_gaussian", "dham_lag_steps", "target_distance", "post_ns"):
            assert json.loads(json.dumps(used[key])) == old[key], f"resume with a different {key}: {old[key]} -> {used[key]}"

    #MD setup
    psf = CharmmPsfFile(os.path.join(sysdir, config.psf_file))
    check_atoms(psf)
    if args.pbc:
        if args.start_state is None:
            sys.exit("--pbc needs --start-state (equilibrated periodic state XML)")
        start = omm.XmlSerializer.deserialize(open(args.start_state).read())
        a, b, c = start.getPeriodicBoxVectors()
        psf.setBox(a[0], b[1], c[2])
        start_positions, start_box = start.getPositions(), (a, b, c)
    else:
        pdb = PDBFile(os.path.join(sysdir, config.pdb_file))
        assert pdb.topology.getNumAtoms() == psf.topology.getNumAtoms(), "pdb and psf atom counts differ"
        start_positions, start_box = pdb.positions, None
    params = read_params(os.path.join(sysdir, config.toppar_file))
    system = build_system(psf, params, args.pbc)
    rng0 = np.random.default_rng([seed, 0])
    params0 = random_initial_bias(rng0)
    add_bias(system, params0, pbc=args.pbc)
    add_cv_force(system, pbc=args.pbc)
    integrator = omm.LangevinIntegrator(config.T*unit.kelvin, config.fricCoef, config.stepsize)
    integrator.setIntegrationForceGroups(INTEGRATION_GROUPS)
    n_resume = 0 if progress is None else progress.get("n_resume", 0) + 1
    integrator.setRandomNumberSeed(derived_seed(seed, 1, n_resume))
    platform = omm.Platform.getPlatformByName(args.platform)
    props = {}
    if args.platform in ("CUDA", "OpenCL"):
        props["Precision"] = args.precision
        if args.device is not None: props["DeviceIndex"] = str(args.device)
    simulation = omm_app.Simulation(psf.topology, system, integrator, platform, props)
    plat = simulation.context.getPlatform()
    log(f"OpenMM {omm.__version__}, platform {plat.getName()}",
        {k: plat.getPropertyValue(simulation.context, k) for k in plat.getPropertyNames() if k in ("Precision", "DeviceIndex", "DeviceName", "Threads")})
    log(f"run {args.run_index} seed {seed} pbc={args.pbc} L={L} steps max_prop={config.num_propagation} amp_bounds={config.amp_bounds} lag={config.dham_lag_steps}")

    #output helpers
    sol = solute_indices(psf)
    io = {"pbc": args.pbc, "solute_idx": sol, "solute_top": subset_topology(psf.topology, sol),
          "shell": mg_shell.shell_sets(psf)}
    log("Mg shell log: oxygens by kind", io["shell"]["counts"])
    if progress is None:
        with open(os.path.join(outdir, "solute.pdb"), "w") as fh:
            PDBFile.writeFile(io["solute_top"], np.asarray(start_positions.value_in_unit(unit.nanometer))[sol] * unit.nanometer, fh)

    rng_guard = np.random.default_rng([seed, 99])
    if progress is None:
        if start_box is not None:
            simulation.context.setPeriodicBoxVectors(*start_box)
        simulation.context.setPositions(start_positions)
        if not run_guard(simulation, rng_guard, params0, outdir, "start"):
            log("bias guard FAILED: OpenMM does not apply the bias the optimiser assumes. Stopping."); return EXIT_GUARD
        simulation.context.setParameter('cv_on', 0.0)
        simulation.context.setVelocitiesToTemperature(config.T*unit.kelvin, derived_seed(seed, 2))
        simulation.context.setParameter('cv_on', 1.0)
        i_prop = 0
        cv_s, Ntr_list, global_params = [], [], []
        set_bias(simulation.context, params0)
        save_params(outdir, 0, params0)
    elif progress.get("phase") == "post":
        #resume inside the unbiased continuation after the first passage
        simulation.loadState(os.path.join(outdir, progress["state"]))
        i_prop = progress["next_prop"]
        expect = (i_prop + progress["post_next"]) * L
        assert simulation.context.getStepCount() == expect, \
            f"checkpoint {progress['state']} is at step {simulation.context.getStepCount()}, expected {expect}"
        if not run_guard(simulation, rng_guard, zero_amplitude(load_params(outdir, i_prop - 1)), outdir, f"resume_{n_resume}"):
            log("bias guard FAILED on resume. Stopping."); return EXIT_GUARD
        log(f"resumed in the post-passage continuation, segment {progress['post_next']} (resume {n_resume})")
    else:
        simulation.loadState(os.path.join(outdir, progress["state"]))
        i_prop = progress["next_prop"]
        #the checkpoint must hold the end of propagation next_prop - 1
        assert simulation.context.getStepCount() == i_prop * L, \
            f"checkpoint {progress['state']} is at step {simulation.context.getStepCount()}, expected {i_prop * L}"
        cv_s = [np.load(os.path.join(outdir, "cv", f"prop_{k:04d}.npz"))["s_A"] for k in range(i_prop)]
        Ntr_list = [count_transitions_one(np.digitize(s_k, config.qspace), config.DHAM_num_bins, config.dham_lag_steps) for s_k in cv_s]
        global_params = [params_to_triples(load_params(outdir, k)) for k in range(i_prop)]
        if not run_guard(simulation, rng_guard, load_params(outdir, i_prop), outdir, f"resume_{n_resume}"):
            log("bias guard FAILED on resume. Stopping."); return EXIT_GUARD
        #a first_passage.json from beyond this checkpoint belongs to an abandoned continuation
        #(possible when a hard-killed job's scratch copy is older than its last output)
        fp_path = os.path.join(outdir, "first_passage.json")
        if os.path.exists(fp_path):
            os.replace(fp_path, fp_path + f".abandoned_{n_resume}")
            log(f"moved a stale first_passage.json aside (resuming before propagation {i_prop})")
        log(f"resumed before propagation {i_prop} (resume {n_resume})")

    if args.check_only:
        t0 = time.time()
        steps = args.check_steps
        s, v_omm, v_py, n_bad = propagate(simulation, 9999, params0, outdir, steps, io)
        el = time.time() - t0
        ns_day = steps * config.stepsize.value_in_unit(unit.nanosecond) / el * 86400
        res = {"steps": steps, "seconds": el, "ns_per_day_with_per_step_logging": ns_day, "n_bias_mismatch": n_bad,
               "max_abs_dV_kcal": float(np.abs(v_omm - v_py).max()), "s_first_last": [float(s[0]), float(s[-1])]}
        write_json_atomic(os.path.join(outdir, "check_md.json"), res)
        log("check MD:", res)
        return EXIT_OK if n_bad == 0 else EXIT_BIAS_MISMATCH

    prop_wall = []
    in_post = progress is not None and progress.get("phase") == "post"
    reach = progress.get("reach") if in_post else None
    while reach is None and not in_post:
        if i_prop > config.num_propagation:
            print("Max propagation reached. Exiting.")
            break
        gaussian_params = load_params(outdir, i_prop)
        set_bias(simulation.context, gaussian_params)
        log(f"propagation number {i_prop} STARTING. bias max on qspace {bias_kcal(config.qspace, gaussian_params).max():.2f} kcal/mol")
        t0 = time.time()
        s, v_omm, v_py, n_bad = propagate(simulation, i_prop, gaussian_params, outdir, L, io)
        t_md = time.time() - t0
        if n_bad:
            log(f"applied bias differs from the Python bias at {n_bad} steps of propagation {i_prop}. Stopping.")
            return EXIT_BIAS_MISMATCH
        cv_s.append(s)
        Ntr_list.append(count_transitions_one(np.digitize(s, config.qspace), config.DHAM_num_bins, config.dham_lag_steps))
        global_params.append(params_to_triples(gaussian_params))

        hit = np.nonzero(s >= config.target_distance)[0]
        if hit.size:
            reach = int(hit[0])   #first visit, per-step index
            write_first_passage(outdir, i_prop + 1, reach, L)
        cur_pos = s[-1]

        t0 = time.time()
        dham_qspace, F_M, M, diag = DHAM_it(Ntr_list, np.stack(global_params), T=config.T, lagtime=config.dham_lag_steps,
                                            num_bins=config.DHAM_num_bins, prop_index=i_prop)
        working_MM, working_indices = get_working_MM(M)
        current_qspace = dham_qspace
        last_visited_state = int(np.digitize(cur_pos, current_qspace))      #DHAM's own bin of cur_pos
        final_index = int(np.digitize(config.target_distance, current_qspace) - 1)  #last bin below the target
        closest_index = int(working_indices[np.argmin(np.abs(working_indices - final_index))])
        dec = {"qspace": current_qspace, "F_M": np.real(F_M), "M_working": np.asarray(working_MM, dtype=np.float64),
               "working_indices": working_indices, "last_visited_state": last_visited_state, "closest_index": closest_index,
               "cur_pos": cur_pos, "params_this_prop": gaussian_params, **{f"dham_{k}": v for k, v in diag.items()}}
        if reach is None:
            print("closest_index (local target) updated. converted to x space that's: ", current_qspace[closest_index])
            print("last visited state (local start) updated. converted to x space that's: ", current_qspace[min(last_visited_state, len(current_qspace)-1)])
            next_params, info = try_and_optim_M(working_MM, working_indices = working_indices, num_gaussian = config.num_gaussian,
                                                start_index = last_visited_state, end_index = closest_index, qspace = current_qspace,
                                                plot = False, rng = np.random.default_rng([seed, i_prop + 1]))
            save_params(outdir, i_prop + 1, next_params)
            dec.update({"next_params": next_params, **{k: v for k, v in info.items()}})
        np.savez(os.path.join(outdir, "decisions", f"prop_{i_prop:04d}.npz"), **dec)
        t_dec = time.time() - t0
        prop_wall.append(t_md + t_dec)
        log(f"propagation {i_prop} done: MD {t_md:.0f} s, DHAM+optimiser {t_dec:.0f} s, s range {s.min():.2f}-{s.max():.2f} A, "
            f"max applied bias {v_omm.max():.2f} kcal/mol, non-adjacent transitions {diag['frac_offdiag_nonadjacent']:.1e}"
            + (f", next bias max {bias_kcal(config.qspace, next_params).max():.2f}, model MFPT {info['optimised_mfpt_ns']:.3g} ns" if reach is None else ""))

        i_prop += 1
        #checkpoint named by the propagation it precedes, recorded in progress.json only after it is
        #complete: a kill at any point leaves a consistent (state, next_prop) pair
        state_name = f"state_{i_prop:04d}.xml"
        simulation.saveState(os.path.join(outdir, state_name + ".tmp"))
        os.replace(os.path.join(outdir, state_name + ".tmp"), os.path.join(outdir, state_name))
        post = reach is not None and args.post_ns > 0
        done = (reach is not None and not post) or (reach is None and i_prop > config.num_propagation)
        write_json_atomic(progress_path, {"next_prop": i_prop, "state": state_name, "done": done, "n_resume": n_resume,
                                          "reached": reach is not None, "reach": reach, "phase": "post" if post else "kombi",
                                          "post_next": 0, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        old_state = os.path.join(outdir, f"state_{i_prop - 2:04d}.xml")
        if os.path.exists(old_state):
            os.remove(old_state)
        if not done and time.time() - t_start + 1.3 * max(prop_wall) > args.wall_h * 3600:
            log(f"wall-time budget: checkpoint before propagation {i_prop}, exit {EXIT_WALL}")
            return EXIT_WALL

    ######### post processing #########
    n_props = i_prop
    if reach is None:   #censored: no first passage within the cap
        write_first_passage(outdir, n_props, None, L)
        return EXIT_OK
    if args.post_ns <= 0:
        return EXIT_OK

    #unbiased continuation after the first passage (bias amplitudes 0, same propagation length);
    #files cv/post_NNNN.npz, shell/post_NNNN.csv, traj/*_post_NNNN.dcd; KOMBI files untouched
    dt_ps = config.stepsize.value_in_unit(unit.picosecond)
    n_post = int(np.ceil(args.post_ns * 1000.0 / (L * dt_ps)))
    zero = zero_amplitude(load_params(outdir, n_props - 1))
    set_bias(simulation.context, zero)
    k0 = progress["post_next"] if in_post else 0
    for k in range(k0, n_post):
        t0 = time.time()
        s, v_omm, v_py, n_bad = propagate(simulation, k, zero, outdir, L, io, tag="post", first_step=(n_props + k) * L)
        if n_bad:
            log(f"bias not zero at {n_bad} steps of post segment {k}. Stopping."); return EXIT_BIAS_MISMATCH
        prop_wall.append(time.time() - t0)
        log(f"post-passage segment {k + 1}/{n_post} done: {prop_wall[-1]:.0f} s, s range {s.min():.2f}-{s.max():.2f} A")
        state_name = f"state_post_{k + 1:04d}.xml"
        simulation.saveState(os.path.join(outdir, state_name + ".tmp"))
        os.replace(os.path.join(outdir, state_name + ".tmp"), os.path.join(outdir, state_name))
        done = k + 1 >= n_post
        write_json_atomic(progress_path, {"next_prop": n_props, "state": state_name, "done": done, "n_resume": n_resume,
                                          "reached": True, "reach": reach, "phase": "post", "post_next": k + 1,
                                          "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        old_state = os.path.join(outdir, f"state_post_{k - 1:04d}.xml")
        if os.path.exists(old_state):
            os.remove(old_state)
        if not done and time.time() - t_start + 1.3 * max(prop_wall) > args.wall_h * 3600:
            log(f"wall-time budget: checkpoint before post segment {k + 1}, exit {EXIT_WALL}")
            return EXIT_WALL
    return EXIT_OK

if __name__ == "__main__":
    sys.exit(main())
