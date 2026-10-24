"""
The Ras KOMBI system with periodic boundaries (PME): equilibration and unbiased production, for the
check that the structure is stable (RMSD) and as the start of the PBC KOMBI runs. Also the matching
non-periodic unbiased control (--nonpbc), with the KOMBI runs' exact nonbonded settings.

PME, HBonds, checkpoint/resume with exit 42 at the wall-time budget; no positional restraints in
production, no HMR, the integrator of the KOMBI runs (LangevinIntegrator, 1/ps, 2 fs, 300 K), and
by default the nonbonded settings of this system's own preparation (KOMBI SI.5: NAMD 3.0, PME,
vdW switch 10-12 A; --cutoff-nm 1.2 --switch-nm 1.0). NPT: MonteCarloBarostat, 1 bar, every 25 steps.

Start: traj_0.restart.pdb, the common start of every KOMBI run. It is the last frame of the NAMD
preparation (10 ns NVT + 10 ns NPT, PBC/PME, 303.15 K) but has no box (CRYST1 zeros)
and no NAMD .xsc is available. Its water oxygens are wrapped into a cube spanning 100.2 A, and the
first-shell water count at the faces matches the interior for an edge of ~99.7 A: the NPT box was
~100 A. The initial box is therefore 10.0 nm; the NPT phases settle it.

Stages
  equil          minimise (heavy solute atoms restrained) -> NVT 100 ps restrained -> NPT 500 ps with
                 backbone restraints -> NPT 1 ns free; writes equil/equil_state.xml and equil.pdb
  prod --rep k   NPT production from equil_state.xml, new velocities (seed 1000 + k), --ns long
  nonpbc --rep k the non-periodic control: CutoffNonPeriodic 1.0 nm (reaction field) as the KOMBI
                 runs (whatever --cutoff-nm says), from traj_0.restart.pdb with new velocities, NVT

Outputs per stage/replicate, one segment per start/resume (segments.json gives each segment's first
step): solute_segNN.dcd (no bulk water, every --solute-ps), full_segNN.dcd (every --full-ps),
thermo_segNN.csv, shell_segNN.csv (Mg2+ first shell, see mg_shell.py), solute.pdb, run_config.json,
state.xml checkpoints, progress.json. Coordinates are written unwrapped (continuous).

usage: python pbc_equil_prod.py equil --system-dir ../system --outdir pbc
       python pbc_equil_prod.py prod --rep 0 --ns 10 --system-dir ../system --outdir pbc
       python pbc_equil_prod.py nonpbc --rep 0 --ns 10 --system-dir ../system --outdir nonpbc
"""
import argparse, json, os, sys, time
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

import mg_shell

BOX_EDGE_NM = 10.0                       # initial cubic box (see above); NPT settles it
KOMBI_CUTOFF = 1.0 * u.nanometer         # non-periodic control: the KOMBI runs' CutoffNonPeriodic 1.0 nm
T0 = 300.0 * u.kelvin
FRIC = 1.0 / u.picosecond
DT = 0.002 * u.picoseconds
PRESSURE = 1.0 * u.bar
MG, P = 7799, 7840


def log(*a):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *a, flush=True)


def read_params(toppar_str):
    here = os.path.dirname(os.path.abspath(toppar_str))
    files = []
    for line in open(toppar_str):
        s = line.split('!')[0].strip()
        if s:
            files.append(s if os.path.isabs(s) else os.path.join(here, s))
    return app.CharmmParameterSet(*files)


def atom_sets(psf):
    """index sets: solute (all but segment SOLV), protein backbone N/CA/C, heavy solute atoms
    (protein, GDP, Mg, Pi), and the Mg-shell sets of mg_shell.py"""
    sol, bb, heavy = [], [], []
    for i, a in enumerate(psf.atom_list):
        seg = a.system
        if seg != "SOLV":
            sol.append(i)
        if seg in ("PROA", "PROB") and a.name in ("N", "CA", "C"):
            bb.append(i)
        if seg in ("PROA", "PROB", "HETA", "HETB", "HETC") and not a.name.startswith("H"):
            heavy.append(i)
    return dict(sol=np.array(sol), bb=np.array(bb), heavy=np.array(heavy), shell=mg_shell.shell_sets(psf))


def add_posres(system, idx, ref_nm, k_kj, pbc):
    expr = "k_posres*periodicdistance(x,y,z,x0,y0,z0)^2" if pbc else "k_posres*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    f = mm.CustomExternalForce(expr)
    f.addGlobalParameter("k_posres", k_kj)
    for n in ("x0", "y0", "z0"):
        f.addPerParticleParameter(n)
    for i in idx:
        f.addParticle(int(i), [float(v) for v in ref_nm[i]])
    system.addForce(f)
    return f


def subset_topology(topology, keep):
    keep = set(int(i) for i in keep)
    mod = app.Modeller(topology, [mm.Vec3(0, 0, 0)] * topology.getNumAtoms())
    mod.delete([a for a in mod.topology.atoms() if a.index not in keep])
    return mod.topology


class Recorder:
    """solute/full DCD, thermo and Mg-shell logs at fixed strides. Every start or resume opens a new
    segment (files *_segNN.*, first step recorded in segments.json): after a hard kill, frames written
    past the last checkpoint stay in the old segment and analysis_rmsd.py drops them, instead of
    being appended twice (or after a truncated frame)."""

    def __init__(self, sim, psf, sets, outdir, solute_ps, full_ps, log_ps, pbc, step0):
        dt_ps = DT.value_in_unit(u.picosecond)
        self.sim, self.sets, self.pbc = sim, sets, pbc
        self.s_sol = int(round(solute_ps / dt_ps)); self.s_full = int(round(full_ps / dt_ps)); self.s_log = int(round(log_ps / dt_ps))
        self.chunk = int(np.gcd.reduce([self.s_sol, self.s_full, self.s_log]))
        seg_path = os.path.join(outdir, "segments.json")
        segs = json.load(open(seg_path)) if os.path.exists(seg_path) else []
        seg = len(segs)
        segs.append({"seg": seg, "step0": int(step0), "solute_stride": self.s_sol, "full_stride": self.s_full, "log_stride": self.s_log})
        json.dump(segs, open(seg_path, "w"), indent=1)
        self.f_sol = open(os.path.join(outdir, f"solute_seg{seg:02d}.dcd"), "wb")
        self.f_full = open(os.path.join(outdir, f"full_seg{seg:02d}.dcd"), "wb")
        top_sol = subset_topology(psf.topology, sets["sol"])
        if pbc:
            top_sol.setPeriodicBoxVectors(sim.context.getState().getPeriodicBoxVectors())
        self.d_sol = app.DCDFile(self.f_sol, top_sol, DT, step0, self.s_sol)
        self.d_full = app.DCDFile(self.f_full, sim.topology, DT, step0, self.s_full)
        self.thermo = open(os.path.join(outdir, f"thermo_seg{seg:02d}.csv"), "w")
        self.shell = open(os.path.join(outdir, f"shell_seg{seg:02d}.csv"), "w")
        self.thermo.write("step,time_ps,potential_kJmol,temperature_K,box_nm,volume_nm3,density_gml\n")
        self.shell.write(mg_shell.HEADER + "\n")
        if seg == 0:
            with open(os.path.join(outdir, "solute.pdb"), "w") as fh:   # topology for the solute DCDs (start frame)
                st = sim.context.getState(getPositions=True)
                app.PDBFile.writeFile(top_sol, st.getPositions(asNumpy=True)[sets["sol"]], fh)
        self.mass = sum(sim.system.getParticleMass(i).value_in_unit(u.dalton) for i in range(sim.system.getNumParticles()))
        self.ndf = 3 * sim.system.getNumParticles() - sim.system.getNumConstraints() - 3

    def record(self, step):
        need_pos = step % self.s_sol == 0 or step % self.s_full == 0 or step % self.s_log == 0
        # continuous (unwrapped) coordinates: wrapping molecule by molecule splits Ras from GAP;
        # the shell distances use the minimum image
        st = self.sim.context.getState(getPositions=need_pos, getEnergy=True)
        box = st.getPeriodicBoxVectors() if self.pbc else None
        if need_pos:
            pos = st.getPositions(asNumpy=True)
            if step % self.s_full == 0:
                self.d_full.writeModel(pos, periodicBoxVectors=box)
            if step % self.s_sol == 0:
                self.d_sol.writeModel(pos[self.sets["sol"]], periodicBoxVectors=box)
        if step % self.s_log == 0:
            T = (2 * st.getKineticEnergy() / (self.ndf * u.MOLAR_GAS_CONSTANT_R)).value_in_unit(u.kelvin)
            if self.pbc:
                Lb = box[0][0].value_in_unit(u.nanometer); vol = Lb * box[1][1].value_in_unit(u.nanometer) * box[2][2].value_in_unit(u.nanometer)
                dens = self.mass / 6.02214076e23 / (vol * 1e-21)
            else:
                Lb = vol = dens = float("nan")
            self.thermo.write(f"{step},{step * DT.value_in_unit(u.picosecond):.1f},{st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole):.2f},{T:.2f},{Lb:.5f},{vol:.3f},{dens:.5f}\n")
            Lv = np.array([box[k][k].value_in_unit(u.angstrom) for k in range(3)]) if self.pbc else None
            self.shell.write(mg_shell.format_row(step, mg_shell.shell_row(pos.value_in_unit(u.angstrom), self.sets["shell"], Lv)))

    def flush(self):
        for f in (self.f_sol, self.f_full, self.thermo, self.shell):
            f.flush()

    def close(self):
        for f in (self.f_sol, self.f_full, self.thermo, self.shell):
            f.close()


def make_system(psf, params, pbc, barostat, cutoff_nm, switch_nm):
    if pbc:
        system = psf.createSystem(params, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_nm * u.nanometer,
                                  switchDistance=switch_nm * u.nanometer, constraints=app.HBonds)
        if barostat:
            system.addForce(mm.MonteCarloBarostat(PRESSURE, T0, 25))
    else:   # the KOMBI runs' settings, as run
        system = psf.createSystem(params, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=KOMBI_CUTOFF,
                                  constraints=app.HBonds)
    return system


def run_md(sim, rec, step0, nsteps, prog_path, state_path, ckpt_steps, t_start, wall_h, extra):
    """advance in chunks from step0 to nsteps, recording; checkpoint; exit 42 at the wall budget"""
    step = step0
    while step < nsteps:
        n = min(rec.chunk, nsteps - step)
        sim.step(n)
        step += n
        rec.record(step)
        if step % ckpt_steps == 0 or step == nsteps:
            rec.flush()
            sim.saveState(state_path + ".tmp"); os.replace(state_path + ".tmp", state_path)
            json.dump({"step": step, "done": step >= nsteps, **extra}, open(prog_path + ".tmp", "w")); os.replace(prog_path + ".tmp", prog_path)
            if step < nsteps and time.time() - t_start > wall_h * 3600:
                log(f"wall budget hit at step {step}: checkpoint, exit 42")
                rec.close(); sys.exit(42)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["equil", "prod", "nonpbc"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--ns", type=float, default=10.0)
    ap.add_argument("--system-dir", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--wall-h", type=float, default=1e9)
    ap.add_argument("--solute-ps", type=float, default=10.0)
    ap.add_argument("--full-ps", type=float, default=100.0)
    ap.add_argument("--log-ps", type=float, default=10.0)
    ap.add_argument("--cutoff-nm", type=float, default=1.2, help="PBC stages: PME real-space cutoff (as the NAMD preparation)")
    ap.add_argument("--switch-nm", type=float, default=1.0, help="PBC stages: LJ switch distance (as the NAMD preparation)")
    ap.add_argument("--smoke", action="store_true", help="code-path test: 40-step phases, 0.02 ps strides, 20 minimiser iterations")
    args = ap.parse_args()
    if args.smoke:
        args.solute_ps = args.full_ps = args.log_ps = 0.02
        args.ns = min(args.ns, 0.0002)
    t_start = time.time()
    sysdir = os.path.abspath(args.system_dir)
    pbc = args.stage != "nonpbc"
    sub = "equil" if args.stage == "equil" else f"{args.stage}_rep{args.rep:02d}"
    out = os.path.join(os.path.abspath(args.outdir), sub)
    os.makedirs(out, exist_ok=True)
    psf = app.CharmmPsfFile(os.path.join(sysdir, "step3_input.psf"))
    if pbc:
        psf.setBox(BOX_EDGE_NM * u.nanometer, BOX_EDGE_NM * u.nanometer, BOX_EDGE_NM * u.nanometer)
    params = read_params(os.path.join(sysdir, "toppar.str"))
    sets = atom_sets(psf)
    start = app.PDBFile(os.path.join(sysdir, "traj_0.restart.pdb"))
    assert start.topology.getNumAtoms() == psf.topology.getNumAtoms()
    ref_nm = np.asarray(start.getPositions().value_in_unit(u.nanometer))
    plat = mm.Platform.getPlatformByName(args.platform)
    props = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL") else {}
    seed = 1000 + args.rep
    dt_ps = DT.value_in_unit(u.picosecond)
    prog_path, state_path = os.path.join(out, "progress.json"), os.path.join(out, "state.xml")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else None
    if prog and prog.get("done"):
        log(f"{sub} already done"); return
    run_cfg = {"stage": args.stage, "rep": args.rep, "pbc": pbc, "openmm": mm.__version__,
               "nonbonded": (f"PME, cutoff {args.cutoff_nm} nm, switch {args.switch_nm} nm" if pbc
                             else "CutoffNonPeriodic 1.0 nm (reaction field), as the KOMBI runs"),
               "cutoff_nm": args.cutoff_nm if pbc else 1.0, "switch_nm": args.switch_nm if pbc else None,
               "initial_box_nm": BOX_EDGE_NM if args.stage == "equil" else None,
               "integrator": "LangevinIntegrator 300 K, 1/ps, 2 fs, HBonds", "barostat": "MC 1 bar /25 steps" if pbc else None,
               "seed": seed, "smoke": args.smoke}
    if args.stage == "prod":   # production must use the equilibration's nonbonded settings
        eq = json.load(open(os.path.join(os.path.abspath(args.outdir), "equil", "run_config.json")))
        assert (eq["cutoff_nm"], eq["switch_nm"]) == (args.cutoff_nm, args.switch_nm), \
            f"production cutoff/switch {args.cutoff_nm}/{args.switch_nm} differ from the equilibration's {eq['cutoff_nm']}/{eq['switch_nm']}"
    json.dump(run_cfg, open(os.path.join(out, "run_config.json"), "w"), indent=1)
    log(f"{sub}: {run_cfg['nonbonded']}")

    if args.stage == "equil":
        # phases: (name, steps, barostat, restraint set, k kJ/mol/nm^2)
        phases = [("nvt_restr", int(100 / dt_ps), False, "heavy", 400.0),
                  ("npt_bb", int(500 / dt_ps), True, "bb", 100.0),
                  ("npt_free", int(1000 / dt_ps), True, None, 0.0)]
        if args.smoke:
            phases = [(n, 40, b, r, k) for n, _, b, r, k in phases]
        done_phase = prog["phase"] if prog else -1
        for ip, (name, nsteps, baro, rset, k) in enumerate(phases):
            if ip < done_phase:
                continue
            system = make_system(psf, params, True, baro, args.cutoff_nm, args.switch_nm)
            if rset:
                add_posres(system, sets[rset], ref_nm, k, True)
                log(f"restraints '{rset}': {len(sets[rset])} atoms (heavy solute = protein/GDP/Mg/Pi heavy atoms, bb = N/CA/C)")
            integ = mm.LangevinIntegrator(T0, FRIC, DT); integ.setRandomNumberSeed(seed + 17 * ip + 1)
            sim = app.Simulation(psf.topology, system, integ, plat, props)
            pdir = os.path.join(out, name); os.makedirs(pdir, exist_ok=True)
            pstate, pprog = os.path.join(pdir, "state.xml"), os.path.join(pdir, "progress.json")
            pp = json.load(open(pprog)) if os.path.exists(pprog) else None
            if pp and pp.get("done"):
                continue
            if pp:     # resume inside this phase
                sim.loadState(pstate); step0 = sim.context.getStepCount()   # the checkpoint itself is authoritative
            elif ip == 0:
                sim.context.setPositions(start.getPositions())
                e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
                # minimise with the phase-0 (heavy solute atom) restraints in place
                sim.minimizeEnergy(tolerance=10 * u.kilojoule_per_mole / u.nanometer, maxIterations=20 if args.smoke else 5000)
                e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
                log(f"minimised: {e0} -> {e1}")
                sim.context.setVelocitiesToTemperature(T0, seed)
                step0 = 0
            else:      # continue from the previous phase (positions, velocities, box)
                prev = os.path.join(out, phases[ip - 1][0], "state.xml")
                st = mm.XmlSerializer.deserialize(open(prev).read())
                sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
                sim.context.setPositions(st.getPositions()); sim.context.setVelocities(st.getVelocities())
                step0 = 0
            log(f"equil phase {name}: {nsteps} steps, barostat={baro}, restraints={rset} k={k}")
            rec = Recorder(sim, psf, sets, pdir, args.solute_ps, args.full_ps, args.log_ps, True, step0)
            run_md(sim, rec, step0, nsteps, pprog, pstate, int(100 / dt_ps), t_start, args.wall_h, {"phase": ip})
            rec.close()
            json.dump({"phase": ip + 1, "done": False}, open(prog_path, "w"))
        final = os.path.join(out, phases[-1][0], "state.xml")
        st = mm.XmlSerializer.deserialize(open(final).read())
        with open(os.path.join(out, "equil_state.xml"), "w") as fh:
            fh.write(mm.XmlSerializer.serialize(st))
        top = psf.topology; top.setPeriodicBoxVectors(st.getPeriodicBoxVectors())
        with open(os.path.join(out, "equil.pdb"), "w") as fh:
            app.PDBFile.writeFile(top, st.getPositions(), fh)
        json.dump({"phase": len(phases), "done": True, "box_nm": st.getPeriodicBoxVectors()[0][0].value_in_unit(u.nanometer)}, open(prog_path, "w"))
        log("equilibration done; box", st.getPeriodicBoxVectors()[0][0])
        return

    nsteps = int(round(args.ns * 1000 / dt_ps))
    system = make_system(psf, params, pbc, pbc, args.cutoff_nm, args.switch_nm)
    integ = mm.LangevinIntegrator(T0, FRIC, DT); integ.setRandomNumberSeed(seed + (prog["n_resume"] + 1 if prog else 0) * 7919)
    sim = app.Simulation(psf.topology, system, integ, plat, props)
    log(f"{sub}: OpenMM {mm.__version__} on {sim.context.getPlatform().getName()}, {nsteps} steps, pbc={pbc}")
    if prog:
        sim.loadState(state_path); step0 = sim.context.getStepCount(); n_resume = prog.get("n_resume", 0) + 1
        log(f"resumed at step {step0}")
    else:
        if pbc:
            st = mm.XmlSerializer.deserialize(open(os.path.join(os.path.abspath(args.outdir), "equil", "equil_state.xml")).read())
            sim.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
            sim.context.setPositions(st.getPositions())
        else:
            sim.context.setPositions(start.getPositions())
        sim.context.setVelocitiesToTemperature(T0, seed)
        step0, n_resume = 0, 0
    rec = Recorder(sim, psf, sets, out, args.solute_ps, args.full_ps, args.log_ps, pbc, step0)
    t0 = time.time()
    run_md(sim, rec, step0, nsteps, prog_path, state_path, int(1000 / dt_ps), t_start, args.wall_h, {"n_resume": n_resume})
    rec.close()
    el = time.time() - t0
    log(f"{sub} COMPLETE: {(nsteps - step0) * dt_ps / 1000:.2f} ns in {el / 3600:.2f} h ({(nsteps - step0) * dt_ps / 1000 / el * 86400:.1f} ns/day)")


if __name__ == "__main__":
    main()
