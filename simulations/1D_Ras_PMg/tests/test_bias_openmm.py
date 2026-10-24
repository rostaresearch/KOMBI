"""The bias OpenMM applies equals the Python bias the optimiser and DHAM use, and the start-up guard
detects a bias applied as ten identical bonds, a sqrt(2)-wide Gaussian, mis-ordered (a, b, c)
parameters or kcal/kJ confusion.

Runs on the Reference platform with a small stand-in system (the Mg and P atoms at the driver's
indices, every other particle massless and force-free), so it needs no GPU and no CHARMM files.
The same guard is run on the full Ras system by `explore_sim_PMg.py --check-only`.
"""
import numpy as np
import openmm as omm
import openmm.app as omm_app
import openmm.unit as unit
import pytest

import config
import explore_sim_PMg as drv
from util import get_total_bias, get_total_bias_derivative

MG, P = config.mg_index, config.p_index
N_ATOMS = P + 1
R_GRID = np.linspace(2.5, 9.5, 57)   # A


def make_simulation(params, legacy_bonds=False, sqrt2_openmm=False, with_cv=True, seed=7, tether_group=3, integ_groups=None):
    """stand-in system: Mg and P with their masses and a weak harmonic tether so that MD moves them.
    The tether sits in group 3, where CharmmPsfFile.createSystem puts the CHARMM bonds."""
    system = omm.System()
    for i in range(N_ATOMS):
        system.addParticle(24.305 if i == MG else 30.974 if i == P else 0.0)
    tether = omm.HarmonicBondForce(); tether.addBond(MG, P, 0.40, 200.0); tether.setForceGroup(tether_group); system.addForce(tether)
    if legacy_bonds or sqrt2_openmm:
        # the bias as ten identical bonds, or the (2c)^2 width on the OpenMM side
        a, b, c = drv.to_openmm_units(params)
        n = config.num_gaussian
        width = "(2*c{i})^2" if sqrt2_openmm else "(2*c{i}^2)"
        f = omm.CustomBondForce(' + '.join(f'a{i}*exp(-(r-b{i})^2/' + width.format(i=i) + ')' for i in range(n)))
        for i in range(n):
            f.addGlobalParameter(f'a{i}', a[i]); f.addGlobalParameter(f'b{i}', b[i]); f.addGlobalParameter(f'c{i}', c[i])
            if legacy_bonds:
                f.addBond(MG, P)
        if not legacy_bonds:
            f.addBond(MG, P)
        f.setForceGroup(drv.BIAS_GROUP); f.setName('KOMBI_bias'); system.addForce(f)
    else:
        drv.add_bias(system, params)
    if with_cv:
        drv.add_cv_force(system)
    integ = omm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    integ.setRandomNumberSeed(seed)
    if integ_groups is not None:
        integ.setIntegrationForceGroups(integ_groups)
    elif with_cv:
        integ.setIntegrationForceGroups(drv.INTEGRATION_GROUPS)
    top = omm_app.Topology(); ch = top.addChain(); res = top.addResidue("X", ch)
    for i in range(N_ATOMS):
        top.addAtom("X", None, res)
    sim = omm_app.Simulation(top, system, integ, omm.Platform.getPlatformByName("Reference"))
    pos = np.zeros((N_ATOMS, 3)); pos[:, 0] = np.arange(N_ATOMS) * 1e-3
    pos[MG] = [1.0, 1.0, 1.0]; pos[P] = [1.0 + 0.33 / np.sqrt(3) * 1.0, 1.0 + 0.33 / np.sqrt(3), 1.0 + 0.33 / np.sqrt(3)]
    sim.context.setPositions(pos)
    return sim


def param_sets():
    rng = np.random.default_rng(11)
    sets = drv.guard_param_sets(rng)
    sets.append(("prop0", drv.random_initial_bias(rng)))
    for k in range(5):
        n = config.num_gaussian
        sets.append((f"r{k}", np.concatenate((rng.uniform(0.1, 1.0, n), rng.uniform(1.0, 8.0, n), rng.uniform(0.7, 2.0, n)))))
    return sets


def energy_force_at(sim, r_A):
    ctx = sim.context
    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer).copy()
    u = (pos[P] - pos[MG]) / np.linalg.norm(pos[P] - pos[MG])
    pos[P] = pos[MG] + u * r_A / 10
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True, groups=1 << drv.BIAS_GROUP)
    e = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    return e, f[P] @ u, f[MG] @ u


@pytest.mark.parametrize("name,params", param_sets())
def test_openmm_bias_energy_and_force_equal_python(name, params):
    sim = make_simulation(params)
    for r in R_GRID:
        e, fp, fmg = energy_force_at(sim, r)
        v = get_total_bias(np.array([r]), params, num_gaussians=config.num_gaussian)[0]
        dv = get_total_bias_derivative(np.array([r]), params, num_gaussians=config.num_gaussian)[0]
        assert e == pytest.approx(v, rel=1e-9, abs=1e-12)
        assert fp == pytest.approx(-dv, rel=1e-7, abs=1e-9)     # force on P along Mg->P is -dV/dr
        assert fmg == pytest.approx(dv, rel=1e-7, abs=1e-9)
    # derivative helper itself, against a central difference
    h = 1e-5
    num = (get_total_bias(R_GRID + h, params, config.num_gaussian) - get_total_bias(R_GRID - h, params, config.num_gaussian)) / (2 * h)
    np.testing.assert_allclose(get_total_bias_derivative(R_GRID, params, config.num_gaussian), num, rtol=1e-6, atol=1e-8)


def test_set_bias_updates_match_python():
    """propagations >= 1 change the bias through setParameter; check that path too"""
    sets = param_sets()
    sim = make_simulation(sets[0][1])
    for name, params in sets[1:]:
        drv.set_bias(sim.context, params)
        for r in (3.0, 3.7, 4.4, 6.1, 7.9):
            e, fp, _ = energy_force_at(sim, r)
            assert e == pytest.approx(get_total_bias(np.array([r]), params, config.num_gaussian)[0], rel=1e-9, abs=1e-12)


def test_guard_passes_on_correct_bias(tmp_path):
    params = param_sets()[0][1]
    sim = make_simulation(params)
    assert drv.check_force_layout(sim) == []
    assert drv.run_guard(sim, np.random.default_rng(1), params, str(tmp_path), "test")


def test_layout_catches_group_collisions(tmp_path):
    """regression: CharmmPsfFile puts CHARMM terms in groups 1-6; a KOMBI group that collides with
    them, or an integrator that skips them, must be caught"""
    params = param_sets()[0][1]
    sim = make_simulation(params, tether_group=drv.BIAS_GROUP)          # physical force in the bias group
    assert any("shares group" in p for p in drv.check_force_layout(sim))
    assert not drv.run_guard(sim, np.random.default_rng(1), params, str(tmp_path), "collide")
    sim = make_simulation(params, integ_groups={0, drv.BIAS_GROUP})     # CHARMM bonds (group 3) not integrated
    assert any("not integrated" in p for p in drv.check_force_layout(sim))


@pytest.mark.parametrize("fault", ["ten_bonds", "sqrt2_width", "scrambled_params", "kcal_not_kJ"])
def test_guard_catches_each_legacy_fault(fault, tmp_path):
    params = param_sets()[0][1]
    if fault == "ten_bonds":
        sim = make_simulation(params, legacy_bonds=True)
    elif fault == "sqrt2_width":
        sim = make_simulation(params, sqrt2_openmm=True)
    else:
        sim = make_simulation(params)
    orig_set_bias = drv.set_bias
    try:
        if fault == "scrambled_params":
            # reshape(10, 3) of [a, b, c] taken as (a, b, c) triples
            drv.set_bias = lambda ctx, p: orig_set_bias(ctx, np.asarray(p).reshape(config.num_gaussian, 3).T.reshape(-1))
        elif fault == "kcal_not_kJ":
            def no_conv(ctx, p):
                n = config.num_gaussian
                for j in range(n):
                    ctx.setParameter(f'a{j}', p[j]); ctx.setParameter(f'b{j}', p[n + j] / 10); ctx.setParameter(f'c{j}', p[2*n + j] / 10)
            drv.set_bias = no_conv
        ok = drv.run_guard(sim, np.random.default_rng(1), params, str(tmp_path), fault)
    finally:
        drv.set_bias = orig_set_bias
    assert not ok
    if fault == "ten_bonds":
        assert any("10 bonds" in p for p in drv.check_force_layout(sim))


@pytest.mark.parametrize("fault", ["util_gaussian_sqrt2", "params_to_triples_scrambled"])
def test_guard_catches_faults_in_dham_and_optimiser_code(fault, tmp_path, monkeypatch):
    """a sqrt(2)-wide util.gaussian or a (10, 3) reshape for DHAM on the Python side, with OpenMM
    correct, must be caught: the guard compares OpenMM with the code DHAM and the optimiser call"""
    import dham
    import util
    params = param_sets()[0][1]
    sim = make_simulation(params)
    assert drv.run_guard(sim, np.random.default_rng(1), params, str(tmp_path), "before")
    if fault == "util_gaussian_sqrt2":
        legacy = lambda x, a, b, c: a * np.exp(-(x - b)**2 / ((2*c)**2))
        monkeypatch.setattr(util, "gaussian", legacy)     # used by util.optimiser_total_bias
        monkeypatch.setattr(dham, "gaussian", legacy)     # DHAM imports it by name
    else:
        monkeypatch.setattr(drv, "params_to_triples", lambda p: np.asarray(p, dtype=float).reshape(config.num_gaussian, 3))
    assert not drv.run_guard(sim, np.random.default_rng(1), params, str(tmp_path), fault)


def test_optimiser_bias_helper_matches_loops():
    """util.optimiser_total_bias equals an explicit loop over the Gaussians"""
    import util
    rng = np.random.default_rng(3)
    for name, p in param_sets():
        n = config.num_gaussian
        a, b, c = p[:n], p[n:2*n], p[2*n:]
        ref = np.zeros_like(config.qspace)
        for j in range(n):
            ref += util.gaussian(config.qspace, a[j], b[j], c[j])
        assert np.array_equal(util.optimiser_total_bias(config.qspace, p, n), ref)


def test_cv_readback_and_no_effect_on_dynamics():
    """the CV group returns the Mg-P distance and does not change the trajectory, including the
    velocity draw (OpenMM's setVelocitiesToTemperature evaluates every group, hence cv_on = 0)"""
    params = param_sets()[0][1]
    traj = []
    for with_cv in (False, True):
        sim = make_simulation(params, with_cv=with_cv, seed=5)
        ctx = sim.context
        if with_cv:
            ctx.setParameter('cv_on', 0.0)
        ctx.setVelocitiesToTemperature(300 * unit.kelvin, 3)
        if with_cv:
            ctx.setParameter('cv_on', 1.0)
        xs = []
        for k in range(300):
            sim.integrator.step(1)
            pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            r = np.linalg.norm(pos[P] - pos[MG])
            if with_cv:
                s, v = drv.read_cv_and_bias(ctx)
                assert s == pytest.approx(r, abs=1e-9)
                assert v == pytest.approx(drv.bias_kcal(np.array([r]), params)[0], rel=1e-9, abs=1e-12)
            xs.append(pos[[MG, P]].copy())
        traj.append(np.array(xs))
    assert np.array_equal(traj[0], traj[1])


def test_parameter_layout_and_units(tmp_path):
    n = config.num_gaussian
    a, b, c = np.arange(n) + 0.5, np.arange(n) + 3.0, np.arange(n) * 0.1 + 0.7
    p = np.concatenate((a, b, c))
    t = drv.params_to_triples(p)
    assert t.shape == (n, 3)
    assert np.array_equal(t[:, 0], a) and np.array_equal(t[:, 1], b) and np.array_equal(t[:, 2], c)
    ak, bn, cn = drv.to_openmm_units(p)
    assert np.allclose(ak, a * 4.184) and np.allclose(bn, b / 10) and np.allclose(cn, c / 10)
    (tmp_path / "params").mkdir()
    drv.save_params(str(tmp_path), 3, p)
    assert np.array_equal(drv.load_params(str(tmp_path), 3), p)   # exact round trip: the applied bias is the saved one
    p0 = drv.random_initial_bias(np.random.default_rng(0))
    assert np.allclose(p0[:n], 0.01) and np.all(np.abs(p0[n:2*n] - 3.7) <= 0.2) and np.all((p0[2*n:] >= 1) & (p0[2*n:] <= 5))


def test_first_passage_accounting():
    """first visit, counted over all propagations: equals the index in the concatenated record + 1"""
    rng = np.random.default_rng(2)
    L = 500
    props = [3.3 + 0.3 * rng.standard_normal(L) for _ in range(4)]
    props[3][123] = 8.2
    props[3][300] = 8.5       # a later visit must not count
    s_all = np.concatenate(props)
    reach = int(np.nonzero(props[-1] >= 8.0)[0][0])
    assert reach == 123
    total = drv.first_passage_steps(len(props), reach, L)
    assert total == int(np.nonzero(s_all >= 8.0)[0][0]) + 1 == 3 * L + 124
