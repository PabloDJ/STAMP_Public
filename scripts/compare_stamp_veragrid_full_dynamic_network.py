#!/usr/bin/env python3
"""Compare the complete 88-state STAMP and VeraGrid dynamic networks."""
from pathlib import Path
import sys
import numpy as np
import scipy.linalg as la
from scipy.optimize import root

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid, STAMP_LOADS


def main() -> None:
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    grid=build_stamp_wscc_grid(dynamic_lines=True,full_dynamic_network=True)
    pf=PowerFlowDriver(grid,power_flow_options()); pf.run()
    problem=RmsProblemDae(grid,RmsOptions(time_step=.001),pf.results)
    problem.set_events_group(RmsEventsGroup("full_dynamic_network"))
    vector=problem.get_x0(); nx=len(problem.state_vars)
    index={str(var):i for i,var in enumerate(problem.state_vars)}
    buses={int(bus.name.removeprefix("Bus")):i for i,bus in enumerate(grid.buses)}

    for branch,line in enumerate(grid.lines):
        f=grid.buses.index(line.bus_from); voltage=pf.results.voltage[f]
        vq,vd=voltage.real,-voltage.imag; power=pf.results.Sf[branch]/grid.Sbase
        total_iq=(power.real*vq-power.imag*vd)/(vq*vq+vd*vd)
        total_id=(power.real*vd+power.imag*vq)/(vq*vq+vd*vd)
        bf=int(line.bus_from.name.removeprefix("Bus")); bt=int(line.bus_to.name.removeprefix("Bus"))
        name=f"NET.{min(bf,bt)}{max(bf,bt)}"
        vector[index[f"{name}.iq"]]=total_iq-line.B*vd/2
        vector[index[f"{name}.id"]]=total_id+line.B*vq/2
    for load_number,(bus_number,p_mw,q_mvar) in enumerate(STAMP_LOADS,1):
        voltage=pf.results.voltage[buses[bus_number]]; vq,vd=voltage.real,-voltage.imag
        p,q=p_mw/grid.Sbase,q_mvar/grid.Sbase; den=vq*vq+vd*vd
        conductance=p/den
        vector[index[f"Load{load_number}.ilq"]]=(p*vq-q*vd)/den-conductance*vq
        vector[index[f"Load{load_number}.ild"]]=(p*vd+q*vq)/den-conductance*vd
    for bus_number,bus_index in buses.items():
        voltage=pf.results.voltage[bus_index]
        vector[index[f"STAMP bus capacitor {bus_number}.vc_q"]]=voltage.real
        vector[index[f"STAMP bus capacitor {bus_number}.vc_d"]]=-voltage.imag

    # With differential states fixed at their physical operating values, solve
    # all line/load powers, bus Vm/Va, and device powers consistently.
    initial_y=vector[nx:].copy()
    solved=root(lambda y: problem.rhs_algebraic(np.r_[vector[:nx],y],np.zeros_like(vector)),initial_y)
    if not solved.success:
        raise RuntimeError(solved.message)
    vector[nx:]=solved.x
    dx=np.zeros(problem.get_diff_var_number()); h=problem.get_dt_value()
    fx=problem.get_j11(vector,dx,h).toarray(); fy=problem.get_j12(vector,dx,h).toarray()
    gx=problem.get_j21(vector,dx,h).toarray(); gy=problem.get_j22(vector,dx,h).toarray()
    jac=np.block([[fx,fy],[gx,gy]]); descriptor=np.zeros_like(jac); descriptor[:nx,:nx]=np.eye(nx)
    vg_all=la.eigvals(jac,descriptor); vg=vg_all[np.isfinite(vg_all)]
    stamp=np.loadtxt(ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_A_matrix.csv",delimiter=",")
    stamp_modes=np.linalg.eigvals(stamp)
    # gy is nonsingular once capacitor power variables close each bus balance.
    reduced=fx-fy@np.linalg.solve(gy,gx)
    stamp_names=(ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_state_names.txt").read_text().splitlines()
    def canonical(name: str) -> str:
        name=(name.replace("STAMP_SG1.","SG1.").replace("STAMP_GFOR1.","GFOR1.")
                  .replace("STAMP_GFOL2.","GFOL2."))
        if name=="SG1.ig_q": return "SG1.ig_qx"
        if name=="SG1.ig_d": return "SG1.ig_dx"
        if name.startswith("NET.") and name.endswith(".iq"):
            return "NET.iq"+name.split('.')[1]
        if name.startswith("NET.") and name.endswith(".id"):
            return "NET.id"+name.split('.')[1]
        if name.startswith("STAMP bus capacitor "):
            bus=name.split()[3].split('.')[0]
            suffix=name.rsplit('.',1)[1]
            return f"vc_{'q' if suffix=='vc_q' else 'd'}{bus}"
        return name
    vg_names=[canonical(str(var)) for var in problem.state_vars]
    order=[vg_names.index(name) for name in stamp_names]
    ordered=reduced[np.ix_(order,order)]
    # Rotate all fixed-network q-d pairs into STAMP's SG-referenced frame.
    import csv
    with (ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_sg_linearization_point.csv").open(newline='',encoding='utf-8-sig') as stream:
        lp={row['field']:float(row['value']) for row in csv.DictReader(stream)}
    shift=np.arctan2(-lp['vd_bus0'],lp['vq_bus0']); cs,sn=np.cos(shift),np.sin(shift)
    rotation=np.asarray([[cs,sn],[-sn,cs]]); transform=np.eye(88); ni={name:i for i,name in enumerate(stamp_names)}
    pairs=[]
    pairs += [(f"NET.iq{edge}",f"NET.id{edge}") for edge in ('12','13','24','36','45','56')]
    pairs += [(f"vc_q{bus}",f"vc_d{bus}") for bus in range(1,7)]
    pairs += [(f"Load{load}.ilq",f"Load{load}.ild") for load in range(1,4)]
    pairs += [("SG1.ig_qx","SG1.ig_dx")]
    pairs += [(f"{dev}.{base}_q",f"{dev}.{base}_d") for dev in ('GFOR1','GFOL2') for base in ('ig','is','ucap')]
    scaled_network_pairs=set((f"vc_q{bus}",f"vc_d{bus}") for bus in range(1,7))
    scaled_network_pairs.update((f"NET.iq{edge}",f"NET.id{edge}") for edge in ('12','13','24','36','45','56'))
    scaled_network_pairs.update((f"Load{load}.ilq",f"Load{load}.ild") for load in range(1,4))
    for q,dname in pairs:
        scale=np.sqrt(2.0/3.0) if (q,dname) in scaled_network_pairs else 1.0
        transform[np.ix_([ni[q],ni[dname]],[ni[q],ni[dname]])]=scale*rotation
    ordered=transform@ordered@np.linalg.inv(transform); difference=ordered-stamp
    print(f"full Jacobian error: max={np.max(np.abs(difference)):.12g}, RMS={np.sqrt(np.mean(difference*difference)):.12g}")
    for flat in np.argsort(np.abs(difference),axis=None)[-12:][::-1]:
        r,c=np.unravel_index(flat,difference.shape)
        print(f"  d({stamp_names[r]})/d({stamp_names[c]}): VG={ordered[r,c]:+.9g}, STAMP={stamp[r,c]:+.9g}, error={difference[r,c]:+.9g}")
    print(f"states={nx}, algebraics={len(problem.algebraic_vars)}, finite={vg.size}")
    print(f"residuals: state={np.max(np.abs(problem.rhs_state(vector,np.zeros_like(vector)))):.12g}, "
          f"algebraic={np.max(np.abs(problem.rhs_algebraic(vector,np.zeros_like(vector)))):.12g}")
    residual=np.asarray(problem.rhs_state(vector,np.zeros_like(vector)))
    for i in np.argsort(np.abs(residual))[-10:][::-1]:
        print(f"  residual {problem.state_vars[i]}={residual[i]:+.12g}")
    for i,var in enumerate(problem.algebraic_vars):
        if "bus capacitor" in str(var): print(f"  {var}={vector[nx+i]:+.12g}")
    for label,modes in (("VeraGrid",vg),("STAMP",stamp_modes)):
        print(f"{label}: unstable={np.count_nonzero(modes.real>1e-8)}, "
              f"rightmost={modes[np.argmax(modes.real)]:.12g}")

if __name__=="__main__": main()
