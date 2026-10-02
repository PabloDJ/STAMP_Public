#!/usr/bin/env python3
"""Compare the complete 88-state STAMP and VeraGrid dynamic networks."""
from pathlib import Path
import argparse
import sys
import numpy as np
import scipy.linalg as la
from scipy.optimize import root

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_veragrid_stamp_wscc import power_flow_options
from veragrid_stamp.wscc_case import build_stamp_wscc_grid, STAMP_LOADS


def main(nonlinear_converters: bool = False) -> None:
    from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
    from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
    grid=build_stamp_wscc_grid(dynamic_lines=True,full_dynamic_network=True,
                               nonlinear_converters=nonlinear_converters)
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
    output_dir = ROOT/"STAMP/02_results/comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix="_nonlinear" if nonlinear_converters else ""
    np.savetxt(output_dir/f"WSCC_SG_GFOR_GFOL_veragrid_full_dynamic{suffix}_eigenvalues.csv",
               np.c_[vg.real, vg.imag], delimiter=",", header="real,imag", comments="")
    if nonlinear_converters:
        from scipy.optimize import linear_sum_assignment
        from scripts.compare_stamp_veragrid_operating_point import converter_point, long_points
        from veragrid_stamp.parameters import STAMP_GFOR, STAMP_GFOL
        import csv
        with (ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_power_flow.csv").open(
                newline="", encoding="utf-8-sig") as stream:
            reference_buses=list(csv.DictReader(stream))
        voltage=np.asarray(pf.results.voltage)
        vm_error=max(abs(abs(v)-float(row["Vm"])) for v,row in zip(voltage,reference_buses))
        va_error=max(abs(np.angle(v)-np.deg2rad(float(row["theta"])))
                     for v,row in zip(voltage,reference_buses))
        vsc_ref=long_points(ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_vsc_linearization_point.csv")
        vsc_errors=[]
        for device,params in enumerate((STAMP_GFOR,STAMP_GFOL),1):
            bus_index=buses[params.bus]
            point=converter_point(abs(voltage[bus_index]),np.angle(voltage[bus_index]),
                                  params.p_pu_system,pf.results.Sbus[bus_index].imag/grid.Sbase)
            vsc_errors.extend(abs(value-vsc_ref[(device,field)]) for field,value in point.items())
        state_error=np.max(np.abs(problem.rhs_state(vector,np.zeros_like(vector))))
        alg_error=np.max(np.abs(problem.rhs_algebraic(vector,np.zeros_like(vector))))
        print(f"PF errors: Vm={vm_error:.12g}, Va={va_error:.12g} rad")
        print(f"VSC operating-point q-d error: {max(vsc_errors):.12g}")
        print(f"initial residuals: state={state_error:.12g}, algebraic={alg_error:.12g}")
        if max(vm_error,va_error,max(vsc_errors),state_error,alg_error)>1e-7:
            residual=np.asarray(problem.rhs_state(vector,np.zeros_like(vector)))
            for i in np.argsort(np.abs(residual))[-8:][::-1]:
                print(f"  residual {problem.state_vars[i]}={residual[i]:+.12g}")
            raise RuntimeError("Nonlinear comparison operating point does not match STAMP")
        cost=np.abs(stamp_modes[:,None]-vg[None,:])
        rows,cols=linear_sum_assignment(cost)
        matched=cost[rows,cols]
        print(f"mode count: STAMP={len(stamp_modes)}, VeraGrid={len(vg)}")
        print(f"eigenvalue assignment error: median={np.median(matched):.12g}, max={max(matched):.12g}")
        print(f"rightmost: STAMP={stamp_modes[np.argmax(stamp_modes.real)]:.12g}, "
              f"VeraGrid={vg[np.argmax(vg.real)]:.12g}")
        reduced=fx-fy@np.linalg.solve(gy,gx)
        stamp_names=(ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_state_names.txt").read_text().splitlines()
        def canonical_nonlinear(name: str) -> str:
            name=(name.replace("STAMP_SG1.","SG1.").replace("STAMP_GFOR1.","GFOR1.")
                      .replace("STAMP_GFOL2.","GFOL2."))
            for device in ("GFOR1", "GFOL2"):
                aliases={"theta":"etheta_x", "p_filt":"p_filt_x", "q_filt":"q_filt_x",
                         "w_filt":"w_filt_x", "igd_ff":"igd_ff_x", "igq_ff":"igq_ff_x"}
                for old,new in aliases.items():
                    if name==f"{device}.{old}": return f"{device}.{new}"
            if name=="SG1.ig_q": return "SG1.ig_qx"
            if name=="SG1.ig_d": return "SG1.ig_dx"
            if name.startswith("NET.") and name.endswith(".iq"):
                return "NET.iq"+name.split('.')[1]
            if name.startswith("NET.") and name.endswith(".id"):
                return "NET.id"+name.split('.')[1]
            if name.startswith("STAMP bus capacitor "):
                bus=name.split()[3].split('.')[0]
                return f"vc_{'q' if name.endswith('.vc_q') else 'd'}{bus}"
            return name
        vg_names=[canonical_nonlinear(str(var)) for var in problem.state_vars]
        order=[vg_names.index(name) for name in stamp_names]
        ordered=reduced[np.ix_(order,order)]
        ni={name:i for i,name in enumerate(stamp_names)}
        transform=np.eye(nx)
        with (ROOT/"STAMP/02_results/multivac/WSCC_SG_GFOR_GFOL_sg_linearization_point.csv").open(
                newline="",encoding="utf-8-sig") as stream:
            lp={row['field']:float(row['value']) for row in csv.DictReader(stream)}
        shift=np.arctan2(-lp['vd_bus0'],lp['vq_bus0'])
        cs,sn=np.cos(shift),np.sin(shift)
        rotation=np.asarray([[cs,sn],[-sn,cs]])
        fixed_pairs=[(f"NET.iq{edge}",f"NET.id{edge}") for edge in ('12','13','24','36','45','56')]
        fixed_pairs += [(f"vc_q{bus}",f"vc_d{bus}") for bus in range(1,7)]
        fixed_pairs += [(f"Load{load}.ilq",f"Load{load}.ild") for load in range(1,4)]
        fixed_pairs += [("SG1.ig_qx","SG1.ig_dx")]
        for qname,dname in fixed_pairs:
            scale=np.sqrt(2/3) if not qname.startswith("SG1.") else 1.0
            transform[np.ix_([ni[qname],ni[dname]],[ni[qname],ni[dname]])]=scale*rotation
        for device in ("GFOR1","GFOL2"):
            for base in ("ig","is","ucap"):
                qname,dname=f"{device}.{base}_q",f"{device}.{base}_d"
                ids=[ni[qname],ni[dname]]
                transform[np.ix_(ids,ids)]=rotation
        from veragrid_stamp.parameters import STAMP_GFOR, STAMP_GFOL, OMEGA_BASE
        transform[ni['GFOR1.p_filt_x'],ni['GFOR1.p_filt_x']]=STAMP_GFOR.frequency_droop_tau
        transform[ni['GFOR1.q_filt_x'],ni['GFOR1.q_filt_x']]=-STAMP_GFOR.voltage_droop_tau
        for base in ('igd','igq'):
            name=f'GFOR1.{base}_ff_x'
            transform[ni[name],ni[name]]=STAMP_GFOR.current_feedforward_tau
        transform[ni['GFOL2.w_filt_x'],ni['GFOL2.w_filt_x']]=OMEGA_BASE*STAMP_GFOL.frequency_droop_tau
        transform[ni['GFOL2.q_filt_x'],ni['GFOL2.q_filt_x']]=-STAMP_GFOL.voltage_droop_tau
        mapped=transform@ordered@np.linalg.inv(transform)
        difference=mapped-stamp
        print(f"mapped Jacobian difference: max={np.max(np.abs(difference)):.12g}, "
              f"RMS={np.sqrt(np.mean(difference*difference)):.12g}")
        for flat in np.argsort(np.abs(difference),axis=None)[-15:][::-1]:
            row,col=np.unravel_index(flat,difference.shape)
            print(f"  d({stamp_names[row]})/d({stamp_names[col]}): "
                  f"VG={mapped[row,col]:+.9g}, STAMP={stamp[row,col]:+.9g}, "
                  f"error={difference[row,col]:+.9g}")
        if np.max(np.abs(difference)) > 1e-3 or np.max(matched) > 1e-5:
            raise RuntimeError("Nonlinear converter Jacobian or eigenvalues differ from STAMP")
        return
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

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--nonlinear-converters", action="store_true")
    main(parser.parse_args().nonlinear_converters)
