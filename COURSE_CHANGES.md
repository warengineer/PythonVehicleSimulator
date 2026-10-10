# Branch `course-v1.0`: changes and notes

This branch is the version of the Python Vehicle Simulator used by the course *Marine Craft Hydrodynamics and Motion
Control* (Prof. Dr. Serhat YILMAZ, Volkan AKILLI). It starts from commit
[a93d1938ba](https://github.com/cybergalactic/PythonVehicleSimulator/commit/a93d1938ba) of
[cybergalactic/PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator) (26 Sep 2026) and adds
34 commits. Most of them add the MSS functions that the course uses to the library `python_vehicle_simulator.lib`; a few
correct or extend existing code. The vehicle classes in `vehicles/` are kept, apart from the changes listed in Part 2.
The `master` branch of this fork is the upstream commit, so
[master...course-v1.0](https://github.com/warengineer/PythonVehicleSimulator/compare/master...course-v1.0) shows every
changed line. The package and these changes are under the licence of the package ([LICENSE](LICENSE), MIT text).

The course compares the functions that follow MSS m-files with MATLAB R2026b runs of those m-files on its MSS fork
([warengineer/MSS](https://github.com/warengineer/MSS/tree/course-v1.0), branch `course-v1.0`).

## Part 1: Added to `python_vehicle_simulator.lib`

Functions that follow an MSS m-file keep its name where possible and name the m-file in their docstring. `numerics.py`
holds numpy counterparts of the MATLAB functions that the m-files call; `allocation.py` and `actuator.py` also hold
allocation methods and actuator models used by the course.

| Module | Added |
|---|---|
| `numerics.py` (new) | `rk4`, `euler2`, `expm`, `expm_taylor`, `invQR`, `c2d`, `c2d_foh`, `lqr`, `care`, `lyap`, `ctrb`, `obsv`, `rank_sv`, `damp`, `freqresp`, `tf_siso`, `interp1`, `interp1_extrap`, `pchip`, `spline_nak`, `levenberg_marquardt`, `mround`, `vnorm` |
| `craft.py` (new) | Vessel models as functions: `container`, `container_parts`, `Lcontainer`, `RRDcontainer`, `mariner`, `tanker`, `frigate`, `frigate_KTn3`, `zeefakkel`, `zeefakkel_KT`, `navalvessel`, `supply`, `supply_matrices`, `npsauv`, `npsauv_B_delta`, `npsauv_matrices`, `remus100`, `remus100_parts`, `remus_propeller`, `DSRV`, `otter`, `otter_matrices`, `otter_thrust`; classes `Otter` and `OSVModel` (`osv.m`) |
| `allocation.py` (new) | `thrConfig`, `allocPseudoinverse`, `ginv`, `alloc_clip`, `alloc_scaled`, `alloc_rpi`, `alloc_minmax`, `alloc_qp`, `qp_box_eq`, `attainable_scale`, `T_extended`, `alloc_extended`, `dls`, `OSVAlloc` (the allocation of `SIMosv.m`, with optional SQP iterations) |
| `environment.py` (new) | `encounter`, `ISOmsi`, `HMmsi`, `blendermann94`, `isherwood72`, `waveSpectrum`, `torsetSpectrum`, `waveDirectionalSpectrum`, `vw2hs`, `hs2vw`, `ForceRAO` (first-order wave loads from RAO tables) |
| `ins.py` (new) | `gravity`, `insSignal`, `ins_signal_accel`, `ins_signal_rate`, `magneticField`, `acc2rollpitch`, `staticRollPitchYaw`, `euler_rate_jacobian`, `quatObserver`, `quatMEKF`, `ins_ahrs`, `ins_euler`, `ins_mekf`, `ins_mekf_psi`, `ins_heave` |
| `navigation.py` (new) | `LowPass` (`makeLowPass.m`), `lowPassFilter`, `highPassFilter`, `notchFilter`, `waveFreqObserver`, `sawToothWave`, `EKF_5states` |
| `maneuvering.py` (new) | `turncircle`, `zigzag`, `pullout` |
| `hydro.py` (new) | `read_wamit` (WAMIT v6 files, as `wamit2vessel.m`), `power_based` and `pm_weight` (as `computeManeuveringModel.m`) |
| `gnc.py` | `eulerang`, `Rquat`, `Tquat`, `quatern`, `quatprod`, `euler2q`, `q2euler`, `R2euler`, `expq`, `quat_step`, `gRvect`, `Rz`, `S3`, `satlim` |
| `guidance.py` | `refModel`, `LOSchi`, `ILOSpsi`, `ALOSpsi`, `ALOS3D`, `LOSobserver`, `crosstrackWpt`, `crosstrackWpt3D`, `addIntermediateWaypoints` |
| `control.py` | `lqtracker`, `IntegralSMCheading`, `PIDnonlinearMIMO` |
| `models.py` | `spheroid`, `imlay61`, `Dmtrx`, `Gmtrx`, `rbody`, `coeffLiftDrag`, `cylinderDrag`, `addedMassSurge`, `XuuITTC`, `clarke83_derivatives`, `lamb_k` |
| `actuator.py` | `steering_machine`, `rate_limit_step`, `dc_motor`, `gyrostabilizer`, `cmg_pyramid` |

The functions follow the corrected m-files of the MSS fork (for example `isherwood72`, `blendermann94`, `HMmsi`,
`euler2q` and `crosstrackWpt3D`).

## Part 2: Changes to the upstream code

| Commit | File | Change |
|---|---|---|
| [5152f44](https://github.com/warengineer/PythonVehicleSimulator/commit/5152f44) | `lib/gnc.py` (`sat`) | Also the symmetric form `sat(x, x_max)` of MSS `sat.m`; arrays element by element. The form `sat(x, x_min, x_max)` works as before. |
| [bfed7d3](https://github.com/warengineer/PythonVehicleSimulator/commit/bfed7d3) | `lib/gnc.py` (`Hoerner`) | The digitized data of MSS `Hoerner.m` at full precision (they were rounded to four decimals). |
| [c7494bf](https://github.com/warengineer/PythonVehicleSimulator/commit/c7494bf) | `lib/gnc.py` (`crossFlowDrag`), `tests/test_crossflow_drag.py` | As MSS `crossFlowDrag.m`: heave force and pitch moment besides sway force and yaw moment, ρ = 1025 kg/m³, drag models `'Hoerner'` and `'cylinder'` (`models.cylinderDrag`). |
| [ba030c1](https://github.com/warengineer/PythonVehicleSimulator/commit/ba030c1) | `lib/gnc.py` (`forceLiftDrag`) | As MSS `forceLiftDrag.m`: the coefficients of `models.coeffLiftDrag` (Oswald efficiency e = 0.3, as `coeffLiftDrag.m`) and an argument `rho` (default 1026 kg/m³). |
| [751ce48](https://github.com/warengineer/PythonVehicleSimulator/commit/751ce48) | `.gitignore` | Ignores only the top-level `lib/` and `lib64/`; the pattern `lib/` also ignored new modules in `src/python_vehicle_simulator/lib`. |
| [a43a8e9](https://github.com/warengineer/PythonVehicleSimulator/commit/a43a8e9) | `lib/gnc.py` (`m2c`) | 3-DOF case with the full momenta p = M ν, as the current MSS `m2c.m`; the result is the same when M(1,2) = M(1,3) = 0. |
| [70f2229](https://github.com/warengineer/PythonVehicleSimulator/commit/70f2229) | `vehicles/otter.py` | KB from the pontoon waterplane area C_w,pont L B_pont, as MSS `otter.m` (KB = 0.12791 m; it was 0.13659 m with the rectangle L B_pont). |
| [2904746](https://github.com/warengineer/PythonVehicleSimulator/commit/2904746) | `vehicles/otter.py` (`controlAllocation`) | A propeller turning astern is allocated with k_neg = 0.5817 k_pos, as `SIMotter.m`: τ = [100 N, 50 N m] is delivered exactly (it was 105.56 N and 47.80 N m). |
| [98f3d7c](https://github.com/warengineer/PythonVehicleSimulator/commit/98f3d7c) | `lib/control.py`, `vehicles/frigate.py`, `vehicles/ROVzefakkel.py`, `vehicles/otter.py`, `vehicles/shipClarke83.py`, `vehicles/tanker.py` | Docstrings only: the heading autopilots are a feedback-only PID on the unwrapped error ψ − ψ_d (no reference feedforward); the code is unchanged. |

The vehicle classes that call `crossFlowDrag` (`otter.py`, `remus100.py`, `torpedo.py`) and `forceLiftDrag`
(`remus100.py`, `torpedo.py`) use the MSS forms of these functions on this branch.

## Part 3: All commits

| Commit | Subject |
|---|---|
| [5152f44](https://github.com/warengineer/PythonVehicleSimulator/commit/5152f44) | `gnc.sat`: symmetric form as MSS `sat.m`, arrays element by element |
| [bfed7d3](https://github.com/warengineer/PythonVehicleSimulator/commit/bfed7d3) | `gnc.Hoerner`: the data of MSS `Hoerner.m` at full precision |
| [c7494bf](https://github.com/warengineer/PythonVehicleSimulator/commit/c7494bf) | `gnc.crossFlowDrag` as MSS `crossFlowDrag.m` |
| [ba030c1](https://github.com/warengineer/PythonVehicleSimulator/commit/ba030c1) | `gnc.forceLiftDrag` as MSS `forceLiftDrag.m` |
| [751ce48](https://github.com/warengineer/PythonVehicleSimulator/commit/751ce48) | `.gitignore`: only the top-level `lib/` and `lib64/` |
| [a367476](https://github.com/warengineer/PythonVehicleSimulator/commit/a367476) | MSS functions: numerics, craft (Mariner, container ship, Zeefakkel, supply vessel, REMUS 100), allocation, environment, INS gravity, low-pass filter; kinematics, models, control and guidance functions |
| [a7f98ba](https://github.com/warengineer/PythonVehicleSimulator/commit/a7f98ba) | Tanker and DSRV models, semisubmersible matrices, turning circle and zigzag, `lowPassFilter`; `PIDnonlinearMIMO` with 6-DOF inputs |
| [d39cf3b](https://github.com/warengineer/PythonVehicleSimulator/commit/d39cf3b) | INS functions of MSS, quaternion kinematics, `numerics.expm_taylor` |
| [b179853](https://github.com/warengineer/PythonVehicleSimulator/commit/b179853) | Signal filters and wave-frequency observer of MSS, `numerics.obsv`, `numerics.freqresp` |
| [4c18278](https://github.com/warengineer/PythonVehicleSimulator/commit/4c18278) | Otter USV as functions and a class, LOS guidance laws and waypoint functions of MSS, `pchip`, `spline_nak` |
| [406c925](https://github.com/warengineer/PythonVehicleSimulator/commit/406c925) | Control allocation methods, `rbody`, `Gmtrx`, `XuuITTC`, `OSVModel`; `craft.supply` returns (xdot, U, M, D) as `supply.m` |
| [1a9bdf1](https://github.com/warengineer/PythonVehicleSimulator/commit/1a9bdf1) | Environment models (wind loads, wave spectra, first-order wave loads), `interp1`; `craft.otter` options and outputs |
| [fa3215a](https://github.com/warengineer/PythonVehicleSimulator/commit/fa3215a) | Naval vessel and NPS AUV models, REMUS 100 propeller, actuator models, `allocation.dls`; `container_parts` |
| [951dc5c](https://github.com/warengineer/PythonVehicleSimulator/commit/951dc5c) | `gnc.quat_step`: exact quaternion step for a constant angular velocity |
| [542cd53](https://github.com/warengineer/PythonVehicleSimulator/commit/542cd53) | Frigate model, `pullout`, `EKF_5states`, `levenberg_marquardt` |
| [c5233a6](https://github.com/warengineer/PythonVehicleSimulator/commit/c5233a6) | `hydro` (WAMIT reader, power-based matrices), `models.clarke83_derivatives` |
| [2a92211](https://github.com/warengineer/PythonVehicleSimulator/commit/2a92211) | `hydro.read_wamit`: one reader for all WAMIT v6 files used |
| [c1871d7](https://github.com/warengineer/PythonVehicleSimulator/commit/c1871d7) | `craft.OSVModel` keeps the linear surge damping in every call, as `osv.m` |
| [591f747](https://github.com/warengineer/PythonVehicleSimulator/commit/591f747) | `environment`: `ForceRAO` with the wave model of MSS; wind loads and `HMmsi` as corrected in the MSS fork |
| [7fa4843](https://github.com/warengineer/PythonVehicleSimulator/commit/7fa4843) | `gnc.euler2q` and `guidance.crosstrackWpt3D` as corrected in the MSS fork |
| [496a330](https://github.com/warengineer/PythonVehicleSimulator/commit/496a330) | `OSVAlloc`: optional SQP iterations |
| [a43a8e9](https://github.com/warengineer/PythonVehicleSimulator/commit/a43a8e9) | `gnc.m2c`: 3-DOF case with the full momenta |
| [9be0a51](https://github.com/warengineer/PythonVehicleSimulator/commit/9be0a51) | Docstring of `gnc.euler2q` |
| [c2b247c](https://github.com/warengineer/PythonVehicleSimulator/commit/c2b247c) | Docstring of `navigation.EKF_5states` |
| [07c31fe](https://github.com/warengineer/PythonVehicleSimulator/commit/07c31fe) | Docstrings of `hydro` |
| [1f3d968](https://github.com/warengineer/PythonVehicleSimulator/commit/1f3d968) | Docstring of `models.cylinderDrag` |
| [52bea02](https://github.com/warengineer/PythonVehicleSimulator/commit/52bea02) | Comment of the air density in `environment` |
| [70f2229](https://github.com/warengineer/PythonVehicleSimulator/commit/70f2229) | `otter.py`: KB from the pontoon waterplane area |
| [2904746](https://github.com/warengineer/PythonVehicleSimulator/commit/2904746) | `otter.py`: k_neg for a propeller turning astern |
| [98f3d7c](https://github.com/warengineer/PythonVehicleSimulator/commit/98f3d7c) | Docstrings of the heading autopilots |
| [bdc8d52](https://github.com/warengineer/PythonVehicleSimulator/commit/bdc8d52) | Docstrings of `allocation` and `craft` cite the papers |
| [a1d6657](https://github.com/warengineer/PythonVehicleSimulator/commit/a1d6657) | Docstrings of `navigation`, `environment`, `control` and `craft` |
| [e46a434](https://github.com/warengineer/PythonVehicleSimulator/commit/e46a434) | Comments of `guidance`, `craft` and `gnc` |
| [4aee4d9](https://github.com/warengineer/PythonVehicleSimulator/commit/4aee4d9) | `numerics.expm_taylor`: factorials above 170 as infinity, as MATLAB (numpy 1.26 raised a TypeError) |

## Part 4: Observations, not changed

The existing vehicle classes differ from the MSS m-files in some points. They are kept as they are; the functions of
Part 1 (for example `craft.otter`, `craft.remus100`, `craft.frigate`) follow the m-files.

| File | Observation |
|---|---|
| `vehicles/otter.py` vs MSS `otter.m`, `SIMotter.m` | X_u̇ = −0.1 m (5.5 kg) instead of the formula of `otter.m` (5.28 kg); the heading autopilot uses ω_n = 2.5 rad/s (`SIMotter.m`: 1.5 rad/s). The current acceleration enters only through the yaw rate, ν̇_c = [r v_c, −r u_c, 0, 0, 0, 0], while `otter.m` uses −S(ω) v_c with all three angular velocities. |
| `vehicles/supply.py` (with `control.DPpolePlacement`) vs MSS `SIMsupply.m` | Same model and thruster data, but the allocation u = Bᵀ(BBᵀ)⁻¹τ with B = T K minimizes Σ u_i² (`SIMsupply.m` uses `allocPseudoinverse` with W = I, Σ f_i²): for [100 kN, 200 kN, 5 MN m] the main propellers get −204.586 and 304.586 kN. K_d = 2ζω_n M_diag − D_diag, and the setpoint filters have the time constants 5/ω_n (50, 50, 25 s) instead of T_f = 10 s. |
| `vehicles/remus100.py` vs MSS `remus100.m`, `SIMremus100.m` | Spheroid with a = L/2, b = D/2 (31.03 kg) and g = 9.81 m/s² (MSS: scale 1.0096, 31.9 kg, gravity at Trondheim); fins limited to 15° with fin and propeller lags (MSS: 20°); the linear sway damping is also faded; Hoerner's cross-flow coefficient instead of the DNV cylinder data; no vertical current and no unit quaternions; `depthHeadingAutopilot` by successive loop closure instead of the gradient-based depth control of `SIMremus100.m`. K_d = 0.5 and K_σ = 0.05 act together, so the gain inside the boundary layer is 1.0, twice that of either `SIMremus100.m` method. |
| `vehicles/frigate.py`, `vehicles/ROVzefakkel.py` vs MSS `frigate.m`, `zeefakkel.m` | Outside the speed table the constants are clamped (`np.interp`; frigate at 5 m/s: K = 0.08 1/s, T = 20 s), while MSS extrapolates; `frigate.py` runs at 5 m/s by default, below the data (6–12 m/s). The autopilots use ω_n = 0.4 rad/s, ω_n,d = ω_n/5 and r_max = 1°/s (`SIMfrigate.m`: 0.5 rad/s, 0.1 rad/s and 5°/s). |
| `vehicles/DSRV.py` | As MSS `DSRV.m`: non-dimensional (prime-system) coefficients in dimensional state equations, without the factors U/L and U²/L. |
