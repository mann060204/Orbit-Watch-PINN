"""Independent reconstructed-orbit loading, frame conversion, and shared metrics."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from fetch_celestrak import parse_epoch, utc_text


def read_esa_orbit(path):
    root = ET.parse(Path(path)).getroot()
    product = root.findtext(".//Fixed_Header/File_Type")
    if product not in {"AUX_RESORB", "AUX_POEORB"}:
        raise ValueError("Only reconstructed/restituted or precise ESA orbit products are accepted")
    if root.findtext(".//Variable_Header/Ref_Frame") != "EARTH_FIXED":
        raise ValueError("Expected explicit EARTH_FIXED reference frame")
    if root.findtext(".//Variable_Header/Time_Reference") != "UTC":
        raise ValueError("Expected explicit UTC time reference")
    times, r, v, dut1, quality = [], [], [], [], []
    osv_list = root.find(".//List_of_OSVs")
    for osv in osv_list.findall("OSV"):
        for name in ("X", "Y", "Z", "VX", "VY", "VZ"):
            expected_unit = "m/s" if name.startswith("V") else "m"
            if osv.find(name).get("unit") != expected_unit:
                raise ValueError(f"Incorrect units for {name}")
        times.append(parse_epoch(osv.findtext("UTC").removeprefix("UTC=")))
        ut1 = parse_epoch(osv.findtext("UT1").removeprefix("UT1="))
        dut1.append((ut1-times[-1]).total_seconds())
        r.append([float(osv.findtext(name))/1000 for name in ("X", "Y", "Z")])
        v.append([float(osv.findtext(name))/1000 for name in ("VX", "VY", "VZ")])
        quality.append(osv.findtext("Quality"))
    r, v = np.array(r), np.array(v)
    if len(times) != int(osv_list.get("count")) or len(times)<2:
        raise ValueError("Unexpected reference record count")
    if any(a>=b for a, b in zip(times, times[1:])):
        raise ValueError("Reference times must be unique and strictly increasing")
    if not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite reference states")
    return {"times":times, "r_itrs_km":r, "v_itrs_km_s":v, "dut1_seconds":np.array(dut1),
            "quality":np.array(quality), "product":product, "mission":root.findtext(".//Fixed_Header/Mission")}


def select_native_samples(reference, epoch, hours=1, step_seconds=60):
    if not np.isfinite(hours) or not 0 < hours <= 3:
        raise ValueError("Use a positive duration of at most three hours")
    if not np.isfinite(step_seconds) or not 10 <= step_seconds <= 60:
        raise ValueError("Use a step of 10 to 60 seconds")
    times = reference["times"]
    candidates = [i for i, timestamp in enumerate(times) if timestamp >= epoch]
    if not candidates:
        raise ValueError("Reference ends before the GP epoch")
    first = candidates[0]
    elapsed = np.array([(timestamp-times[first]).total_seconds() for timestamp in times])
    target = np.arange(0, hours*3600+1e-6, step_seconds)
    if abs(target[-1]-hours*3600)>1e-6:
        raise ValueError("Duration must be an exact multiple of the time step")
    indices = np.searchsorted(elapsed, target)
    if np.any(indices >= len(times)) or not np.allclose(elapsed[indices], target, rtol=0, atol=1e-6):
        raise ValueError("Requested native reference timestamps are unavailable; no interpolation/extrapolation is performed")
    if not np.all(reference["quality"][indices] == "NOMINAL"):
        raise ValueError("Only NOMINAL reference states are used")
    return indices, target


def reference_to_teme(reference, indices, eop_path):
    from astropy import units as u
    from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
    from astropy.time import Time
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    eop = iers.IERS_A.open(str(eop_path))
    times = Time([reference["times"][i] for i in indices], scale="utc")
    # The ESA orbit product provides UT1 for each OSV; use its actual UT1-UTC.
    times.delta_ut1_utc = reference["dut1_seconds"][indices]
    xp, yp, flags = eop.pm_xy(times, return_status=True)
    if np.any(flags < 0):
        raise ValueError("Earth orientation table does not cover the benchmark")
    r, v = reference["r_itrs_km"][indices], reference["v_itrs_km_s"][indices]
    state = CartesianRepresentation(r.T*u.km, differentials=CartesianDifferential(v.T*u.km/u.s))
    with iers.earth_orientation_table.set(eop):
        transformed = ITRS(state, obstime=times).transform_to(TEME(obstime=times))
        back = transformed.transform_to(ITRS(obstime=times))
    transformed_r = transformed.cartesian.xyz.to_value(u.km).T
    transformed_v = transformed.cartesian.differentials["s"].d_xyz.to_value(u.km/u.s).T
    r_error = float(np.max(np.abs(back.cartesian.xyz.to_value(u.km).T-r)))
    v_error = float(np.max(np.abs(back.cartesian.differentials["s"].d_xyz.to_value(u.km/u.s).T-v)))
    if r_error>1e-7 or v_error>1e-7:
        raise ValueError("State frame-transformation round trip failed")
    info = {"from":"ITRS / CPOD ITRF2020", "to":"TEME", "positions_and_velocities_transformed":True,
            "ut1_source":"ESA OSV UT1 minus UTC", "polar_motion_source":"IERS finals2000A",
            "eop_status_legend":{"0":"IERS B", "1":"IERS A", "2":"IERS A prediction"},
            "polar_motion_status_counts":{str(int(flag)):int(np.count_nonzero(flags==flag)) for flag in np.unique(flags)},
            "roundtrip_maximum_position_difference_km":r_error, "roundtrip_maximum_velocity_difference_km_s":v_error,
            "note":"Standard geocentric ITRS/TEME transformation; residual orbit, Earth-orientation and terrestrial-realization uncertainties remain."}
    eop_rows = [{"utc":utc_text(reference["times"][index]), "ut1_minus_utc_seconds":float(reference["dut1_seconds"][index]),
                 "polar_motion_x_arcsec":float(xp[j].to_value(u.arcsec)), "polar_motion_y_arcsec":float(yp[j].to_value(u.arcsec)),
                 "polar_motion_status":int(flags[j])} for j, index in enumerate(indices)]
    return transformed_r, transformed_v, info, eop_rows


def orbit_error_metrics(predicted_r, predicted_v, reference_r, reference_v):
    arrays = [np.asarray(a, dtype=float) for a in (predicted_r, predicted_v, reference_r, reference_v)]
    shape = arrays[0].shape
    if len(shape)!=2 or shape[1]!=3 or shape[0]<2 or any(a.shape!=shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Four matching finite N by 3 state arrays are required")
    pr, pv, rr, rv = arrays
    radius, angular_momentum = np.linalg.norm(rr, axis=1), np.cross(rr, rv)
    h = np.linalg.norm(angular_momentum, axis=1)
    if np.any(radius<=0) or np.any(h<=0):
        raise ValueError("Reference cannot define an RTN basis")
    radial, normal = rr/radius[:, None], angular_momentum/h[:, None]
    transverse = np.cross(normal, radial)
    basis = np.stack((radial, transverse, normal), axis=1)
    dr, dv = pr-rr, pv-rv
    pe, ve = np.linalg.norm(dr, axis=1), np.linalg.norm(dv, axis=1)
    rtn_r = np.einsum("nij,nj->ni", basis, dr)
    rtn_v = np.einsum("nij,nj->ni", basis, dv)
    metrics = {"future_samples":shape[0]-1, "initial_position_error_km":float(pe[0]),
               "initial_velocity_error_km_s":float(ve[0]), "position_vector_rmse_km":float(np.sqrt(np.mean(pe[1:]**2))),
               "position_vector_mae_km":float(np.mean(pe[1:])), "velocity_vector_rmse_km_s":float(np.sqrt(np.mean(ve[1:]**2))),
               "velocity_vector_mae_km_s":float(np.mean(ve[1:])), "position_p95_km":float(np.quantile(pe[1:], .95)),
               "position_max_km":float(np.max(pe[1:])), "velocity_p95_km_s":float(np.quantile(ve[1:], .95)),
               "velocity_max_km_s":float(np.max(ve[1:])), "final_position_error_km":float(pe[-1]),
               "final_velocity_error_km_s":float(ve[-1]),
               "position_rtn_rmse_km":np.sqrt(np.mean(rtn_r[1:]**2, axis=0)).tolist(),
               "position_rtn_mae_km":np.mean(np.abs(rtn_r[1:]), axis=0).tolist(),
               "velocity_rtn_rmse_km_s":np.sqrt(np.mean(rtn_v[1:]**2, axis=0)).tolist(),
               "velocity_rtn_mae_km_s":np.mean(np.abs(rtn_v[1:]), axis=0).tolist(),
               "position_xyz_rmse_km":np.sqrt(np.mean(dr[1:]**2, axis=0)).tolist(),
               "position_xyz_mae_km":np.mean(np.abs(dr[1:]), axis=0).tolist(),
               "velocity_xyz_rmse_km_s":np.sqrt(np.mean(dv[1:]**2, axis=0)).tolist(),
               "velocity_xyz_mae_km_s":np.mean(np.abs(dv[1:]), axis=0).tolist(),
               "position_rtn_bias_km":np.mean(rtn_r[1:], axis=0).tolist(),
               "position_xyz_bias_km":np.mean(dr[1:], axis=0).tolist(),
               "rtn_axes":["radial", "transverse", "normal"],
               "note":"Metrics exclude t0, reported separately. Vector MAE is mean Euclidean norm; component MAE is mean absolute component. Velocity RTN is projected inertial velocity error, not time derivative of rotating RTN coordinates."}
    errors = {"position_xyz_km":dr, "velocity_xyz_km_s":dv, "position_norm_km":pe,
              "velocity_norm_km_s":ve, "position_rtn_km":rtn_r, "velocity_rtn_km_s":rtn_v}
    return metrics, errors
