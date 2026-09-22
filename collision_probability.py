"""Optional 2D Gaussian encounter probability from explicitly supplied covariance.

No uncertainty, object radii, or probabilities are synthesized by this module.
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from fetch_celestrak import parse_epoch


def gaussian_disk_probability(mean, covariance, radius_km):
    mean=np.asarray(mean,dtype=float);covariance=np.asarray(covariance,dtype=float)
    if mean.shape!=(2,) or covariance.shape!=(2,2) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite 2D mean and 2x2 covariance are required")
    if not np.allclose(covariance,covariance.T,rtol=1e-10,atol=1e-12) or np.linalg.eigvalsh(covariance).min()<=0:
        raise ValueError("Covariance must be symmetric positive definite")
    if not math.isfinite(radius_km) or radius_km<=0:raise ValueError("A positive measured/justified hard-body radius is required")
    sigma_x=math.sqrt(covariance[0,0])
    beta=covariance[1,0]/covariance[0,0]
    conditional_sigma=math.sqrt(covariance[1,1]-covariance[1,0]**2/covariance[0,0])
    lo=max(-12,(-radius_km-mean[0])/sigma_x)
    hi=min(12,(radius_km-mean[0])/sigma_x)
    if lo>=hi:return 0.0
    def integrand(z):
        x=mean[0]+sigma_x*z
        bound=math.sqrt(max(0,radius_km**2-x*x))
        conditional_mean=mean[1]+beta*(x-mean[0])
        lower=(-bound-conditional_mean)/conditional_sigma
        upper=(bound-conditional_mean)/conditional_sigma
        interval=ndtr(-lower)-ndtr(-upper) if lower>0 else ndtr(upper)-ndtr(lower)
        return math.exp(-0.5*z*z)/math.sqrt(2*math.pi)*interval
    probability,error=quad(integrand,lo,hi,epsabs=1e-12,epsrel=1e-7,limit=200)
    return float(np.clip(probability,0,1))


def apply_covariances(events, inputs=None):
    if inputs is None:
        return {"status":"unavailable","computed_events":0,
                "reason":"No encounter-time position covariances and combined hard-body radii supplied. No values were assumed."}
    if not isinstance(inputs.get("events"),list):raise ValueError("Covariance input requires an events array")
    used=set();computed=0
    for item in inputs["events"]:
        if item.get("frame")!="TEME" or item.get("units")!="km,km^2":
            raise ValueError("Covariances must already be in TEME with km,km^2 units")
        if item.get("cross_covariance_assumption")!="independent":
            raise ValueError("This implementation requires explicitly justified independent object errors")
        if not item.get("source") or not item.get("radius_source"):
            raise ValueError("Covariance and hard-body-radius provenance are required")
        key=tuple(sorted((int(item["object1_id"]),int(item["object2_id"]))))
        matches=[event for event in events if tuple(sorted((event["object1_id"],event["object2_id"])))==key
                 and abs((parse_epoch(event["tca_utc"])-parse_epoch(item["tca_utc"])).total_seconds())<=1]
        if len(matches)!=1:raise ValueError("Covariance must match exactly one PINN encounter within 1 second of its TCA")
        event=matches[0]
        if event["event_id"] in used:raise ValueError("Duplicate covariance input for an event")
        used.add(event["event_id"])
        combined=np.zeros((3,3))
        for name in ("covariance1_km2","covariance2_km2"):
            matrix=np.asarray(item[name],dtype=float)
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T):
                raise ValueError("Each covariance must be a finite symmetric 3x3 matrix")
            if np.linalg.eigvalsh(matrix).min()<0:raise ValueError("Covariance must be positive semidefinite")
            combined+=matrix
        relative_velocity=np.asarray(event["relative_velocity_teme_km_s"])
        speed=np.linalg.norm(relative_velocity)
        if speed<0.01:raise ValueError("Slow encounters need a different probability model; this straight-line method is unsuitable")
        normal=relative_velocity/speed
        trial=np.eye(3)[np.argmin(np.abs(normal))]
        axis1=np.cross(normal,trial);axis1/=np.linalg.norm(axis1)
        projection=np.stack((axis1,np.cross(normal,axis1)))
        mean=projection@(np.asarray(event["position1_teme_km"])-np.asarray(event["position2_teme_km"]))
        probability=gaussian_disk_probability(mean,projection@combined@projection.T,float(item["combined_hard_body_radius_km"]))
        event.update(collision_probability=probability,probability_status="computed_with_supplied_covariance_and_radius",
                     probability_source=item["source"],probability_method="2D Gaussian encounter-plane integration; independent errors, rectilinear motion")
        computed+=1
    return {"status":"computed_from_supplied_inputs","computed_events":computed,
            "note":"Conditional model probabilities depend on supplied covariance/radius validity and do not include separately calibrated PINN model error."}
