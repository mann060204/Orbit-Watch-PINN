"""Conjunction screening using PINN predictions only; SGP4 is not called here."""
from datetime import timedelta

import numpy as np

from fetch_celestrak import utc_text
from orbit_engine import interval_candidates
from pinn_model import LENGTH_KM, SPEED_KM_S, predict_step


def refine_batch(model, states_a, states_b, duration, device):
    n = len(states_a)
    def separation(times, return_states=False):
        a = predict_step(model, states_a, times, device)
        b = predict_step(model, states_b, times, device)
        distance = np.linalg.norm(a[:,:3]-b[:,:3],axis=1)*LENGTH_KM
        return (distance, a, b) if return_states else distance
    left = np.zeros(n)
    right = np.full(n, duration, dtype=float)
    ratio = (np.sqrt(5)-1)/2
    c = right-ratio*(right-left)
    d = left+ratio*(right-left)
    fc,fd = separation(c), separation(d)
    for _ in range(27):
        choose_left = fc < fd
        right = np.where(choose_left,d,right)
        left = np.where(choose_left,left,c)
        c = right-ratio*(right-left)
        d = left+ratio*(right-left)
        fc,fd = separation(c),separation(d)
    times = (left+right)/2
    interior = separation(times)
    first,last = separation(np.zeros(n)),separation(np.full(n,duration))
    times = np.where((first<=interior)&(first<=last),0,np.where(last<=interior,duration,times))
    distance,a,b = separation(times,return_states=True)
    return times,distance,a,b


def screen_pinn(model, trajectories, rows, start, seconds, threshold_km, shared_pairs, device,
                report=lambda *args:None):
    excluded = {tuple(sorted((p["object1_id"],p["object2_id"]))) for p in shared_pairs}
    candidates = []
    for interval in range(len(seconds)-1):
        pairs,_,_ = interval_candidates(trajectories[:,interval,:3]*LENGTH_KM,
                                        trajectories[:,interval+1,:3]*LENGTH_KM,
                                        threshold_km,0.05,float(seconds[interval+1]-seconds[interval]))
        for a,b in pairs:
            if tuple(sorted((int(rows[a]["NORAD_CAT_ID"]),int(rows[b]["NORAD_CAT_ID"])))) not in excluded:
                candidates.append((int(a),int(b),interval))
    if not candidates:
        return [],[],{"candidate_intervals":0,"method":"PINN swept chords + batched direct-PINN time refinement"}
    triples = np.array(candidates,dtype=int)
    refined=[]
    for offset in range(0,len(triples),2048):
        batch=triples[offset:offset+2048]
        a,b,index=batch.T
        durations=seconds[index+1]-seconds[index]
        # Normal grids have one duration. Handle the optional short final interval separately.
        for duration in np.unique(durations):
            selection=durations==duration
            aa,bb,ii=a[selection],b[selection],index[selection]
            t,distance,pa,pb=refine_batch(model,trajectories[aa,ii],trajectories[bb,ii],float(duration),device)
            for j in range(len(t)):
                refined.append({"a":int(aa[j]),"b":int(bb[j]),"interval":int(ii[j]),
                                "offset_seconds":float(seconds[ii[j]]+t[j]),"miss_distance_km":float(distance[j]),
                                "relative_speed_km_s":float(np.linalg.norm(pa[j,3:]-pb[j,3:])*SPEED_KM_S),
                                "relative_velocity_teme_km_s":((pa[j,3:]-pb[j,3:])*SPEED_KM_S).tolist(),
                                "position1_teme_km":(pa[j,:3]*LENGTH_KM).tolist(),
                                "position2_teme_km":(pb[j,:3]*LENGTH_KM).tolist()})
        report(f"Refined {min(offset+2048,len(triples)):,}/{len(triples):,} neural candidate intervals")
    by_pair={}
    for candidate in refined:
        by_pair.setdefault((candidate["a"],candidate["b"]),[]).append(candidate)
    events=[]
    for (a,b),values in by_pair.items():
        values.sort(key=lambda row:row["interval"])
        episodes=[]
        for value in values:
            if not episodes or value["interval"] > episodes[-1][-1]["interval"]+1:
                episodes.append([])
            episodes[-1].append(value)
        for episode in episodes:
            best=min(episode,key=lambda x:x["miss_distance_km"])
            if best["miss_distance_km"] > threshold_km:
                continue
            events.append({"object1_id":int(rows[a]["NORAD_CAT_ID"]),"object2_id":int(rows[b]["NORAD_CAT_ID"]),
                           "object1_name":rows[a].get("OBJECT_NAME",""),"object2_name":rows[b].get("OBJECT_NAME",""),
                           "tca_utc":utc_text(start+timedelta(seconds=best["offset_seconds"])),
                           **{key:best[key] for key in ("miss_distance_km","relative_speed_km_s","offset_seconds",
                                                       "position1_teme_km","position2_teme_km","relative_velocity_teme_km_s")},
                           "method":"PINN neural propagation and direct neural time refinement",
                           "collision_probability":None,"probability_status":"missing_covariance_and_hard_body_radius"})
    events.sort(key=lambda event:event["miss_distance_km"])
    for index,event in enumerate(events):event["event_id"]=f"PINN-{index+1:05d}"
    return events,refined,{"candidate_intervals":len(triples),"method":"PINN swept chords + batched direct-PINN time refinement"}
