
# old function for SAC dataset
def prep_wfs(wf_dir, crop_1=[-4, 10], crop_2=[-1, 1], prefilt=[2.0, 16.0], taper=.1, phase="P", debug_max_wf=-1) -> dict:
    """
    Read specially prepared SAC files from directory. Return a dict containing waveform (meta)data
    """
    wfs, evt_ids, delay = [], [], []
    files = Path(wf_dir).glob("*")
    files = list(files)

    assert phase in ("P", "S"), f"unknown phase {phase}"

    for wf_path in tqdm(files):
        sac = obs.read(str(wf_path))
        wf = sac[0]
        try:
            if phase=="S":
                pick_time: float = wf.stats.sac.t2 # t1: auto P, t2: auto S, t3: model P, t4: model S
            elif phase=="P":
                pick_time: float = wf.stats.sac.t1 # t1: auto P, t2: auto S, t3: model P, t4: model S

        except:
            continue
        else:
            if (debug_max_wf>0) & (len(wfs)==debug_max_wf):
                break
            delay.append(pick_time)
            evt_id = wf.stats.sac.user1
            fs = wf.stats.sampling_rate
            ref_time = wf.stats.starttime + pick_time

            t1, t2 = ref_time + crop_1[0], ref_time + crop_1[1]
            wf = wf.slice(t1, t2)
            wf = wf.detrend('demean')
            wf.taper(taper, type='cosine')
            wf = wf.filter('bandpass', freqmin=prefilt[0], freqmax=prefilt[1], zerophase=True)

            t3, t4 = ref_time + crop_2[0], ref_time + crop_2[1]
            wf = wf.slice(t3, t4)
            wf.taper(taper, type='cosine')
            wfs.append(wf.data)
            evt_ids.append(int(evt_id))

    # sort all lists according to event ID, important for writing out results later
    evt_ids, delay, wfs = sort_by_event_id(evt_ids, delay, wfs)

    data = {
        "evt_ids": evt_ids,
        "delay": np.array(delay),
        "waveforms": np.array(wfs),
        "count": len(wfs),
        "phase": phase,
        "wf_dir": wf_dir,
        "npts": np.array(wfs).shape[1],
        "fs": fs
    }
    return data

