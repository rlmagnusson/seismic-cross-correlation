from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any
import pandas as pd
import obspy as obs
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Union, Tuple
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser, Namespace
from pathlib import Path
import h5py
import pickle
import os, sys
import time

from .processing import *



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


def read_waveforms_sac(wf_dir: str) -> WaveformData:
    pass


def read_waveforms_h5(station: str, channel: str, file_path: str, target_samples: int = 0) -> WaveformData:
    group_name = f"{station}/{channel}"

    event_ids = []
    waveform_arrays = []
    pick_offsets = []
    event_offsets = []
    start_times = []
    original_sampling_rate = []

    with h5py.File(file_path, "r") as f:
        group = f[group_name]
        for event_name in group:
            dset = group[event_name]
            samples = dset[()]
            meta = {key: dset.attrs[key] for key in dset.attrs}

            if len(samples) >= target_samples:
                event_ids.append(meta["event_number"])
                if target_samples > 0:
                    samples = samples[:target_samples]
                waveform_arrays.append(samples)
                pick_offsets.append(meta["pick_offset"])
                event_offsets.append(meta["event_offset"])
                start_times.append(obs.UTCDateTime(meta["start_time"]))
                original_sampling_rate.append(meta["original_sampling_rate"])

    waveforms = np.array(waveform_arrays)

    wd = WaveformData(
        station=station,
        channel=channel,
        waveforms=np.array(waveform_arrays),
        pick_offsets=pick_offsets,
        event_offsets=event_offsets,
        event_ids=event_ids,
        count=len(event_ids),
        phase=meta["phase"], # TODO
        npts=len(waveform_arrays[0]),
        sampling_rate=meta["sampling_rate"],
        original_sampling_rate=original_sampling_rate,
        start_times=start_times
    )
    return wd



def run_xcorr(
        source: Union[str, dict], 
        crop_1: List[float] = [-4.0, 10.0], 
        crop_2: List[float] = [-1.0, 1.0], 
        prefilt: List[float] = [2.0, 16.0], 
        taper: float = .1, 
        phase: str = "P", 
        max_wf: int = -1,
        fft_size: Optional[int] = None,
        n_procs: Optional[int] = None
    ) -> dict:
    if isinstance(source, dict):
        wfs = source
    else:
        print("preprocessing...")
        wfs = prep_wfs(wf_dir=wf_dir, crop_1=crop_1, crop_2=crop_2, prefilt=prefilt,
                       taper=taper, phase=phase, debug_max_wf=max_wf)
    matrices = fft_wfs(wfs, fft_size=fft_size)
    print("cross-correlating...")
    if n_procs:
        lag_matrix, coeff_matrix = xcorr_MP(matrices, n_procs=n_procs)
    else:
        lag_matrix, coeff_matrix = correlation_by_matrix(matrices)
        
    out = {"lag_matrix": lag_matrix, "coeff_matrix": coeff_matrix, "event_ids": wfs["evt_ids"]}
    return out



def to_csv(xcorr_results: CorrelationData, filename: str, station: str, phase: str, min_C: float = .6, max_dt: float = .5) -> None:
    """
    Write the cross correlation results to csv file with the columns:
    evt1, evt2, coeff, dtcc, station, phase
    Where evt1 and evt2 are the event IDs, coeff is the maximum correlation coefficient,
    dtcc is the time lag (TODO check evt1-evt2 or evt2-evt1), station is the station code
    and phase (P or S) is the phase type being cross correlated
    """
    lm = xcorr_results.lag_time
    cm = xcorr_results.coefficient
    eid = xcorr_results.event_ids

    N = len(eid)
    xcorr_lines = []
    low_coeff, large_dt = 0, 0
    for i in range(1, N):
        for j in range(i):
            correlation_coeff = cm[i, j]
            if correlation_coeff < min_C:
                low_coeff += 1
                continue
            lag_time = lm[i, j]
            if abs(lag_time) > max_dt:
                large_dt += 1
                continue
            ev1 = eid[i]
            ev2 = eid[j]
            line = (ev1, ev2, correlation_coeff, lag_time)
            xcorr_lines.append(line)

    df = pd.DataFrame(xcorr_lines, columns=["evt1", "evt2", "coeff", "dtcc"])
    df = df.sort_values(["evt1", "evt2"])
    df["station"] = station
    df["phase"] = phase
    
    df.to_csv(filename, index=0, float_format="%.6f")


def get_args() -> 'argparse.Namespace':
    ap = ArgumentParser()

    io_group = ap.add_argument_group(title="files")
    io_group.add_argument("-i", "--input", required=True, help="input directory containing SAC files")
    io_group.add_argument("-o", "--output", default="./output", help="directory to place output files")
    io_group.add_argument("--pickle", action="store_true", help="if supplied, output the cross correlated matrix as a pickle object")

    cc_group = ap.add_argument_group(title="cross-correlation")
    cc_group.add_argument("--prep_crop", default="[-4, 10]", help="cropping applied prior to preprocessing step")
    cc_group.add_argument("--p_crop", default="[-1.0, 1.0]", help="cropping applied to P phase waveforms prior to cross-correlation")
    cc_group.add_argument("--s_crop", default="[-0.5, 1.0]", help="cropping applied to S phase waveforms prior to cross-correlation")
    cc_group.add_argument("--p_prefilt", default="[2.0, 20.0]", help="prefiltering applied to P phase waveforms prior to cross-correlation")
    cc_group.add_argument("--s_prefilt", default="[2.0, 18.0]", help="prefiltering applied to S phase waveforms prior to cross-correlation")
    cc_group.add_argument("--fft_size", default=256, type=int, help="number of samples used for FFT") # TODO think about this

    args = ap.parse_arguments()
    return args


def process_multiple_dirs(results_dir: str, prep_params: dict):
    subdirs = Path(results_dir).glob("*")

    subdir = next(subdirs)
    data = prep_wfs(subdir, **prep_params)
    while True:
        # load data for the next phase while working on the current one
        next_dir = next(subdirs)
        prep_params["source"] = str(next_dir)
        p = Process(target=prep_data, args=prep_params)
        p.start()

        run_xcorr(data)
        # get next data
        data = p.join()

if __name__=="__main__":
    wf_dir = sys.argv[1]
    try:
        output_dir = sys.argv[2]
    except:
        output_dir = "./outputs"

    wf_dir_basename = os.path.basename(wf_dir)
    sys.stderr.write(f"WORKING ON: {wf_dir_basename}\n")
    print(f"[{pd.Timestamp.now()}]: running xcorr on {wf_dir}")

    # get the phase from the dirname, should be XXX_HHZ for P, else S
    z_component = wf_dir_basename.endswith("Z")
    station = wf_dir_basename.split("_")[0]
    phase = "P" if z_component else "S"

    params = {
        "P": {
            "crop_2": [-.5, .5],
            "prefilt": [2.0, 20.0],
            },
        "S": {
            "crop_2": [-.25, 1.0],
            "prefilt": [2.0, 16.0],
            }
        }

    t0 = pd.Timestamp.now()
    # xcorr = run_xcorr(wf_dir, crop_2=[-.50, 1.0], prefilt=[2.0, 20.0], phase=phase, fft_size=256, n_procs=62)
    xcorr = run_xcorr(wf_dir, phase=phase, n_procs=62, fft_size=256, **params[phase])
    csv_name = f"{output_dir}/{wf_dir_basename}.csv"
    to_csv(xcorr, csv_name, station, phase)

    # debug
    pickle_name = f"{output_dir}/{wf_dir_basename}.p"
    with open(pickle_name, "wb") as f:
        pickle.dump(xcorr, f)
    print(pd.Timestamp.now() - t0)
