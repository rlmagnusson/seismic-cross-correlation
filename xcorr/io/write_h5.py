import pandas as pd
from tqdm import tqdm
import h5py
import os
from ..processing import WaveformData
import numpy as np
from obspy import UTCDateTime, Stream
from typing import Protocol


class WaveformSource(Protocol):
    def get_waveforms(self, network: str, station: str, location: str, channel: str, starttime: UTCDateTime, endtime: UTCDateTime) -> Stream: ...


def process_channel(
        station: str,
        channel: str,
        sampling_rate: int,
        phase_info: pd.DataFrame,
        temp_name: str,
        datasource: WaveformSource,
        progressbar: bool = False,
        ) -> int:
    print(f"processing {station}.{channel} at {sampling_rate} sps")
    written = 0
    sub = phase_info[phase_info.station.eq(station) & phase_info.channel.eq(channel)]
    if sub.empty:
        raise Exception(f"no phases associated with {station}.{channel} found in phase info")

    with h5py.File(temp_name, "w") as f:
        grp = f.create_group(f"{station}/{channel}")
        if progressbar:
            data_iterator = tqdm(sub.itertuples(), total=len(sub))
        else:
            data_iterator = sub.itertuples()
        for phase in data_iterator:
            # event_name = f"event_{phase.event_number:05d}"
            wf = datasource.get_waveforms(
                            network=phase.network,
                            station=phase.station,
                            location=phase.location,
                            channel=phase.channel,
                            starttime=phase.starttime,
                            endtime=phase.endtime
                        )
            if not wf:
                continue
            if len(wf) > 1:
                continue

            trace = wf[0]
            expected_samples = int((phase.endtime - phase.starttime) * trace.stats.sampling_rate)

            if trace.stats.npts < expected_samples:
                print(f"skipping {station}.{channel} trace with {trace.stats.npts} samples")
                continue
            original_sampling_rate = int(trace.stats.sampling_rate)

            if original_sampling_rate < sampling_rate:
                trace.detrend("linear")
                trace.interpolate(sampling_rate=sampling_rate, method="lanczos", a=20)
            elif original_sampling_rate > sampling_rate:
                trace.detrend("linear")
                trace.detrend("demean")
                trace.taper(type="cosine", max_percentage=0.05)
                trace.filter("lowpass", freq=float(sampling_rate) / 2.000001, corners=2, zerophase=True)
                trace.interpolate(sampling_rate=sampling_rate, method="lanczos", a=20)

            samples = trace.data
            if len(samples) > expected_samples:
                samples = samples[:expected_samples]

            dset_name = f"event_{phase.event_number:05d}_{phase.phase}"
            try: # this catches the strange case where an event has more than one P or S pick from same channel
                dset = grp.create_dataset(dset_name, data=samples)
            except ValueError:
                print(f"error creating dataset {dset_name} for {station}/{channel}")
                continue
            dset.attrs["phase"] = phase.phase
            dset.attrs["event_number"] = phase.event_number
            dset.attrs["event_time"] = phase.event_time.isoformat()
            dset.attrs["start_time"] = trace.stats.starttime.isoformat()
            dset.attrs["original_sampling_rate"] = original_sampling_rate
            dset.attrs["sampling_rate"] = sampling_rate
            dset.attrs["pick_offset"] = phase.pick_offset
            dset.attrs["event_offset"] = phase.pre_pad

            written += 1

    return written


def generate_jobs(sampling_rate: int, phase_info: pd.DataFrame, datasource: WaveformSource, temp_dir: str, progressbar: bool = False) -> list[dict[str, str]]:
    os.makedirs(temp_dir, exist_ok=True)
    jobs = phase_info[["station", "channel"]].copy()
    jobs["code"] = jobs.station + "." + jobs.channel

    job_dicts = []
    for code in jobs.code.unique():
        split = code.split(".")
        station = split[0]
        channel = split[1]
        d = {
            "station": station,
            "channel": channel,
            "sampling_rate": sampling_rate,
            "temp_name": f"{temp_dir}/temp_{station}_{channel}.h5",
            "phase_info": phase_info,
            "progressbar": progressbar,
            "datasource": datasource,
        }
        job_dicts.append(d)
    return job_dicts


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
                start_times.append(UTCDateTime(meta["start_time"]))
                original_sampling_rate.append(meta["original_sampling_rate"])

    waveforms = np.array(waveform_arrays)

    wd = WaveformData(
        station=station,
        channel=channel,
        waveforms=np.array(waveform_arrays),
        pick_offsets=pick_offsets,
        travel_times=pick_offsets,
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

