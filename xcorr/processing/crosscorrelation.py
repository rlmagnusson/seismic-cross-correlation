import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
import obspy as obs
from pathlib import Path
from multiprocessing import Pool
from typing import NamedTuple, Any
import matplotlib.pyplot as plt


def clone_row(x: NDArray[Any], idx: int, shape: tuple[int] | None = None) -> NDArray[np.float64]:
    """
    Return an array of the same shape as `x` where every row is x[idx, :]
    """
    if shape is None:
        shape = x.shape

    a = np.zeros(shape, dtype=x.dtype)
    for i in range(len(x)):
        a[i, :] = x[idx]
    return a


@dataclass
class WaveformData:
    """
    travel_times: can be offset by some constant, only the differences are relevant
    """
    station: str
    channel: str
    waveforms: NDArray[np.float64] = field(repr=False)
    pick_offsets: list[float] = field(repr=False) # used for plotting
    travel_times: list[float] = field(repr=False) # contains information on traveltime differences
    event_offsets: list[float] = field(repr=False)
    event_ids: list[int] = field(repr=False)
    count: int
    phase: str
    npts: int
    sampling_rate: int
    original_sampling_rate: list[int] = field(repr=False)
    start_times: list[obs.UTCDateTime] = field(repr=False)

    def __str__(self):
        s = f"WaveformData [{self.station}.{self.channel}] count: {self.count}, npts: {self.npts}"
        return s

    def __post_init__(self):
        self._event_id_to_index = {}
        for i, event_id in enumerate(self.event_ids):
            self._event_id_to_index[int(event_id)] = i

    def plot_element(self, 
                     index: int | None = None, 
                     event_id: int | None = None,
                     window_size: float = -1.0,
                     figsize: tuple[float, float] = (12.0, 5.0),
                     plot_pick: bool = True,
                     zero_at_origin: bool = False,
                     ):
        # idx = event_id if index is None else index
        # if idx is None:
        #     raise ValueError("`index` and `event_id` can not both be None")
        if index is None:
            idx = self._event_id_to_index[event_id]
        else:
            idx = index

        wf = self.waveforms[idx]
        offset = self.pick_offsets[idx]

        time = np.arange(0, self.npts) / self.sampling_rate
        if zero_at_origin:
            shift = self.event_offsets[idx]
            time -= shift
            offset -= shift

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time, wf, 'k')
        ax.vlines(offset, wf.min(), wf.max(), "r")
        if window_size > 0:
            x1 = offset - window_size / 2.0
            x2 = offset + window_size / 2.0
            ax.set_xlim((x1, x2))
        plt.grid()

    def plot_pair(self, event_id_1: int, event_id_2: int, scale: bool = True, figsize: tuple[float, float] = (12.0, 4.0)):
        fig, ax = plt.subplots(figsize=figsize)
        idx_1 = self._event_id_to_index[event_id_1]
        idx_2 = self._event_id_to_index[event_id_2]

        trace_1 = self.waveforms[idx_1]
        trace_2 = self.waveforms[idx_2]

        time = np.arange(0, self.npts) / self.sampling_rate

        if scale:
            trace_1 /= trace_1.max()
            trace_2 /= trace_2.max()

        ax.plot(time, trace_1, "k", label=f"{self.station}.{self.channel} event {event_id_1}")
        ax.plot(time, trace_2, "b", label=f"{self.station}.{self.channel} event {event_id_2}")

        plt.grid()
        plt.legend()


@dataclass(repr=False)
class FrequencyDomain:
    samples: NDArray[np.complex128]
    travel_times: list[float]
    event_ids: list[int]
    coefficients: list[float]
    lag_values: NDArray[np.float64]
    count: int

    def __repr__(self):
        return f"{__class__.__name__}(count={self.count}, shape={self.samples.shape}, type={self.samples.dtype})"


@dataclass(repr=False)
class CorrelationData:
    coefficient: NDArray[np.float64]
    lag_time: NDArray[np.float64]
    event_ids: list[int]
    cc_lag_time: NDArray[np.float64] # result from cc
    tt_lag_time: NDArray[np.float64]

    def __repr__(self) -> str:
        return f"{__class__.__name__}(count={len(self.event_ids)})"


class PartialCorrelationData(NamedTuple):
    """
    Structure for temporary data that is computed from cross-correlating one row with all others
    """
    row_index: int
    coefficient: NDArray[np.float64]
    lag_time: NDArray[np.float64]
    cc_lag_time: NDArray[np.float64]
    tt_lag_time: NDArray[np.float64]


def sort_by_event_id(evt_ids: list[int], delay: list[float], waveforms: list[NDArray[np.float64]]) -> tuple[list[int], list[float], list[NDArray[np.float64]]]:
    idx = np.arange(len(evt_ids))
    sort_idx = sorted(idx, key=lambda x: evt_ids[x])
    sorted_delay = [delay[si] for si in sort_idx]
    sorted_waveforms = [waveforms[si] for si in sort_idx]
    sorted_eids = [evt_ids[si] for si in sort_idx]
    return sorted_eids, sorted_delay, sorted_waveforms


def next_pow2(x: int) -> int:
    """
    Returns 2^k where k is the smallest k so that 2^k >= x
    """
    k = 1
    while 2**k < x:
        k += 1
    return int(2**k)


def preprocess(waveform_samples: NDArray[np.float32], start_time: obs.UTCDateTime, pick_offset: float, crop_1: list[float], crop_2: list[float], prefilt: list[float], taper: float, sampling_rate: int) -> NDArray[np.float32]:
    # recover trace
    ref_time = obs.UTCDateTime(start_time) + pick_offset
    wf = obs.Trace(waveform_samples, header={
        "sampling_rate": sampling_rate, 
        "starttime": start_time
    })
    # crop 1 and filter
    t1, t2 = ref_time - abs(crop_1[0]), ref_time + crop_1[1]
    wf = wf.slice(t1, t2)

    wf.detrend("demean")
    wf.taper(taper, type="cosine")
    wf = wf.filter("bandpass", freqmin=prefilt[0], freqmax=prefilt[1], zerophase=True)

    # crop 2 and final taper
    t3, t4 = ref_time - abs(crop_2[0]), ref_time + crop_2[1]
    wf = wf.slice(t3, t4)
    wf.taper(taper, type="cosine")

    return wf.data


def batch_preprocess(waveform_data: WaveformData, crop_1: list[float], crop_2: list[float], prefilt: list[float], taper: float) -> WaveformData:
    processed = []
    event_offsets = []
    pick_offsets = [] # pick offsets are changed but travel times (differences) remain intact
    start_times = []
    for i in range(waveform_data.count):
        p = preprocess(
            waveform_data.waveforms[i],
            start_time=waveform_data.start_times[i],
            pick_offset=waveform_data.pick_offsets[i],
            crop_1=crop_1,
            crop_2=crop_2,
            prefilt=prefilt,
            taper=taper,
            sampling_rate=waveform_data.sampling_rate
        )
        processed.append(p)
        pick_offsets.append(crop_2[0])

        event_offset = (waveform_data.pick_offsets[i] - waveform_data.event_offsets[i] - crop_2[0]) * -1
        event_offsets.append(event_offset)
        start_times.append(obs.UTCDateTime(waveform_data.start_times[i]) + waveform_data.pick_offsets[i] - crop_2[0]) # TODO check

    processed_array = np.array(processed)

    processed_data = WaveformData(
        station=waveform_data.station,
        channel=waveform_data.channel,
        waveforms=processed_array,
        pick_offsets=pick_offsets,
        travel_times=waveform_data.travel_times, # pick offsets are changed but travel times (differences) remain intact
        event_ids=waveform_data.event_ids,
        count=waveform_data.count,
        phase=waveform_data.phase,
        npts=processed_array.shape[1],
        sampling_rate=waveform_data.sampling_rate,
        original_sampling_rate=waveform_data.original_sampling_rate,
        start_times=start_times,
        event_offsets=event_offsets
    )
    return processed_data


def fft_waveforms(waveform_data: WaveformData, fft_size: int | None = None) -> FrequencyDomain:
    wfs = waveform_data.waveforms
    if fft_size is None:
        fft_size = next_pow2(waveform_data.npts)

    mid_idx = fft_size // 2
    waveform_fft = np.fft.fft(wfs, n=fft_size)

    scaling = 1.0 / np.sqrt(np.sum(wfs * wfs, axis=1))

    lag_values = np.arange(fft_size) / waveform_data.sampling_rate
    max_offset = mid_idx / waveform_data.sampling_rate
    lag_values -= max_offset
    freq_domain = FrequencyDomain(
        samples=waveform_fft,
        # pick_offsets=waveform_data.pick_offsets,
        travel_times=waveform_data.travel_times,
        event_ids=waveform_data.event_ids,
        coefficients=scaling,
        lag_values=lag_values,
        count=waveform_data.count
    )
    return freq_domain


def single_row(i: int, fd_data: FrequencyDomain) -> PartialCorrelationData:
    """
    Compute the cross correlation of one waveform with every other waveform by matrix multiplication
    """
    original = fd_data.samples
    conjugated = original.conj()
    N, M = original.shape
    m = M // 2

    other_rows = conjugated[i:N, :] # i:N prevents multiple computations of same pair
    row = clone_row(original, i)
    row = row[i:N, :]

    corr_freq_domain = row * other_rows
    corr = np.fft.ifft(corr_freq_domain)
    corr_m = np.zeros(corr.shape)

    # reordering so that dt = 0 is in the middle
    corr_m[:, :m] = np.real(corr[:, m:]) # imaginary part should be negligible after inverse transform
    corr_m[:, m:] = np.real(corr[:, :m])
    corr = corr_m

    calculated_lag, corr_coeff = subsample_interpolate(corr, fd_data.lag_values)
    # travel_time_lag = fd_data.travel_times[i:N] + fd_data.travel_times[i]
    corr_coeff = fd_data.coefficients[i] * fd_data.coefficients[i:] * corr_coeff
    final_lag = calculated_lag - fd_data.travel_times[i:N] + fd_data.travel_times[i]
    # final_lag = calculated_lag - travel_time_lag
    travel_time_lag = final_lag - calculated_lag

    row_results = PartialCorrelationData(row_index=i, coefficient=corr_coeff, lag_time=final_lag, cc_lag_time=calculated_lag, tt_lag_time=travel_time_lag)

    return row_results


def frequency_domain_xcorr(fd_data: FrequencyDomain, n_procs: int = 4) -> CorrelationData:
    N = fd_data.count
    args = [[i, fd_data] for i in range(N)]

    with Pool(n_procs) as pool:
        partial_results: list[PartialCorrelationData] = pool.starmap(single_row, args)

    lag_matrix = np.zeros([N, N])
    coeff_matrix = np.zeros([N, N])
    cc_lag = np.zeros([N, N])
    tt_lag = np.zeros([N, N])

    for result in partial_results:
        i = result.row_index
        coeff_matrix[i:N, i] = result.coefficient
        lag_matrix[i:N, i] = result.lag_time
        cc_lag[i:N, i] = result.cc_lag_time
        tt_lag[i:N, i] = result.tt_lag_time

    final_results = CorrelationData(
        coefficient=coeff_matrix, 
        lag_time=lag_matrix, 
        event_ids=fd_data.event_ids,
        cc_lag_time=cc_lag,
        tt_lag_time=tt_lag
    )

    return final_results


def subsample_interpolate(corr: NDArray[np.float32], lag: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Fit a second order polynomial to the correlation peak and interpolate to find the 
    maximum subsample correlation coefficient
    """
    max_c_idx = np.argmax(corr, axis=1)
    subsample_lag = []
    subsample_coeff = []
    for idx, c_ in zip(max_c_idx, corr):
        # attempt second order polynomial fit around the highest coefficient
        fit_lag = lag[idx-1: idx+2]
        fit_corr = c_[idx-1: idx+2]

        if not ((len(fit_lag) == 3) & (len(fit_corr) == 3)):
            # sys.stderr.write(f"interpolation error at index {idx}\n")
            max_corr_lag = lag[idx]
            max_corr_coeff = c_[idx]

        else:
            a, b, c = np.polyfit(fit_lag, fit_corr, 2)
            max_corr_lag = -.5 * b / a
            max_corr_coeff = a*max_corr_lag**2 + b*max_corr_lag + c

        subsample_lag.append(max_corr_lag)
        subsample_coeff.append(max_corr_coeff)
    return np.array(subsample_lag), np.array(subsample_coeff)

