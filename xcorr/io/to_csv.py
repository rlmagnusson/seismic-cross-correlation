from ..processing import CorrelationData
from pandas import DataFrame


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
            # ev1 = eid[i]
            # ev2 = eid[j]
            ev1 = eid[j] # changed so that ev1 < ev2
            ev2 = eid[i] # TODO add 12 or 21 settings
            line = (ev1, ev2, correlation_coeff, lag_time)
            xcorr_lines.append(line)

    df = DataFrame(xcorr_lines, columns=["evt1", "evt2", "coeff", "dtcc"])
    df = df.sort_values(["evt1", "evt2"])
    df["station"] = station
    df["phase"] = phase

    df.to_csv(filename, index=0, float_format="%.6f")

