import pandas as pd
import obspy as obs


def read_phase_info(filename: str) -> pd.DataFrame:
    phase_info = pd.read_csv(
        filename, 
        names=["EventID", "event_time", "event_latitude", "event_longitude", "event_depth", "pick_time", "network", "station", "location", "channel", "phase", "event_number"],
        dtype={"location": str}
    )
    phase_info.event_time = phase_info.event_time.apply(obs.UTCDateTime)
    phase_info.pick_time = phase_info.pick_time.apply(obs.UTCDateTime)
    phase_info["offset"] = phase_info.pick_time - phase_info.event_time
    phase_info.location = phase_info.location.fillna("")
    return phase_info


def output_growclust_evtlist(phase_info: pd.DataFrame, filename: str) -> None:
    with open(filename, "w") as f:
        for evt in phase_info.drop_duplicates("event_number").itertuples():
            line = f"{evt.event_time.strftime("%Y %m %d %H %M %S.%f")}  {evt.event_latitude:.6f} {evt.event_longitude:.6f} {evt.event_depth:.6f} 0.000 0.000 0.000 0.000 {evt.event_number}\n"
            f.write(line)


def flip_horizontal(channel_code: str) -> str:
    base = channel_code[:2]
    orientation = channel_code[-1]
    match orientation:
        case "N":
            return base + "E"
        case "E":
            return base + "N"
        case "1":
            return base + "2"
        case "2":
            return base + "1"
        case _:
            raise ValueError(f"could not flip channel code that ends with {channel_code[-1]}")


def duplicate_s_phases(phase_info: pd.DataFrame) -> pd.DataFrame:
    s_phases = phase_info[phase_info.phase.eq("S")].copy()
    s_phases.channel = s_phases.channel.apply(flip_horizontal)
    total = pd.concat([phase_info, s_phases])
    total = total.sort_values(["event_number", "station", "phase"])
    total = total.reset_index(drop=True)

    return total


def enhance_phase_info(phase_info: pd.DataFrame, pre_pad: float, post_pad: float) -> pd.DataFrame:
    phase_info["pre_pad"] = pre_pad
    phase_info["post_pad"] = post_pad
    phase_info["starttime"] = phase_info.event_time - pre_pad
    phase_info["endtime"] = phase_info.event_time + post_pad
    phase_info["pick_offset"] = phase_info.offset + pre_pad
    phase_info = duplicate_s_phases(phase_info)
    return phase_info


