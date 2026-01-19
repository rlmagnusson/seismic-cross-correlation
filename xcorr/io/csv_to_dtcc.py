import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
from argparse import ArgumentParser


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--max-dtcc", type=float, default=0.5)
    ap.add_argument("--min-weight", type=float, default=0.5)
    ap.add_argument("--filter", type=str, default=None)
    ap.add_argument("--evtid-column", type=int, default=-1, help="0-based index of event id in filter file")
    ap.add_argument("-o", "--output", type=str, default="dt.cc")
    ap.add_argument("--min-phases", type=int, default=-1, help="only include pairs that have at least n phases")
    ap.add_argument("csv_dir", type=str)
    args = ap.parse_args()
    return args


# using unsigned int16 to reduce memory consumption by ~20%. Change if using event IDs above 65535
def read_matlab_output_csv(fn):
    dtypes = {
        "Ev1": "uint16",
        "Ev2": "uint16",
        "dtcc": "float32",
        "Weight": "float32",
        "Station": "category",
        "Phase": "category"
    }

    df = pd.read_csv(fn, dtype=dtypes)
    return df

def read_xcorr_csv_output(fn):
    dtypes = {
        "evt1": "uint16",
        "evt2": "uint16",
        "coeff": "float32", # weight = coeff**2
        "dtcc": "float32",
        "station": "category",
        "phase": "category"
    }

    df = pd.read_csv(fn, dtype=dtypes)
    # df["weight"] = df.coeff * df.coeff
    df.coeff *= df.coeff # square to get weight
    df.columns = ["Ev1", "Ev2", "Weight", "dtcc", "Station", "Phase"]
    return df

def tprint(string):
    ts = pd.Timestamp.now()
    ts = ts.strftime("[%H:%M:%S] ")
    out = ts + string
    print(out)


def no_filter(event_id_1: int, event_id_2: int, valid_ids: set) -> bool:
    return False


def id_filter(event_id_1: int, event_id_2: int, valid_ids: set) -> bool:
    ev1 = event_id_1 in valid_ids
    ev2 = event_id_2 in valid_ids
    return not (ev1 & ev2)


def write_pair_lines(pair_lines: list[str], file_handle, min_phases: int = -1) -> bool:
    write = True
    if (min_phases > 1):
        write = len(pair_lines) > min_phases # > instead of >= because header adds to len

    if write:
        for line in pair_lines:
            file_handle.write(line)

    return write


def main_loop(df, filename, diff_S_dtcc=0.002, valid_ids: set | None = None):
    if valid_ids is None:
        filter = no_filter
    else:
        filter = id_filter

    out = open(filename, "w")
    dual_S = False # used to flag double S handling
    ppair = NullPair() # previous pair
    ppair_uid = ""

    pair_phases = []
    for i, pair in tqdm(enumerate(df.itertuples(name="Pair")), mininterval=.5, total=len(df)):
        if filter(pair.Ev1, pair.Ev2, valid_ids):
            continue

        pair_uid = str(pair.Ev1) + "_" + str(pair.Ev2)
        # if pair.uid != ppair.uid: # check if event pair has changed
        # if not ((pair.Ev1==ppair.Ev1) & (pair.Ev2==ppair.Ev2)):
        if pair_uid != ppair_uid:
            write_pair_lines(pair_phases, out, min_phases=args.min_phases)

            pair_phases = []
            write_header(pair, pair_phases)
            
        if pair.Phase == "P": # P is simple, just write
            write_traveltime(pair, pair_phases)
            
        elif pair.Phase == "S":
            if dual_S: # pair and ppair are both S phases of the same station and event pair
                diff_weight = abs(pair.Weight - ppair.Weight) # float version, probably much faster
                diff_dtcc = abs(pair.dtcc - pair.dtcc)
                if (diff_weight < 0.01) & (diff_dtcc <= diff_S_dtcc): # if this is satisfied we use the average
                        dtcc = (pair.dtcc + ppair.dtcc) / 2
                        weight = (pair.Weight + ppair.Weight) / 2
                        used = pair # either one is fine
                        
                else: # if not satisfied use the higher weight phase
                    used = pair if pair.Weight > ppair.Weight else ppair
                    dtcc = used.dtcc
                    weight = used.Weight
                
                write_traveltime(used, pair_phases, dtcc=dtcc, weight=weight)
                dual_S = False
                        
            else:
                try:
                    npair = df.iloc[i+1] # next pair
                    npair_uid = str(npair.Ev1) + "_" + str(npair.Ev2)
                    if (npair.Phase == "S") & (npair.Station == pair.Station) & (npair_uid == pair_uid):
                        dual_S = True
                        ppair = pair
                        ppair_uid = pair_uid
                        continue # handle both S phases at next step
                    else:
                        write_traveltime(pair, pair_phases)
                        
                except IndexError:
                    write_traveltime(pair, pair_phases)
        ppair = pair
        ppair_uid = pair_uid
        if (i+1) % 100000:
            out.flush()
    write_pair_lines(pair_phases, out, min_phases=args.min_phases)

def write_header(pair, pair_phases: list[str]):
    h = f"# {pair.Ev1} {pair.Ev2} 0.0\n"
    # file_handle.write(h)
    pair_phases.append(h)


def write_traveltime(pair, pair_phases: list[str], dtcc=None, weight=None):
    if dtcc is None:
        dtcc = pair.dtcc
    if weight is None:
        weight = pair.Weight
        
    line = f"{pair.Station.ljust(5)} {dtcc:.4f}  {weight:.2f} {pair.Phase}\n"
    # return line
    # file_handle.write(line)
    pair_phases.append(line)

def get_valid_event_ids(filename: str) -> set:
    ids = []
    with open(filename) as f:
        for line in f.readlines():
            ids.append(int(line))
    ids = set(ids)
    return ids
    

class NullPair:
    def __init__(self):
        self.Ev1 = -1
        self.Ev2 = -1
        self.Station = ""
        self.Phase = ""
        self.dtcc = 0
        self.Weight = 0
        self.uid = ""


if __name__=="__main__":
    args = get_args()

    max_dtcc = args.max_dtcc
    # min_cc = args.min_cc
    # min_weight = min_cc**2
    min_weight = args.min_weight

    csv_path = args.csv_dir
    csv_path = Path(csv_path)
    csv_files = list(csv_path.glob("*.csv"))
    csv_files = sorted(csv_files)

    if args.filter:
        valid_ids = get_valid_event_ids(args.filter)
    else:
        valid_ids = None

    n_files = len(csv_files)
    dfs = []
    for i, fn in enumerate(csv_files):
        tprint(f"loading {fn}")
        # df = read_matlab_output_csv(fn)
        df = read_xcorr_csv_output(fn)
        memuse = df.memory_usage().sum() / 1024 / 1024
        # k_lines = len(df) // 1000
        # tprint(f"loaded {k_lines}k lines using {memuse:.1f}MB of memory")

        df = df[df.Weight.ge(min_weight)]
        # keep_lines = len(df)
        # if keep_lines == 0:
            # continue
        # tprint(f"keeping {keep_lines} ({keep_lines/k_lines/10:.1f}%) lines with weight above {min_weight}")

        df = df[df.dtcc.abs().lt(max_dtcc)]
        # keep_lines = len(df)
        # tprint(f"keeping {keep_lines} ({keep_lines/k_lines/10:.1f}%) lines with abs(dtcc) less than  {max_dtcc}")
        # df["uid"] = df.Ev1.apply(str) + "_" + df.Ev2.apply(str)

        dfs.append(df)
        total = pd.concat(dfs)
        total_mem = total.memory_usage().sum() / 1024 / 1024 / 1024
        M_lines = len(total) / 1_000_000
        percentage = (i+1)/n_files * 100
        tprint(f"({percentage:.1f}%) total lines: {M_lines:.1f}M total memory use: {total_mem:.3f}GB")

        dfs = [total]

    del dfs

    tprint("sorting values by Ev1, Ev2...")
    total = total.sort_values(["Ev1", "Ev2"])
    total = total.reset_index(drop=True)
# tprint("generating event pair uids")
# total["uid"] = total.Ev1.apply(str) + "_" + total.Ev2.apply(str)

    tprint("starting iteration through data")
    main_loop(total, args.output, valid_ids=valid_ids)

# print("writing to pickle...")
# import pickle
# with open('total.p', 'wb') as f:
#     pickle.dump(total ,f)
# print("Done")
