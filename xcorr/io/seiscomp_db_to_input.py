import pymysql
import pandas as pd
from argparse import ArgumentParser, Namespace

def get_args() -> Namespace:
    ap = ArgumentParser()
    ap.add_argument("-d", "--db", required=True, help="seiscomp style database info")
    ap.add_argument("--origins", action="store_true", help="pass this flag if the input IDs are origin IDs")
    ap.add_argument("-o", "--output", default="seiscomp_xcorr_inp.txt")
    ap.add_argument("events")
    args = ap.parse_args()
    return args


def db_string_to_dict(database_string: str) -> dict[str, str]:
    # mysql://user:password@host/database
    stub = database_string.split("://")[1]
    user = stub.split(":")[0]
    password = stub.split(":")[1].split("@")[0]
    host = stub.split("@")[1].split("/")[0]
    database = stub.split("/")[-1]

    db_info = {
        "user": user,
        "password": password,
        "host": host,
        "database": database
    }
    return db_info


def get_preferred_origin(event_id: str, db: dict[str, str]) -> str | None:
    event_query = "SELECT preferredOriginID FROM Event E JOIN PublicObject POE ON POE._oid=E._oid WHERE POE.publicID='{event_id}'"
    
    conn = pymysql.connect(**db)
    with conn.cursor() as cur:
        q = event_query.format(event_id=event_id)
        cur.execute(q)
        r = cur.fetchall()
    conn.close()
    try:
        origin_id = r[0][0]
    except IndexError:
        return None
    return origin_id


def get_event_phases( db: dict[str, str], origin_id: str | None = None, event_id: str | None = None):
    if (origin_id is None) == (event_id is None):
        raise ValueError("only one of origin_id or event_id may be supplied")
        
    arrival_query = """
    SELECT 
    O.time_value AS otime,
    O.time_value_ms AS otime_ms,
    O.latitude_value AS origin_lat,
    O.longitude_value AS origin_lon,
    O.depth_value AS origin_depth,
    P.time_value AS pick_time,
    P.time_value_ms AS pick_time_ms,
    P.waveformID_networkCode AS net,
    P.waveformID_stationCode AS sta,
    P.waveformID_locationCode AS loc,
    P.waveformID_channelCode AS cha,
    P.phaseHint_code AS phase,
    A.weight AS weight
    FROM 
    PublicObject PO 
    JOIN Origin O ON O._oid=PO._oid
    JOIN Arrival A ON A._parent_oid=O._oid
    JOIN PublicObject POP ON POP.publicID=A.pickID
    JOIN Pick P ON P._oid=POP._oid
    WHERE PO.publicID='{origin_id}'
    AND weight > 0
    """

    if origin_id is None:
        origin_id = get_preferred_origin(event_id, db)

    q = arrival_query.format(origin_id=origin_id)
    conn = pymysql.connect(**db)
    with pymysql.cursors.DictCursor(conn) as cur:
        cur.execute(q)
        r = cur.fetchall()
    arrivals = pd.DataFrame(r)

    # output_id = event_id if origin_id is None else origin_id
    output_id = origin_id if event_id is None else event_id
    # output_id = output_id.replace("/", "_")
    arrivals["event_id"] = output_id
    return arrivals


def get_event_phases_event_id(event_id: str, db: dict[str, str]) -> pd.DataFrame:
    return get_event_phases(event_id=event_id, db=db)


def get_event_phases_origin_id(origin_id: str, db: dict[str, str]) -> pd.DataFrame:
    return get_event_phases(origin_id=origin_id, db=db)


def df_to_lines(arrivals: pd.DataFrame, event_number: int) -> str:
    event_lines = ""
    for arr in arrivals.itertuples():
        otime = arr.otime + pd.Timedelta(arr.otime_ms, unit="us")
        ptime = arr.pick_time + pd.Timedelta(arr.pick_time_ms, unit="us")
        # line = f"{arr.event_id}, {otime.isoformat()},{ptime.isoformat()},{arr.net},{arr.sta},\"{arr.loc}\",{arr.cha},{arr.phase},{event_number}\n"
        line = f"{arr.event_id},{otime.isoformat()},{arr.origin_lat},{arr.origin_lon},{arr.origin_depth}," \
                f"{ptime.isoformat()},{arr.net},{arr.sta},\"{arr.loc}\",{arr.cha},{arr.phase},{event_number}\n"
        event_lines += line
    return event_lines


def iterate_events(args):
    db = db_string_to_dict(args.db)
    with open(args.events) as f:
        event_ids = [ev.strip() for ev in f.readlines()]

    if args.origins:
        get_function = lambda x: get_event_phases(origin_id=x, db=db)
    else:
        get_function = lambda x: get_event_phases(event_id=x, db=db)

    with open(args.output, "w") as f:
        for i, evt in enumerate(event_ids):
            print(evt)
            evt_data = get_function(evt)
            lines = df_to_lines(evt_data, event_number=i+1)
            f.write(lines)


if __name__=="__main__":
    args = get_args()
    print(args)
    iterate_events(args)
