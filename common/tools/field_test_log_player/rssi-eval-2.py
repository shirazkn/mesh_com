import sqlite3 as sql

# Get Pandas dataframes from SQLite ~

FILE_NAMES = ["data/march23/sad03_mission1.db3", "data/march23/sad03_mission2.db3", "data/march23/sadgs01_mission2.db3"]

def read_sqlite(file):
    from pandas import read_sql_query, read_sql_table

    with sql.connect(file) as con:
        tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", con)['name'])
        df = {table : read_sql_query(f"SELECT * from {table}", con) for table in tables}

    return df

dfs = [read_sqlite(FILE_NAME) for FILE_NAME in FILE_NAMES]


# Try to 'deserialize' the data ~

from rosbags.highlevel.anyreader import deserialize_cdr

for df in dfs:
    message_names = {}
    for t in df["topics"].itertuples():
        message_names[t.id] = t.type

    df["messages"]["data_parsed"] = df["messages"].apply(lambda t: deserialize_cdr(t["data"], message_names[t["id"]]) if message_names[t["id"]] == 'geometry_msgs/msg/PoseStamped' else None, axis=1)
