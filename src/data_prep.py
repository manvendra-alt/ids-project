import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells",
"num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
"srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label","difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]


def load_raw(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def prepare_data(train_path, test_path, val_size=0.2, seed=42):
    train_df = load_raw(train_path)
    test_df = load_raw(test_path)

    y_full = train_df["is_attack"].values
    y_test = test_df["is_attack"].values
    X_full = train_df.drop(columns=["label", "is_attack"])
    X_test = test_df.drop(columns=["label", "is_attack"])

    # 1. split validation data off FIRST
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_size, stratify=y_full, random_state=seed)

    # 2. fit encoder + scaler on TRAIN ONLY
    num_cols = [c for c in X_train.columns if c not in CATEGORICAL]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL])
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])

    def transform(X):
        num = scaler.transform(X[num_cols])
        cat = encoder.transform(X[CATEGORICAL])
        return np.hstack([num, cat]).astype(np.float32)

    feature_names = num_cols + list(encoder.get_feature_names_out(CATEGORICAL))
    return {
        "X_train": transform(X_train), "y_train": y_train,
        "X_val": transform(X_val), "y_val": y_val,
        "X_test": transform(X_test), "y_test": y_test,
        "feature_names": feature_names,
        "encoder": encoder, "scaler": scaler, "num_cols": num_cols,
    }