import pandas as pd


def eta_in_anni(data_accettazione, data_nascita):
    acc = pd.to_datetime(data_accettazione)
    nsc = pd.to_datetime(data_nascita)
    return int((acc - nsc).days // 365)
