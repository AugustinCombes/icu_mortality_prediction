import pandas as pd
import vaex as vx
from tqdm import tqdm
import numpy as np
import json
from collections import ChainMap
from os.path import join
import sys

dayconverter = lambda x: np.timedelta64(x, 'ns') / np.timedelta64(1, 'D') #utility function to convert to days

if __name__=="__main__":
    path = sys.argv[1]
    # path = '../data/raw_MIMIC_files'

    ## Read CHARTEVENTS file
    print('[1/7] reading CHARTEVENTS file...')

    df = vx.open(join(path, 'CHARTEVENTS.csv'))
    df = df[['ICUSTAY_ID', 'ITEMID', 'VALUENUM', 'CHARTTIME']]
    df = df.dropna()

    ## Get minimum & maximum dates of CHARTTIME for each unique ICUSTAY_ID
    print('[2/7] getting minimum & maximum dates of CHARTTIME for each unique ICUSTAY_ID...')

    adm_dates = df.groupby('ICUSTAY_ID').agg({'ICUSTAY_ID': 'mean', 'CHARTTIME': 'min'})
    adm_dates = adm_dates[~adm_dates.ICUSTAY_ID.isnan()]
    exit_dates = df.groupby('ICUSTAY_ID').agg({'ICUSTAY_ID': 'mean', 'CHARTTIME': 'max'})
    exit_dates = exit_dates[~exit_dates.ICUSTAY_ID.isnan()]

    ## Compute the icu stay duration
    print('[3/7] computing the icu stay duration...')

    exit_dates.rename('CHARTTIME', 'OUTTIME')
    adm_dates.rename('CHARTTIME', 'INTIME')
    icudates = adm_dates.join(exit_dates, on='ICUSTAY_ID')
    icudates['TIME'] = icudates["OUTTIME"] - icudates["INTIME"]
    # icudates = icudates.drop(columns=['INTIME', 'OUTTIME'])
    icudates = icudates[icudates['TIME'] >= np.timedelta64(1, "D")]
    icudates = icudates[icudates['TIME'] <= np.timedelta64(2, "D")]

    # Filter out the events from stays not in [1, 2] days
    print('[4/7] filtering out the events from stays not in [1, 2] days...')

    df = df.join(icudates, on='ICUSTAY_ID', how='inner')
    df['DT'] = df["CHARTTIME"] - df["INTIME"]

    # Filter out the events outside the first 3 hours
    print('[5/7] filtering out the events outside the first 3 hours...')

    df = df.drop(columns=["CHARTTIME", "INTIME", "OUTTIME", "TIME"])
    df = df[df["DT"] < np.timedelta64(3, "h")]
    df["DT"] = df["DT"].apply(lambda x: np.timedelta64(x, 'ns') / np.timedelta64(1, 'm'))

    pdf = df.to_pandas_df()

    pdf['VAL'] = pd.Series(zip(pdf.ITEMID, pdf.VALUENUM))
    pdf.drop(columns=['ITEMID', 'VALUENUM'], inplace=True)

    pdf = pdf.groupby(["ICUSTAY_ID", "DT"])["VAL"].apply(list)
    pdf = pdf.reset_index('DT')
    pdf = pdf.apply(lambda row: {row.DT: row.VAL} , axis=1)

    pdf = pd.DataFrame(pdf.groupby("ICUSTAY_ID").agg(lambda x:{k:v for d in x for k,v in d.items()}), columns=["stays_dict"])
    pdf = pd.DataFrame(pdf.groupby("ICUSTAY_ID").agg({'stays_dict': lambda x : dict(ChainMap(*x))}), columns=["stays_dict"])

    # Filter out the events > 100
    print('[6/7] filtering out the events > 100...')
    res = dict()
    mem = list(zip(pdf.index, pdf.stays_dict))

    for patient, data in mem :
        s = list(data.keys())
        s.sort()
        data = {int(k):data[k] for k in s}

        row = dict()
        c = 0
        for k in data.keys():
            if c <100:
                if c+len(data[k])<=100:
                    row[k] = list(map(lambda x: (int(x[0]), x[1]), data[k]))
                else :
                    row[k] = list(map(lambda x: (int(x[0]), x[1]), data[k]))[:100-c-len(data[k])]
                c+= len(data[k])

        l = sum(list(map(len, list(row.values()))))
        if l>100:
            print('>100')

        res[int(patient)] = row

    # Adding labels with admissions table
    adm = pd.read_csv(join(path, 'ADMISSIONS.csv'))[['HADM_ID', 'DEATHTIME']]
    icustays = pd.read_csv(join(path, 'ICUSTAYS.csv'))[['HADM_ID', 'ICUSTAY_ID']]

    icustay2hadm = dict(zip(icustays.ICUSTAY_ID, icustays.HADM_ID))
    adm.DEATHTIME = adm.DEATHTIME.isna().astype(int)
    hadm2label = dict(zip(adm.HADM_ID, adm.DEATHTIME))

    icustay2label = dict()
    for icu_k in icustay2hadm.keys():
        if icu_k in icustay2hadm.keys():
            hadm_k = icustay2hadm[icu_k]
            if hadm_k in hadm2label.keys():
                icustay2label[icu_k] = hadm2label[hadm_k]

    restricted_icustay2label = dict()
    for k in icustay2label.keys():
        if k in res.keys():
            restricted_icustay2label[k] = icustay2label[k]

    codes_id = list()
    for v in res.values():
        for vv in v.values():
            for t in vv :
                codes_id.append(t[0])

    codes2encoding = dict()
    for code_id in set(codes_id):
        codes2encoding[code_id] = len(codes2encoding)
    codes = list(map(lambda x: codes2encoding[x], codes_id))

    ## Create a dataframe from the list
    print('[7/7] creating jsons...')
    df = pd.DataFrame(codes)

    ## Use the value_counts() function to get the count of each unique value
    counts = df[0].value_counts()

    ## Filter rare EHR codes
    kept_codes = counts>100
    kept_codes = kept_codes[kept_codes].index.to_list()
    print("Kept", len(kept_codes), "unique EHR codes")

    ## Save json results
    r = json.dumps(res, sort_keys=True)
    with open('events.json', "w") as f:
        f.write(r)
    r = json.dumps(restricted_icustay2label, sort_keys=True)
    with open('targets.json', "w") as f:
        f.write(r)