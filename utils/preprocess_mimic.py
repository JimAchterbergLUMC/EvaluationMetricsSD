import pandas as pd


# df = pd.read_csv("data/admissions.csv.gz", compression="gzip")
# print(df.race.value_counts(normalize=True, dropna=False))
# exit()


# select cohort based on diagnoses
cohort = pd.read_csv("data/diagnoses_icd.csv.gz", compression="gzip")
cohort["icd_code"] = cohort["icd_code"].astype(str)
cohort["icd_code"] = cohort["icd_code"].str[:3]
cohort = cohort[cohort["icd_code"].isin(["I50", "428"])]
cohort = cohort[["hadm_id"]]
cohort = cohort.drop_duplicates().reset_index(drop=True)  # type: ignore

# merge admissions info to cohort (whilst creating length-of-stay feature)
admissions = pd.read_csv(
    "data/admissions.csv.gz",
    compression="gzip",
    usecols=("subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "race", "marital_status", "hospital_expire_flag"),  # type: ignore
)
admissions["race"] = admissions["race"].apply(
    lambda x: "White" if x.lower().startswith("white") else x
)
admissions["race"] = admissions["race"].apply(
    lambda x: "Black" if x.lower().startswith("black") else x
)
admissions["race"] = admissions["race"].apply(
    lambda x: "Other" if x not in ["White", "Black"] else x
)
admissions["marital_status"] = (
    admissions["marital_status"]
    .astype(str)
    .apply(
        lambda x: (
            "Unknown"
            if x.lower() not in ["married", "single", "widowed", "divorced"]
            else x
        )
    )
)

admissions["admittime_numeric"] = pd.to_datetime(admissions["admittime"]).apply(
    lambda x: x.toordinal()
    + (x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400
)
admissions["dischtime_numeric"] = pd.to_datetime(admissions["dischtime"]).apply(
    lambda x: x.toordinal()
    + (x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400
)
admissions["los"] = admissions["dischtime_numeric"] - admissions["admittime_numeric"]
admissions = admissions.rename(columns={"hospital_expire_flag": "mortality"})

cohort = cohort.merge(admissions, on="hadm_id", how="left")


# merge patients info to cohort
patients = pd.read_csv(
    "data/patients.csv.gz",
    compression="gzip",
    usecols=("subject_id", "anchor_age", "gender"),  # type: ignore
)
patients = patients.rename(columns={"anchor_age": "age", "gender": "sex"})

cohort = cohort.merge(patients, on="subject_id", how="left")


# count diagnoses per patient and merge to cohort
diagnoses = pd.read_csv("data/diagnoses_icd.csv.gz", compression="gzip")
diagnoses = diagnoses.groupby("hadm_id").count().iloc[:, 0]
diagnoses.name = "n_diagnoses"
diagnoses = diagnoses.reset_index(drop=False)

cohort = cohort.merge(diagnoses, on="hadm_id", how="left")


# count medications per patient and merge to cohort

# retrieve BMI from omr and merge to cohort
omr = pd.read_csv(
    "data/omr.csv.gz",
    compression="gzip",
    usecols=("subject_id", "chartdate", "result_name", "result_value"),  # type: ignore
)
bmi = omr[omr["result_name"] == "BMI"]
bmi = bmi.rename(columns={"result_value": "BMI"})
bmi = bmi.drop(columns=["result_name"])

bmi["chartdate"] = pd.to_datetime(bmi["chartdate"])
cohort["admittime"] = pd.to_datetime(cohort["admittime"])
bmi = bmi.sort_values("chartdate")
cohort = cohort.sort_values("admittime")

cohort = pd.merge_asof(
    cohort,
    bmi,
    by="subject_id",
    left_on="admittime",
    right_on="chartdate",
    direction="nearest",
)


# retrieve BP from omr, split to systolic and diastolic, and merge to cohort
bp = omr[omr["result_name"].str.startswith("Blood Pressure")]
bp = bp.rename(columns={"result_value": "BP"})
bp_expanded = bp["BP"].str.split("/", expand=True)
bp_expanded.columns = ["BP_systolic", "BP_diastolic"]
bp = pd.concat([bp, bp_expanded], axis=1)
bp = bp.drop(columns=["BP", "result_name"])


bp["chartdate"] = pd.to_datetime(bp["chartdate"])
bp = bp.sort_values("chartdate")
cohort = pd.merge_asof(
    cohort,
    bp,
    by="subject_id",
    left_on="admittime",
    right_on="chartdate",
    direction="nearest",
)

# select only relevant features and cast dtypes
cohort = cohort[
    [
        "age",
        "sex",
        "n_diagnoses",
        "BMI",
        "BP_systolic",
        "BP_diastolic",
        "los",
        "mortality",
        "admission_type",
        "marital_status",
        "race",
    ]
]
cohort = cohort.dropna().reset_index(drop=True)
dtypes = {
    "age": int,
    "sex": str,
    "n_diagnoses": int,
    "BMI": float,
    "BP_systolic": int,
    "BP_diastolic": int,
    "los": float,
    "mortality": int,
    "admission_type": str,
    "marital_status": str,
    "race": str,
}
cohort = cohort.astype(dtypes)

# save dataset to data folder
cohort.to_csv("data/cohort.csv", index=False)
