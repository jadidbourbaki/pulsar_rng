import pathlib
import os
from enum import Enum

class Dataset(Enum):
    NANOGRAV = "NANOGrav"
    EPTA = "EPTA"

    def __str__(self):
        return self.value

# Path for the NANOGrav Nine Year Dataset
NANOGRAV_SUFFIX = "_NANOGrav_9yv1"
NANOGRAV_DATA_PATH = pathlib.Path("external/NANOGrav_9y")
assert NANOGRAV_DATA_PATH.is_dir()
NANOGRAV_TIM_PATH = NANOGRAV_DATA_PATH / "tim"
assert NANOGRAV_TIM_PATH.is_dir()
NANOGRAV_PULSARS = sorted([entry.removesuffix(NANOGRAV_SUFFIX + ".tim") for entry in os.listdir(NANOGRAV_TIM_PATH) if os.path.isfile(os.path.join(NANOGRAV_TIM_PATH, entry))])

# Path for the European Pulsar Timing Array DR2
EPTA_DATA_SET = "DR2full"
EPTA_DATA_PATH = pathlib.Path("external/EPTA-DR2/EPTA-DR2") / EPTA_DATA_SET
assert EPTA_DATA_PATH.is_dir()
EPTA_PULSARS = sorted([entry for entry in os.listdir(EPTA_DATA_PATH) if os.path.isdir(os.path.join(EPTA_DATA_PATH, entry))])

def get_epta_par_file(pulsar_name):
    par_file = EPTA_DATA_PATH / pulsar_name / (pulsar_name + ".par")
    assert par_file.is_file()

    tdb_par_file = EPTA_DATA_PATH / pulsar_name / (pulsar_name + "_TDB.par")
    os.system(f"tcb2tdb {par_file} {tdb_par_file}")
    assert tdb_par_file.is_file()

    print("Converted TCB to TDB:", par_file, " -> ", tdb_par_file)
    return tdb_par_file

def get_nanograv_par_file(pulsar_name):
    par_file = NANOGRAV_DATA_PATH / "par" / f"{pulsar_name}{NANOGRAV_SUFFIX}.gls.par"
    assert par_file.is_file()
    return par_file

def get_par_file(pulsar_name, dataset):
    if dataset == Dataset.NANOGRAV:
        return get_nanograv_par_file(pulsar_name)
    elif dataset == Dataset.EPTA:
        return get_epta_par_file(pulsar_name)
    
    raise ValueError("Invalid dataset selected")

def get_epta_tim_file(pulsar_name):
    tim_file = EPTA_DATA_PATH / pulsar_name / (pulsar_name + "_all.tim")
    assert tim_file.is_file()
    return tim_file

def get_nanograv_tim_file(pulsar_name):
    tim_file = NANOGRAV_DATA_PATH / "tim" / (pulsar_name + NANOGRAV_SUFFIX + ".tim")
    assert tim_file.is_file()
    return tim_file

def get_tim_file(pulsar_name, dataset):
    if dataset == Dataset.NANOGRAV:
        return get_nanograv_tim_file(pulsar_name)
    elif dataset == Dataset.EPTA:
        return get_epta_tim_file(pulsar_name)
    
    raise ValueError("Invalid dataset selected")