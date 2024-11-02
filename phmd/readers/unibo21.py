from phmd.readers import base as b
import pandas as pd


ABNORMAL_CYCLE_RECORDS = [
    ('006-EE-2.85-0820-S', 621391),  # Discharge capacity dropped abnormally
    ('006-EE-2.85-0820-S', 621392)  # Discharge capacity dropped abnormally
]
ABNORMAL_CAPACITY_RECRODS = [
    ('007-EE-2.85-0820-S', 623002)  # Not the last row in cycle
]


CHARGE_LINE = 37
DISCHARGE_LINE = 40

def read_file(f, file_path, _):
    X = pd.read_csv(f)

    return X
def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        files = [f for f in files if 'test_result_trial_end.csv' in f]
        return dirs, files

    Cap = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    Cap = Cap[0]
    Cap = Cap.drop(Cap[(Cap.line != 37) & (Cap.line != 40)].index)

    Cap = Cap.drop(Cap[(Cap.test_name == '007-EE-2.85-0820-S') & (Cap.record_id == 623002)].index) # Not the last row in cycle
    Cap = Cap.drop(Cap[(Cap.test_name == 'CH00_Cicli di vita')].index)
    Cap = Cap.drop(Cap[(Cap.test_name == 'test')].index)



    def ffilter(dirs, files):
        files = [f for f in files if 'test_result.csv' in f]
        return dirs, files

    Cycle = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    Cycle = Cycle[0]
    Cycle = Cycle.drop(Cycle[(Cycle.test_name == 'CH00_Cicli di vita')].index)
    Cycle = Cycle.drop(Cycle[(Cycle.test_name == 'test')].index)

    # get charge and discharge lines
    Cycle = Cycle.drop(Cycle[(Cycle.line != CHARGE_LINE) & (Cycle.line != DISCHARGE_LINE)].index)

    # Voltage outside 0.1 ~ 5.0 are seen as abnormal dataset
    Cycle = Cycle.drop(Cycle[(Cycle['voltage'] > 5.0) | (Cycle['voltage'] < 0.1)].index)

    Cycle = Cycle.drop(Cap[(Cap.test_name == '006-EE-2.85-0820-S') & (Cap.record_id == 621391)].index ) # Discharge capacity dropped abnormally
    Cycle = Cycle.drop(Cap[(Cap.test_name == '006-EE-2.85-0820-S') & (Cap.record_id ==  621392)].index)  # Discharge capacity dropped abnormally

    # Charge lines
    Charge_Cap = Cap[Cap.line == CHARGE_LINE].copy()
    Charge_Cyc = Cycle[Cycle.line == CHARGE_LINE].copy()
    Charge_Cap["nominal_cap"] = Charge_Cap.test_name.map(lambda x: float(x[7:10]))

    # Discharge lines
    Discharge_Cap = Cap[Cap.line == DISCHARGE_LINE].copy()
    Discharge_Cyc = Cycle[Cycle.line == DISCHARGE_LINE].copy()
    
    #
    #charge_cap_cycles = Charge_Cap[['test_name', 'cycle_count']].drop_duplicates()
    #discharge_cap_cycles = Discharge_Cap[['test_name', 'cycle_count']].drop_duplicates()

    #charge_cap_cyc = Charge_Cyc[['test_name', 'cycle_count']].drop_duplicates()
    #discharge_cap_cyc = Discharge_Cyc[['test_name', 'cycle_count']].drop_duplicates()

    # SOC: (last charge cycle capacity - discharging capacity) / last charge cycle capacity
    cc = Discharge_Cyc.groupby(['test_name']).charging_capacity.transform('last')
    Discharge_Cyc['soc'] = (cc - Discharge_Cyc.discharging_capacity) / cc
    Discharge_Cyc["nominal_cap"] = Discharge_Cyc.test_name.map(lambda x: float(x[7:10]))

    # Test name convention: 000-XW-Y.Y-AABB-T (7~10 chars are cell
    Discharge_Cap["nominal_cap"] = Discharge_Cap.test_name.map(lambda x: float(x[7:10]))

    # SOH: (Last charging cycle capacity / nominal cell capacity
    cc = Discharge_Cap.groupby(['test_name', 'cycle_count']).charging_capacity.transform('last')
    Discharge_Cap['soh'] = cc / Discharge_Cap.nominal_cap

    if task['target'] == 'soc':
        X = Discharge_Cyc[['test_name', 'voltage', 'temperature', 'current', 'soc']]

        return X
    elif task['target'] == 'rul':

        CAPACITY_THRESHOLDS = {
            3.0: 0.95,  # th 90% - min 2.1, 70%
            2.8: 0.995,  # th 94.7% - min 2.622, 92%
            2.7: 0.97,
            2.0: 0.99,  # th 96.5% - min 1.93, 96.5%
            4.0: 0.97,  # th 94.2% - min 3.77 94.2%
            4.9: 0.96,  # th 95.9% - min 4.3, 87.7%
            5.0: 0.9  # th 90% - min 3.63, 72.6%
        }
        CAPACITY_THRESHOLDS = pd.DataFrame({'nominal_cap': k, 'thr': v} for k, v in CAPACITY_THRESHOLDS.items())

        def fault_cycles(Charge_Cap, CAPACITY_THRESHOLDS):
            X = Charge_Cap.copy()
            X = X.merge(CAPACITY_THRESHOLDS, on='nominal_cap')
            X['charging_capacity'] = X.groupby(['test_name', 'cycle_count']).charging_capacity.transform('max')

            # X.groupby(['nominal_cap', 'test_name']).charging_capacity.min()

            # X.groupby(['nominal_cap', 'test_name']).charging_capacity.aggregate(['min', 'max'])

            X['fault'] = X.charging_capacity < (X.nominal_cap * X.thr)

            X['fault_anytime'] = X.groupby(['test_name']).fault.transform('max')
            X = X[X.fault_anytime]
            F = X[X.fault].groupby('test_name').cycle_count.min().reset_index()
            F.columns = ['test_name', 'fault_cycle']
            F = F[F.fault_cycle > 1]

            return F

        F = fault_cycles(Charge_Cap, CAPACITY_THRESHOLDS)
        X = Charge_Cyc.copy()
        X = X.merge(F, on='test_name')
        X['rul'] = X.fault_cycle - X.cycle_count
        
        X = X[['test_name', 'voltage', 'current', 'temperature', 'rul']]

        return X


    return Cap

