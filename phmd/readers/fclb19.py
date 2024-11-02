import os
import scipy as scipy
from phmd.readers import base as b
import pandas as pd
import h5py
import numpy as np

FILES = ['2017-06-30_batchdata_updated_struct_errorcorrect.mat',
         '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         '2018-04-12_batchdata_updated_struct_errorcorrect.mat']

def unified_len(arrays):
    max_len = max([len(v) for v in arrays])

    arrays = [np.pad(v, ((0, max_len - len(v)),), mode="edge") for v in arrays]

    return arrays


def sample(arrays, nvalues):
    if arrays[0].shape[0] < nvalues:
        return arrays

    idx = np.arange(0, arrays[0].shape[0], arrays[0].shape[0] // nvalues)

    arrays = [v[idx] for v in arrays]

    return arrays


def read_file(f, file_path, _):
    file_id = FILES.index(os.path.split(file_path)[1])
    f = h5py.File(f)
    batch = f['batch']

    num_cells = batch['summary'].shape[0]


    # from: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
    frames = []
    for i in range(num_cells):
        cl = f[batch['cycle_life'][i, 0]][()][0,0]
        """
        policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
        summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
            summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                   'cycle': summary_CY}
        """
        cycles = f[batch['cycles'][i, 0]]
        #cycle_dict = {}

        unitt, It, Qct, Qdt, Qdlint, Tt, Tdlint, Vt, dQdVt, Cyclet, rult = (np.array([]), np.array([]), np.array([]),
                                                                           np.array([]), np.array([]), np.array([]), 
                                                                           np.array([]), np.array([]), np.array([]), 
                                                                           np.array([]) ,np.array([]))

        #for j in range(cycles['I'].shape[0]):
        for j in range(100):
            I = np.hstack((f[cycles['I'][j, 0]][()]))
            Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
            Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
            Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
            T = np.hstack((f[cycles['T'][j, 0]][()]))
            Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
            V = np.hstack((f[cycles['V'][j, 0]][()]))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
            #t = np.hstack((f[cycles['t'][j, 0]][()]))


            I, Qc, Qd, Qdlin, T, Tdlin, V, dQdV, Cycle, rul, unit = unified_len([I, Qc, Qd, Qdlin, T, Tdlin, V,
                                                                                 dQdV, np.array([j]),
                                                                                 np.array([cl - j]),
                                                                                 np.array([file_id + i/100])
                                                                                 ])

            I, Qc, Qd, Qdlin, T, Tdlin, V, dQdV, Cycle, rul, unit = sample([I, Qc, Qd, Qdlin, T, Tdlin, V, dQdV,
                                                                            Cycle, rul, unit], 100)

            It = np.hstack((It, I))
            Qct = np.hstack((Qct, Qc))
            Qdt = np.hstack((Qdt, Qd))
            Qdlint = np.hstack((Qdlint, Qdlin))
            Tt = np.hstack((Tt, T))
            Vt = np.hstack((Vt, V))
            dQdVt = np.hstack((dQdVt, dQdV))
            rult = np.hstack((rult, rul))
            unitt = np.hstack((unitt, unit))

        frames.append(pd.DataFrame({
            'I': It,
            'Qc': Qct,
            'Qd': Qdt,
            'Qdlin': Qdlint,
            'T': Tt,
            'V': Vt,
            'dQdV': dQdVt,
            'rul': rult,
            'unit': unitt,
        }))

    X = pd.concat(frames)
    return X


def read_data(file_path, task: dict = None, filters: dict = None):

    def ffilter(dirs, files):
        return dirs, files

    X = b.read_dataset(file_path, None, ffilter=ffilter, fread_file=read_file, fprocess=None,
                          show_pbar=True, data_file_deep=3)

    X = X[0]
    X = X[X.rul > 0]
    return X
