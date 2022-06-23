from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf


def download_data_as_dataframe(path, set_no='both', subject=1):
    tmin, tmax = 0., 4.
    if set_no == 'both':
        event_id = dict(baseline=1, hands=2, feet=3)
        event_id_T = dict(T0=1, T1=2, T2=3)
        runs = [1, 6, 10, 14]
        labels = ['baseline', 'hands', 'feet']
        new_labels = [0, 1, 2]
    elif set_no == 'side':
        event_id = dict(baseline=1, left=4, right=5)
        event_id_T = dict(T0=1, T1=4, T2=5)
        runs = [4, 8, 12]
        labels = ['baseline', 'left', 'right']
        new_labels = [0, 3, 4]
    raw_fnames = eegbci.load_data(subject, runs, path=path)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels(lambda x: x.strip('.'))
    # raw.filter(l_freq=None, h_freq=4.)
    events, _ = events_from_annotations(raw, event_id=event_id_T)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    df = epochs.to_data_frame()
    df = df.drop(['time', 'epoch'], axis=1)
    df = df.replace(labels, new_labels)
    df['label'] = df.iloc[:, 0]
    df = df.drop(['condition'], axis=1)
    return df

# path = ...
# df1, epoch = download_data_as_dataframe(path, set_no='both', subject=1)
# df2 = download_data_as_dataframe(path, set_no='side', subject=1)
# df = pd.concat([df1, df2], axis=0)
