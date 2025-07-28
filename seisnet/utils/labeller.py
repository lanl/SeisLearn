import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from glob import glob
from multiprocessing import cpu_count

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from natsort import natsorted
from tqdm import tqdm


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def gaussian_pick(onset, length, sigma):
    r"""
    Extracted from seisbench
    Create probabilistic representation of pick in time series.
    PDF function given by:

    .. math::
        \mathcal{N}(\mu,\,\sigma^{2})

    :param onset: The nearest sample to pick onset
    :type onset: float
    :param length: The length of the trace time series in samples
    :type length: int
    :param sigma: The variance of the Gaussian distribution in samples
    :type sigma: float
    :return prob_pick: 1D time series with probabilistic representation of pick
    :rtype: np.ndarray
    """
    x = np.arange(length)
    return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))


def label_train_hdf5_waveforms(picks:pd.DataFrame, h5_files_path:str,
                               output_npz_path:str, label_sigma:int=20,
                               sample_length:int=3000):
    """
    Label format is P,S,Noise 
    """
    ncpus = int(cpu_count()*0.75)

    if not os.path.exists(output_npz_path):
        os.makedirs(output_npz_path)

    existing_files = glob(f"{output_npz_path}/*.npz")
    ex_event_ids = []
    ex_station_ids = []
    for file in existing_files:
        file = os.path.basename(file)
        file = file.replace(".npz","").split("--")
        ex_event_ids.append(file[0])
        ex_station_ids.append(file[1])
    
    remaining_picks = picks[~((picks.event_id.isin(ex_event_ids)) & \
                              (picks.station_id.isin(ex_station_ids)))]
    remaining_picks = remaining_picks.reset_index(drop=True)
    rem_itrs = len(remaining_picks)

    h5_files_list = natsorted(glob(h5_files_path),reverse=False)

    logger.info("Extracting file paths (Takes a while) .....")
    dataset_paths = []
    with tqdm(position=0, desc="Extraction ....") as pbar:
        for h5file in h5_files_list:
            with h5py.File(h5file, "r") as rf:
                event_ids = list(rf.keys())
                anl_picks = remaining_picks[remaining_picks.event_id.isin(event_ids)].reset_index(drop=True)
                event_ids = anl_picks.event_id.unique()
                
                for evid in event_ids:
                    sta_dict = rf[evid]
                    station_ids = list(sta_dict.keys())
                    station_picks = anl_picks[(anl_picks.event_id==evid) & 
                                              (anl_picks.station_id.isin(station_ids))]
                    station_picks = station_picks[[
                        "event_id","station_id","dt_s","azimuth","back_azimuth",
                        "distance_km","elevation_m","component","depth_km",
                        "instrument","latitude","longitude","local_depth_m",
                        "p_phase_index","s_phase_index",
                        "p_phase_polarity","s_phase_polarity",
                        "p_phase_score", "s_phase_score",
                        "phase_index","phase_picking_channel","phase_polarity",
                        "phase_remark","phase_status","phase_time",
                        "snr","takeoff_angle","unit",]].reset_index()
                    station_picks = station_picks.rename(columns={"phase_index":"orig_phase_index"})

                    for _,dfRow in station_picks.iterrows():
                        dataset_paths.append([h5file,evid,dfRow.station_id,dfRow,output_npz_path])
                        pbar.update(1)
    logger.info("File paths extraction completed!!!")
    
    logger.info("Cropping waveforms ......")

    single_proc = partial(single_process_train_labeler_centered, label_sigma=label_sigma, sample_length=sample_length)

    # Serial process
    # for arg_items in tqdm(dataset_paths, total=len(dataset_paths), position=0, desc="Serial labeling waveforms ..."):
    #     single_proc(arg_items)

    # Multi-thread - multi-processing is susceptible to race conditions and locks
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        results = list(
            tqdm(executor.map(single_proc, dataset_paths),
                              total=len(dataset_paths), position=0,
                              desc="Multi-thread labeling waveforms ...")
        )
    logger.info("Cropping and labeling completed!!!")


def single_process_train_labeler_centered(args_list, label_sigma:int=20,sample_length:int=6000):
    """
    Label a training waveform with a single process that can be parallelized.
    Waveform length is 6000s and the p-arrival is centered. This model does not
    have 
    """
    filepath,evid,stid,pck_row,out_path = args_list

    with h5py.File(filepath, "r") as mf:
        event = mf[evid]
        waveform_values = event[stid]

        try:
            metadata = pck_row.to_dict()
            del metadata["index"]

            ppk = pck_row.p_phase_index
            start_crop = ppk - 3000
            p_label = ppk - start_crop
            end_crop = start_crop+sample_length

            # Crop the waveform to the input sample size
            waveform_values = np.array(waveform_values)
            cropped = waveform_values[:,start_crop:end_crop]
            labels = np.expand_dims(np.zeros_like(cropped[0]), 0)

            # Create the label array
            labels[0] = gaussian_pick(p_label,sample_length,label_sigma)
            metadata["p_phase_index"] = p_label

            # Save the npz file
            file_name = f"{evid}--{stid}.npz"
            file_path = f"{out_path}/{file_name}"
            np.savez(file_path,X=cropped,y=labels,metadata=metadata)
        
        except KeyboardInterrupt:
            logger.info(f"{evid}--{stid} label was interrupted")
        
        except Exception as e:
            logger.info(f"An error occurred - {e}")
        
        finally:
            pass
    
    return

def single_process_train_labeler_random_shift(args_list, label_sigma:int=20,
                                              sample_length:int=3000):
    """
    Label a training waveform with a single process that can be parallelized
    """
    filepath,evid,stid,pck_row,out_path = args_list

    with h5py.File(filepath, "r") as mf:
        event = mf[evid]
        waveform_values = event[stid]

        try:
            metadata = pck_row.to_dict()
            del metadata["index"]

            ppk,spk = pck_row.p_phase_index,pck_row.s_phase_index
            start_crop = 0
            # Find the nearest value to the p-arrival index
            start_crop = find_nearest_index([
                np.random.randint(int(.1*ppk), int(.8*ppk)),
                np.random.randint(int(ppk-(.6*sample_length)), int(ppk-(.4*sample_length))),
                ],
                                            ppk)
            p_label = ppk - start_crop
            s_label = spk - start_crop
            end_crop = start_crop+sample_length

            # Crop the waveform to the input sample size
            waveform_values = np.array(waveform_values)
            cropped = waveform_values[:,start_crop:end_crop]
            labels = np.zeros_like(cropped)

            # Create the label array
            labels[0] = gaussian_pick(p_label,sample_length,label_sigma)
            metadata["p_phase_index"] = p_label
            if s_label < end_crop:
                labels[1] = gaussian_pick(s_label,sample_length,label_sigma)
                metadata["s_phase_index"] = s_label
            else:
                metadata["s_phase_index"] = None
            labels[2] = 1 - labels[0] - labels[1]

            # Save the npz file
            file_name = f"{evid}--{stid}.npz"
            file_path = f"{out_path}/{file_name}"
            np.savez(file_path,X=cropped,y=labels,metadata=metadata)
        
        except KeyboardInterrupt:
            logger.info(f"{evid}--{stid} label was interrupted")
        
        except Exception as e:
            logger.info(f"An error occurred - {e}")
        
        finally:
            pass
    
    return


def label_test_npz_waveforms(data_name:str, input_data_dir:str, 
                             general_data_dir:str, label_sigma:int=20, 
                             sample_length:int=3000):
    """
    Create labels for the test dataset. This function requires the event and pick 
    dataframes, as  well as the training npz waveforms to be inside a single 
    directory with consistent naming structures as provided by Yongsoo
    """
    ncpus = int(cpu_count()*0.75)

    # Load the dataframes
    df_arv = pd.read_csv(f"{input_data_dir}/{data_name}_arrivals.csv")
    df_evt = pd.read_csv(f"{input_data_dir}/{data_name}_events.csv")
    df = df_evt.merge(df_arv,on="evid").drop_duplicates(subset=["evid","station","phase"])
    df["phase"] = df["phase"].apply(lambda x: x.strip())

    # Create the output dir
    out_folder = f"{general_data_dir}/test_{data_name}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    existing_files = glob(f"{out_folder}/*.npz")
    ex_event_ids = []
    ex_station_ids = []
    for file in existing_files:
        file = os.path.basename(file)
        file = file.replace(".npz","").split("--")
        ex_event_ids.append(int(file[0]))
        ex_station_ids.append(file[1])
    
    df_rmng = df[~((df.evid.isin(ex_event_ids)) & (df.station.isin(ex_station_ids)))]
    df_rmng = df_rmng.reset_index(drop=True)

    # Extract the grouped event ids and stations
    target_entries = [(key[0], key[1], group) for key,group in df_rmng.groupby(["evid","station"])]

    # Crop and label the waveforms
    logger.info("Cropping waveforms ......")
    single_proc = partial(single_process_test_labeler, input_data_dir=input_data_dir,
                          test_npz_dir=out_folder, label_sigma=label_sigma, 
                          sample_length=sample_length)
    
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        results = list(
            tqdm(executor.map(single_proc, target_entries),
                              total=len(target_entries), position=0,
                              desc="Multi-thread labeling test waveforms ...")
        )
    
    # with ProcessPoolExecutor(max_workers=ncpus) as executor:
    #     results = list(
    #         tqdm(executor.map(single_proc, target_entries),
    #                           total=len(target_entries), position=0,
    #                           desc="Multi-process labeling test waveforms ...")
    #     )
    logger.info("Cropping and labeling completed!!!")



def single_process_test_labeler(entry:tuple, input_data_dir:str,
                                test_npz_dir:str, label_sigma:int,
                                sample_length:int, samp_frq:int=100):
    """
    Label a test waveform with a single process that can be parallelized
    """
    evid, station, picks = entry
    base_name = os.path.basename(test_npz_dir).replace("test_","")
    metadata = picks[picks.phase=="P"].reset_index(drop=True)
    metadata = metadata[["evid","station","time","lat","lon","dep","mag"]].to_dict(orient="records")

    try:
        inp_file = f"{input_data_dir}/{base_name}/{station}.P.npz"
        if os.path.exists(inp_file):
            npz_file = np.load(inp_file)
            wvfm_idxs = np.where(npz_file["evids"]==evid)[0]

            if len(picks)==1:
                pIdx = 3000
                sIdx = None
            elif len(picks)==2:
                pIdx = 3000
                ps_diff = picks[picks.phase=="S"].arrival.to_numpy()[0] - \
                    picks[picks.phase=="P"].arrival.to_numpy()[0]
                ps_samples = int(ps_diff*samp_frq)
                sIdx = pIdx + ps_samples if ps_samples>0 else None

            if len(wvfm_idxs):
                metadata = metadata[0]
                waveform_index = wvfm_idxs[0]
                waveform = npz_file["x"][waveform_index]

                start_crop = np.random.randint(300, 2500)
                p_label = pIdx - start_crop
                s_label = sIdx - start_crop if sIdx else None
                end_crop = start_crop+sample_length
                
                cropped = waveform[:,start_crop:end_crop]
                labels = np.zeros_like(cropped)

                labels[0] = gaussian_pick(p_label,sample_length,label_sigma)
                metadata["p_phase_index"] = p_label
                if s_label and (s_label < end_crop):
                    labels[1] = gaussian_pick(s_label,sample_length,label_sigma)
                    metadata["s_phase_index"] = s_label
                else:
                    metadata["s_phase_index"] = None
                labels[2] = 1 - labels[0] - labels[1]

                # Save the npz file
                file_name = f"{evid}--{station}.npz"
                file_path = f"{test_npz_dir}/{file_name}"
                np.savez(file_path,X=cropped,y=labels,metadata=metadata)
            
    except KeyboardInterrupt:
        logger.info(f"{evid}--{station} label was interrupted")
        
    except Exception as e:
        logger.info(f"An error occurred - {e}")
    
    finally:
        pass