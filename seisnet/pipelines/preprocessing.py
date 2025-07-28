from seisnet.utils import (get_data_dir, label_test_npz_waveforms,
                           label_train_hdf5_waveforms,
                           load_picks_without_ridgecrest)
import click

@click.command()
@click.option("-sig","--label_sigma", nargs=1, type=click.INT, default=20, 
              help="Sigma value for gaussian labels around phase pick labels. Defaults to 20.")
@click.option("-sps","--sample_length", nargs=1, type=click.INT, default=6000, 
              help="Total number of samples for training data. Defaults to 6000.")
def label_training_data_workflow_cli(label_sigma,sample_length):
    """
    Create P, S, and Noise labels for waveforms cropped to 
    the phasenet 6000 sample length
    """
    picks = load_picks_without_ridgecrest(f"{get_data_dir()}/metadata/picks.csv")
    h5_files_path = f"{get_data_dir()}/train_h5/*.h5"
    output_npz_path = f"{get_data_dir()}/train_npz"

    label_train_hdf5_waveforms(picks,h5_files_path,output_npz_path,label_sigma,sample_length)

    return True


@click.command()
@click.option("-data","--data_name", nargs=1, type=click.STRING, required=True, 
              help="Name of test dataset e.g. `ridgecrest`")
@click.option("-idr","--input_data_dir", nargs=1, default=f"{get_data_dir()}/AML",
              type=click.Path(exists=True, readable=True), 
              help="Path to test data directory where dataframes are saved")
@click.option("-gdr","--general_data_dir", nargs=1, default=get_data_dir(),
              type=click.Path(exists=True, readable=True), 
              help="Root directory for training data. New folders will be created inside")
@click.option("-sig","--label_sigma", nargs=1, type=click.INT, default=20, 
              help="Sigma value for gaussian labels around phase pick labels. Defaults to 20.")
@click.option("-sps","--sample_length", nargs=1, type=click.INT, default=3000, 
              help="Total number of samples for test data. Defaults to 3000.")
def label_test_data_workflow_cli(data_name,input_data_dir,general_data_dir,label_sigma,sample_length):
    
    assert data_name in ["hawaii", "ridgecrest", "yellowstone"], "Invalid test data name"
    label_test_npz_waveforms(data_name,input_data_dir,general_data_dir,label_sigma,sample_length)
    
    return True