import argparse
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from structure import Patient
from preprocessor import preprocess
from features import compute_features
from labels import add_labels
from csv_writer import save_config, save_patient

parser = argparse.ArgumentParser(
    description="Extracts the epoch-wise features of each TUAR Patient's recordings and saves them as a CSV file."
)
parser.add_argument(
    "-i", "--input", required=True, help="Path to the molded TUAR dataset folder."
)
args = parser.parse_args()

# Verify folder exists
if not os.path.isdir(args.input):
    raise FileNotFoundError(f"Folder does not exist: {args.input}")

# Create output parent directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_dir = os.path.join("output", f"output_{timestamp}")

assert not os.path.exists(output_dir), f"Output directory already exists: {output_dir}"
os.makedirs(output_dir)
output_dir = Path(os.path.abspath(output_dir))

################################################################################

# Save the "config"
save_config(output_dir)

# Get list of subdirectories (Patients) excluding hidden ones folders
subdirs = [
    d
    for d in os.listdir(args.input)
    if os.path.isdir(os.path.join(args.input, d)) and not d.startswith(".")
]

# Add each patient
patients: list[Patient] = []
pbar = tqdm(subdirs)
for dir_name in pbar:
    pbar.set_description(f"Reading {dir_name}")

    dir_path = os.path.join(args.input, dir_name)
    patients.append(Patient(dir_path))


# For debugging
# patients = patients[:2]  # smaller quantity => faster output
# patients = [
#     patient for patient in patients if patient.name == "philipp"
# ]  # specific Patient


# Extract each patient's features, save them, free memory
pbar = tqdm(patients)
for patient in pbar:

    # Process and extract
    pbar.set_description(f"Extracting {patient.name}'s features")
    for recording in patient.recordings:
        preprocess(recording)  # Preprocesses, resamples, and cuts into epochs
        compute_features(recording)  # Computes the features
        add_labels(recording)  # Adds the artifact labels

    # Save
    pbar.set_description(f"Saving {patient.name}'s extracted features")
    save_patient(patient, output_dir)

    # Free from memory
    del patient.recordings


print(f"Output has been saved to {output_dir.absolute()}")
