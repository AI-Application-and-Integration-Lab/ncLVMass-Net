import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np

NIFTI_EXTS = (".nii.gz", ".nii")
SITK_EXTS = (".nrrd", ".nhdr", ".mha", ".mhd")
ALL_EXTS = NIFTI_EXTS + SITK_EXTS

NIFTI_UNIT_TO_MM = {"unknown": 1.0, "meter": 1000.0, "mm": 1.0, "micron": 0.001}

# --------------------------------------------------------------------------------
# Format-specific readers: each returns (3D numpy array, spacing in mm as a tuple)
# --------------------------------------------------------------------------------
def read_nifti(entry: os.DirEntry):
    import nibabel as nib

    img = nib.load(entry.path)
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()[: data.ndim]
    unit = img.header.get_xyzt_units()[0]
    factor = NIFTI_UNIT_TO_MM.get(unit, 1.0)
    if unit not in NIFTI_UNIT_TO_MM:
        print(f"[Warning] {entry.name}: unknow unit '{unit}', assuming mm", file=sys.stderr)
    spacing = tuple(float(z) * factor for z in zooms)
    return data, spacing

def read_sitk(entry: os.DirEntry):
    import SimpleITK as sitk

    img = sitk.ReadImage(entry.path)
    data = sitk.GetArrayFromImage(img)  # array order is (z, y, x)
    spacing = tuple(float(s) for s in img.GetSpacing())  # spacing order is (x, y, z), in mm
    return data, spacing

def load_segmentation(entry: os.DirEntry):
    name = entry.name.lower()
    if name.endswith(NIFTI_EXTS):
        data, spacing = read_nifti(entry)
    elif name.endswith(SITK_EXTS):
        data, spacing = read_sitk(entry)
    else:
        raise ValueError(f"Unsupported file format: {entry.name}")

    # Handle non-single-volume data (e.g. 4D, or an extra channel dimension)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"{entry.name}: data is {data.ndim}D, only 3D is supported (shape={data.shape})")
    if len(spacing) != 3:
        raise ValueError(f"{entry.name}: got {len(spacing)} spacing values, expected 3 ({spacing})")
    return data, spacing

# ------------------
# Volume computation
# ------------------
def compute_volumes(data: np.ndarray, spacing_mm, target_label:int):
    """Return {label: (voxel_count, volume_cm3)}"""
    if not np.issubdtype(data.dtype, np.integer):
        rounded = np.rint(data)
        if not np.allclose(data, rounded):
            raise ValueError(
                "Data contains non-integer values; this does not look like a label map "
                "(is it a softmax / probability output?)"
            )
        data = rounded.astype(np.int64)

    voxel_mm3 = float(np.prod(spacing_mm))
    result = np.sum(data==target_label) * voxel_mm3 / 1000.0  # mm^3 -> cm^3
    return result

def get_label(dataset_json, label_name):
    """Read {int: name} from an nnU-Net dataset.json."""
    with open(dataset_json, 'rt') as f:
        dataset = json.load(f)
    labels = dataset.get('labels', {})
    label = None
    for k, v in labels.items():
        if isinstance(k, str) and label_name in k.lower():
            label = int(v)
        elif isinstance(v, str) and label_name in v.lower():
            label = int(k)
    return label

def label_mass(label_name:str):
    parser = argparse.ArgumentParser(description="Compute specific label volume (cm^3) and mass (g) from nnU-Net segmentations")
    parser.add_argument("--input", type=str, required=True, help="The directory path which conaining mask files output from nnU-Net framework")
    parser.add_argument("--dataset-json", type=str, required=True, help="nnU-Net dataset.json, used to display label names")
    parser.add_argument("--rho", type=float, default=1.055, help="Myocardium density in g/cm^3. Default is 1.055 g/cm^3.")
    args = parser.parse_args()

    out = 'density (g/cm^3): ' + str(args.rho) + '\n'
    for entry in os.scandir(args.input):
        if entry.is_file() and entry.name.lower().endswith(ALL_EXTS):
            data, spacing = load_segmentation(entry)
            label = get_label(args.dataset_json, label_name)
            if label is None:
                print(f"[Warning] Label '{label_name}' not found in {args.dataset_json}")
                continue
            volume_cm3 = compute_volumes(data, spacing, label)
            mass_g = volume_cm3 * args.rho
            out += f"{entry.name}: volume={volume_cm3:.3f} cm^3, mass={mass_g:.3f} g\n"

    txt_path = os.path.join(args.input, f'{label_name}_mass.txt')
    with open(txt_path, 'wt') as f:
        f.write(out)


if __name__ == "__main__":
    label_mass("myocardium")