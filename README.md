# ncLVMass-Net
An AI model for left ventricular (LV) myocardium segmentation and mass quantification from non-contrast cardiac CT (NCCT) images. The model was trained with the [nnU-Net](https://github.com/mic-dkfz/nnunet) framework on 2,021 patients, all from Taiwan. It achieved an overall Dice score of 0.8827. To use it, provide an input image in NIfTI format (`.nii.gz`) and the model will output the myocardium mask, along with estimated LV mass.

# Quick Install
We recommend first installing [PyTorch](https://pytorch.org/get-started/locally/) in a Python environment (version 3.10 or later), and then installing ncLVMass-Net.
```
pip install nclvmass-net
```

# Usage
```
nclvmnet -i ct.nii.gz -o output_dir
nclvmnet -i input_dir -o output_dir
```

# Change pretrained model
The ncLVMass-Net pretrained model weights are automatically downloaded from [Hugging Face](https://huggingface.co/NTU-AI2Lab/ncLVMass-Net) and stored in the local Hugging Face cache. If you would like to use a different pretrained model, specify the model directory name with the `--dataset` argument and the parent directory containing the model with `--mdl-local-dir`.

For example, if you have previously downloaded the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) `heartchambers_highres` model (make sure you have obtained the appropriate [license](https://backend.totalsegmentator.com/license-academic/)), you can switch to this model for comparison.

```
nclvmnet -i ct.nii.gz -o output_dir_TS --mdl-local-dir $HOME/.totalsegmentator/nnunet/results --dataset Dataset301_heart_highres_1559subj
```

For non-commercial purposes, you may download the pretrained model weights directly from Hugging Face and fine-tune them on your own data.
```
hf download --local-dir $HOME/path/to/ncLVMass-Net/model NTU-AI2Lab/ncLVMass-Net
```

# Further development
The package directly invokes `nnUNetv2_predict` and `nnUNetv2_apply_postprocessing` with the specified input (`-i`), output (`-o`), and pretrained model (`--mdl-local-dir` and `--dataset`) paths, without modifying the original nnU-Net workflow. Note that the custom trainer used for other pretrained model (e.g., one with a non-default number of epochs) may not be available in the original nnU-Net codebase. For further customization or development, install the package in editable mode.
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
git clone https://github.com/AI-Application-and-Integration-Lab/ncLVMass-Net.git
pip install -e ./nnUNet
pip install -e ./ncLVMass-Net
```