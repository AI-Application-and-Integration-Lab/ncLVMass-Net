import os
import argparse
import tempfile
import json
import sys
import shutil
from huggingface_hub import snapshot_download, try_to_load_from_cache
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
from nnunetv2.postprocessing.remove_connected_components import entry_point_apply_postprocessing
from nclvmnet.label_mass import label_mass


def main():
    parser = argparse.ArgumentParser(description="ncLVMass-Net: A tool for estimating left ventricular mass from non-contrast CT images using nnU-Net.")
    parser.add_argument('-i', '--input', type=str, required=True, help="A CT nifti image path or a folder containing CT nifti images.")
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder path.')
    parser.add_argument('--mdl-local-dir', type=str, default=None, help='Local directory of the model folder.')
    parser.add_argument('--hf-token', type=str, default=None, help='Hugging Face token for accelerate download process.')
    parser.add_argument('-d', '--dataset', type=str, default=None, help='Select a dataset in the model folder to predict. If not specified, the first finding dataset will be selected.')
    parser.add_argument('-tr', '--trainer', type=str, default=None, help='What nnU-Net trainer class was used for training? If not specified, the first finding folder will be selected.')
    parser.add_argument('-p', '--plans', type=str, default=None, help='Plans identifier. If not specified, the first finding folder will be selected.')
    parser.add_argument('-c', '--config', type=str, default=None, help='nnU-Net configuration (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres). If not specified, the first finding folder will be selected.')
    parser.add_argument('-f', '--fold', nargs='+', type=str, default=['0','1','2','3','4'], help='Which fold(s) to use for prediction. Default is depends on first finding "crossval_results_folds_x_x_x..." folder. If not exist, select "all" if exist, otherwise select the first fold.')
    parser.add_argument('-chk', '--checkpoint', type=str, default='checkpoint_best.pth', help='Checkpoint file name. Should be checkpoint_best.pth (default) or checkpoint_final.pth.')
    parser.add_argument('--myo-density', type=float, default=1.055, help='Myocardium density in g/cm^3. Default is 1.055 g/cm^3.')
    args = parser.parse_args()

    # ---------------------------------
    # verify and update input arguments
    # ---------------------------------
    if args.mdl_local_dir is None:
        # Download repository from huggingface and get the local path of the model folder
        repo_id = "NTU-AI2Lab/ncLVMass-Net"
        readme_path = try_to_load_from_cache(repo_id=repo_id, filename="README.md")
        if readme_path is not None:
            args.mdl_local_dir = os.path.dirname(readme_path)
            print('ncLVMass-Net model weights already downloaded.')
        else:
            if args.hf_token is None:
                args.mdl_local_dir = snapshot_download(repo_id=repo_id)
            else:
                args.mdl_local_dir = snapshot_download(repo_id=repo_id, token=args.hf_token)
    else:
        if not os.path.isdir(args.mdl_local_dir):
            raise ValueError(f"The specified model local directory does not exist: {args.mdl_local_dir}")

    # -------------------
    # find Dataset folder
    # -------------------
    if args.dataset is None:
        args.dataset = next(f.name for f in os.scandir(args.mdl_local_dir) if f.is_dir() and f.name.startswith("Dataset"))
    else:
        if not os.path.isdir(os.path.join(args.mdl_local_dir, args.dataset)):
            if args.dataset.isdecimal():
                args.dataset = next(f.name for f in os.scandir(args.mdl_local_dir) if f.is_dir() and f.name.startswith(f"Dataset{args.dataset}"))
            else:
                args.dataset = next(f.name for f in os.scandir(args.mdl_local_dir) if f.is_dir() and f.name.startswith("Dataset") and args.dataset in f.name)

    if args.trainer is None or \
       args.plans is None or \
       args.config is None or \
       not os.path.isdir(os.path.join(args.mdl_local_dir, args.dataset, f'{args.trainer}__{args.plans}__{args.config}')):
        # find plans, trainer, config
        for f in os.scandir(os.path.join(args.mdl_local_dir, args.dataset)):
            if f.is_dir() and '__' in f.name:
                parts = f.name.split('__')
                if len(parts) == 3:
                    args.trainer, args.plans, args.config = parts
                    break

    subfolder = os.path.join(args.mdl_local_dir, args.dataset, f'{args.trainer}__{args.plans}__{args.config}')
    if os.path.isdir(os.path.join(subfolder, 'crossval_results_folds_' + '_'.join(args.fold))):
        postprocess = True
        for f in args.fold:
            assert os.path.isdir(os.path.join(subfolder, f'fold_{f}')), 'Fold folder does not exist: ' + os.path.join(subfolder, f'fold_{f}')
            if not os.path.isfile(os.path.join(subfolder, f'fold_{f}', args.checkpoint)):
                if f in {'0', 'all'}:
                    for chk in ('checkpoint_best.pth', 'checkpoint_final.pth'):
                        if os.path.isfile(os.path.join(subfolder, f'fold_{f}', chk)):
                            args.checkpoint = chk
                            break
                    assert os.path.isfile(os.path.join(subfolder, f'fold_{f}', args.checkpoint)), 'Checkpoint file does not exist: ' + os.path.join(subfolder, f'fold_{f}', args.checkpoint)
                else:
                    raise FileNotFoundError('Checkpoint file does not exist: ' + os.path.join(subfolder, f'fold_{f}', args.checkpoint))
    else:
        crf = next((f.name for f in os.scandir(subfolder) if f.is_dir() and f.name.startswith('crossval_results_folds_')), None)
        if crf is not None:
            if crf.split('_')[3:] != args.fold:
                print(f'Warning: The specified folds {args.fold} do not match the folds in the cross-validation results folder "{crf}". Using folds from the folder instead.')
                args.fold = crf.split('_')[3:]
            postprocess = True
            for f in args.fold:
                assert os.path.isdir(os.path.join(subfolder, f'fold_{f}')), 'Fold folder does not exist: ' + os.path.join(subfolder, f'fold_{f}')
                if not os.path.isfile(os.path.join(subfolder, f'fold_{f}', args.checkpoint)):
                    if f in {'0', 'all'}:
                        for chk in ('checkpoint_best.pth', 'checkpoint_final.pth'):
                            if os.path.isfile(os.path.join(subfolder, f'fold_{f}', chk)):
                                args.checkpoint = chk
                                break
                        assert os.path.isfile(os.path.join(subfolder, f'fold_{f}', args.checkpoint)), 'Checkpoint file does not exist: ' + os.path.join(subfolder, f'fold_{f}', args.checkpoint)
                    else:
                        raise FileNotFoundError('Checkpoint file does not exist: ' + os.path.join(subfolder, f'fold_{f}', args.checkpoint))
        else:
            postprocess = False
            fold_folder = next((f.name for f in os.scandir(subfolder) if f.is_dir() and f.name.startswith('fold_')), None)
            if fold_folder is not None:
                if args.fold != [fold_folder.split('_')[1]]:
                    print(f'Warning: The specified folds {args.fold} do not match the fold in the folder "{fold_folder}". Using fold from the folder instead.')
                    args.fold = [fold_folder.split('_')[1]]
                if not os.path.isfile(os.path.join(subfolder, fold_folder, args.checkpoint)):
                    for chk in ('checkpoint_best.pth', 'checkpoint_final.pth'):
                        if os.path.isfile(os.path.join(subfolder, fold_folder, chk)):
                            args.checkpoint = chk
                            break
                    assert os.path.isfile(os.path.join(subfolder, fold_folder, args.checkpoint)), 'Checkpoint file does not exist: ' + os.path.join(subfolder, fold_folder, args.checkpoint)
            else:
                raise FileNotFoundError('No fold folder found in: ' + subfolder)
   
    # ---------------------------------------
    # check input image file name and output folder
    # ---------------------------------------
    if os.path.exists(args.input):
        args.input = os.path.abspath(args.input)
    else:
        raise FileNotFoundError(f"The specified input path does not exist: {args.input}")

    dataset_json_path = os.path.join(subfolder, 'dataset.json')
    if not os.path.isfile(dataset_json_path):
        raise FileNotFoundError(f"dataset.json file does not exist in the model folder: {dataset_json_path}")
    with open(dataset_json_path, 'rt') as f:
        dataset_json = json.load(f)

    # find the CT channel index
    ct_channel = None
    for k, v in dataset_json['channel_names'].items():
        if v.upper()=='CT':
            ct_channel = k.zfill(4)
            break
    assert ct_channel is not None, 'CT channel not found in dataset.json'

    # build a temporary folder and copy input images in it, filename adding the CT channel suffix if necessary
    if os.path.isfile(args.input):
        src_img_dir = os.path.dirname(args.input)
        src_img_name = [os.path.basename(args.input)]
    else:
        src_img_dir = args.input
        src_img_name = [f.name for f in os.scandir(src_img_dir) if f.is_file() and f.name.endswith(dataset_json['file_ending'])]
    if len(src_img_name) == 0:
        raise FileNotFoundError(f"No input image files found in the specified path: {args.input}")
    extL = len(dataset_json['file_ending'])
    tmp_img_dir = tempfile.mkdtemp(dir=src_img_dir)
    tmp_img_name = []
    for src_name in src_img_name:
        src = os.path.join(src_img_dir, src_name)
        if src_name.endswith(f'_{ct_channel}{dataset_json["file_ending"]}'):
            tmp_img_name.append(src_name)
        else:
            tmp_img_name.append(f'{src_name[:-extL]}_{ct_channel}{dataset_json["file_ending"]}') # add channel
        dst = os.path.join(tmp_img_dir, tmp_img_name[-1])
        shutil.copyfile(src, dst)

    # prepare output folder
    if os.path.exists(args.output):
        if not os.path.isdir(args.output):
            raise NotADirectoryError(f"The specified output path is not a directory: {args.output}")
    else:
        os.makedirs(args.output)
     
    # ----------------------------------------
    # create environment variables for nnU-Net
    # ----------------------------------------
    if os.environ.get('nnUNet_raw') is None:
        os.environ['nnUNet_raw'] = '<path_to_nnUNet_raw>'
    if os.environ.get('nnUNet_preprocessed') is None:
        os.environ['nnUNet_preprocessed'] = '<path_to_nnUNet_preprocessed>'
    # overwrite nnUNet_results to the model folder
    os.environ['nnUNet_results'] = args.mdl_local_dir

    # ======================
    # run nnU-Net prediction
    # ======================
    if postprocess:
        print('Starting prediction...')
        output_predict = os.path.join(tmp_img_dir, 'predicted')
        sys.argv = ["predict_entry_point", "-i", tmp_img_dir, "-o", output_predict, "-d", args.dataset, "-tr", args.trainer, "-p", args.plans, "-c", args.config, "-f"] + args.fold + ["-chk", args.checkpoint]
        predict_entry_point()
        
        # post-processing
        print('Applying post-processing...')
        pkl_file = os.path.join(subfolder, 'crossval_results_folds_' + '_'.join(args.fold), 'postprocessing.pkl')
        plans_json_file = os.path.join(subfolder, 'plans.json')
        sys.argv = ["entry_point_apply_postprocessing", "-i", output_predict, "-o", args.output, "-pp_pkl_file", pkl_file, "-plans_json", plans_json_file, "-dataset_json", dataset_json_path]
        entry_point_apply_postprocessing()

        # Move other files (e.g., *.json) in the output_predict folder to the final output folder
        for f in os.scandir(output_predict):
            if f.is_file() and not f.name.endswith(dataset_json['file_ending']):
                os.rename(f.path, os.path.join(args.output, f.name))
    else:
        print('Starting prediction...')
        sys.argv = ["predict_entry_point", "-i", tmp_img_dir, "-o", args.output, "-d", args.dataset, "-tr", args.trainer, "-p", args.plans, "-c", args.config, "-f"] + args.fold + ["-chk", args.checkpoint]
        predict_entry_point()

    # remove image files and temp directory after prediction
    shutil.rmtree(tmp_img_dir)

    # ----------------
    # estimate LV mass
    # ----------------
    print('Estimating LV mass...')
    sys.argv = ['label_mass', '--input', args.output, '--dataset-json', dataset_json_path, '--rho', str(args.myo_density)]
    label_mass('myocardium')
    print('All done!')



if __name__ == "__main__":
    main()
