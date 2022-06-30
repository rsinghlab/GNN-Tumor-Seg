import argparse
import os

import Filepaths
from .viz_helpers import get_data_for_plotting
from .VolumeViewingTool import multi_slice_viewer

#Plots brain volume with predictions overlaid.
#Can use j and k to navigate slices


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_dir', default=Filepaths.INPUT_MRI_DIR, help='path to the directory where data is stored',type=str)
parser.add_argument('-s', '--seg_dir', default="", help='Where the predicted segmentation is saved',type=str)
parser.add_argument('-i', '--mri_id', default=None,help='mri id you want to plot',type=str)
parser.add_argument('-l', '--plot_gt', default=False,help='whether to plot ground truth segmentation',action='store_true')
args = parser.parse_args()

flair_ext = "_flair.nii.gz"
t1ce_ext="_t1ce.nii.gz"

flair,t1ce,preds,gt = get_data_for_plotting(os.path.expanduser(args.data_dir),os.path.expanduser(args.seg_dir),args.mri_id,flair_ext,t1ce_ext,read_labels=args.plot_gt)


if(args.plot_gt):
    multi_slice_viewer([
        {'arr':flair, 'cmap':'gray', 'stride':1, 'title':"FLAIR"},
        {'arr':t1ce, 'cmap':'gray', 'stride':1, 'title':"T1CE"},
        {'arr':preds, 'cmap':'gray', 'stride':1, 'title':"Predictions"},
        {'arr':gt, 'cmap':'gray', 'stride':1, 'title':"Ground Truth"}
    ])
else:
    multi_slice_viewer([
        {'arr':flair, 'cmap':'gray', 'stride':1, 'title':"FLAIR"},
        {'arr':t1ce, 'cmap':'gray', 'stride':1, 'title':"T1CE"},
        {'arr':preds, 'cmap':'gray', 'stride':1, 'title':"Segmentation"},
    ])