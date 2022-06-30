
import argparse
import os

import Filepaths
from .viz_helpers import *


#plots hardcoded slices 

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_dir', default=Filepaths.INPUT_MRI_DIR, help='path to the directory where data is stored',type=str)
parser.add_argument('-s', '--seg_dir', default="", help='Where the predicted segmentation is saved',type=str)
parser.add_argument('-i', '--mri_id', default=None,help='mri id you want to plot',type=str)
parser.add_argument('-cp', '--coronal', default=100,help='coronal plane',type=int)
parser.add_argument('-sp', '--sagittal', default=100,help='sagittal plane',type=int)
parser.add_argument('-hp', '--horizontal', default=100,help='horizontal plane',type=int)
parser.add_argument('-l', '--plot_gt', default=False,help='whether to plot ground truth segmentation',action='store_true')
args = parser.parse_args()

flair_ext = "_flair.nii.gz"
t1ce_ext="_t1ce.nii.gz"

mod1,mod2,preds,gt = get_data_for_plotting(os.path.expanduser(args.data_dir),os.path.expanduser(args.seg_dir),args.mri_id,flair_ext,t1ce_ext,read_labels=args.plot_gt)

hs = args.horizontal #horizontal slice
cs = args.coronal #coronal slice
ss = args.sagittal #saggittal slice
#These numbers will require some tuning to get slices that actually have tumor for a specific brain


horizontal_preds = preds[:,:,hs]
horizontal_mod1 = mod1[:,:,hs]
horizontal_mod2 = mod2[:,:,hs]
horizontal_gt = gt[:,:,hs]

coronal_preds = preds[:,cs,:]
coronal_mod1 = mod1[:,cs,:]
coronal_mod2 = mod2[:,cs,:]
coronal_gt = gt[:,cs,:]

saggittal_preds = preds[ss,:,:]
saggittal_mod1 = mod1[ss,:,:]
saggittal_mod2 = mod2[ss,:,:]
saggittal_gt = gt[ss,:,:]



if(args.plot_gt):
    fig, axs = plt.subplots(4,3,figsize=(12,6))
    plot_four_on_axis(axs[:,0],horizontal_mod1,horizontal_mod2,horizontal_preds,horizontal_gt)
    plot_four_on_axis(axs[:,1],coronal_mod1,coronal_mod2,coronal_preds,coronal_gt)
    plot_four_on_axis(axs[:,2],saggittal_mod1,saggittal_mod2,saggittal_preds,saggittal_gt)

else:
    fig, axs = plt.subplots(3,3,figsize=(12,6))
    plot_three_on_axis(axs[0],horizontal_mod1,horizontal_mod2,horizontal_preds)
    plot_three_on_axis(axs[1],coronal_mod1,coronal_mod2,coronal_preds)
    plot_three_on_axis(axs[2],saggittal_mod1,saggittal_mod2,saggittal_preds)

fig.tight_layout(pad=0)
fig.subplots_adjust(wspace=-0.7)

# fig.text(1, 1, 'Ground Truth', rotation=90,fontsize=14)
# fig.text(-530, 110, 'Predicted', rotation=90,fontsize=14)

plt.show()