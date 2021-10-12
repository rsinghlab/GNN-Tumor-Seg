import matplotlib.pyplot as plt
import numpy as np



#to_display is list of dicts of form {arr, cmap}
def multi_slice_viewer(to_display):
    remove_keymap_conflicts({'j', 'k'})
    num_subplots = len(to_display)
    nrows = 2
    ncols = (num_subplots+1)//2
    fig, axs = plt.subplots(2,ncols)
    i = 0
    j = 0
    curr_pos = 0
    while(curr_pos<num_subplots):
        curr_img = to_display[curr_pos]
        ax = axs[i][j] if num_subplots>2 else axs[i]
        arr = curr_img['arr']
        ax.volume = arr
        ax.index = arr.shape[2] // 2
        ax.stride = curr_img['stride']
        ax.cmap = curr_img['cmap']
        ax.imshow(arr[:,:,ax.index],cmap=ax.cmap)
        ax.set_title(curr_img['title'])
        if(j==ncols-1):
            i+=1
            j=0
        else:
            j+=1
        curr_pos = ncols*i+j
        #print("i",i)
        #print("j",j)

    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

#private
def process_key(event):
    fig = event.canvas.figure
    for ax in fig.axes:
        if(not hasattr(ax,'volume')):
            continue
        elif event.key == 'j':
            change_slice(ax,-1)
        elif event.key == 'k':
            change_slice(ax,1)
    fig.canvas.draw()

#private
def change_slice(ax,delta):
    volume = ax.volume
    ax.index = (ax.index + delta*ax.stride) % volume.shape[2]
    #ax.images[0].set_array(volume[:,:,int(ax.index)])
    #ax.images[0].set_cmap=
    ax.imshow(volume[:,:,ax.index],cmap=ax.cmap)

#private
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)