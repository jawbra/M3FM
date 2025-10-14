
#%%
import matplotlib.pyplot as plt
import numpy as np
#%%

image = np.load('test_image.npy')
mask = np.load('test_mask.npy')
image.shape

plt.imshow(image[0,0,0,:,:], cmap='gray')
# %%
