import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

flow1 = np.loadtxt('flow_vectors_if.txt') * 1000
flow2 = np.loadtxt('flow_vectors_if2.txt') * 1000
image1 = '/home/francesco/event_planar_fra/results_inference/a56eb6d230f14d71b32fe5f1d0a5e5f6/results/eval_7/poster_rotation_090/flow/000000174.png'
image2 = '/home/francesco/event_planar_fra/results_inference/a56eb6d230f14d71b32fe5f1d0a5e5f6/results/eval_7/poster_rotation_090/flow/000000468.png'
image3 = '/home/francesco/event_planar_fra/results_inference/a56eb6d230f14d71b32fe5f1d0a5e5f6/results/eval_7/poster_rotation_090/flow/000000786.png'
image4 = '/home/francesco/event_planar_fra/results_inference/a56eb6d230f14d71b32fe5f1d0a5e5f6/results/eval_8/poster_rotation_190/flow/000000567.png'
image5 = '/home/francesco/event_planar_fra/results_inference/a56eb6d230f14d71b32fe5f1d0a5e5f6/results/eval_8/poster_rotation_190/flow/000000825.png'
image6 = '/home/francesco/event_planar_fra/results_inference/a56eb6d230f14d71b32fe5f1d0a5e5f6/results/eval_8/poster_rotation_190/flow/000000976.png'
colormap = '/home/francesco/Desktop/colormap.png'

plt.figure(figsize=(18, 6))
f = 18
# Load and plot the first image
img1 = mpimg.imread(image1)
plt.subplot(2, 6, 1)
plt.imshow(img1)
plt.axis('off')

# Load and plot the second image
img2 = mpimg.imread(image2)
plt.subplot(2, 6, 2)
plt.imshow(img2)
plt.axis('off')

# Load and plot the third image
img3 = mpimg.imread(image3)
plt.subplot(2, 6, 3)
plt.imshow(img3)
plt.axis('off')

# Load and plot the third image
img4 = mpimg.imread(image4)
plt.subplot(2, 6, 4)
plt.imshow(img4)
plt.axis('off')

# Load and plot the third image
img5 = mpimg.imread(image5)
plt.subplot(2, 6, 5)
plt.imshow(img5)
plt.axis('off')

# Load and plot the third image
img6 = mpimg.imread(image6)
plt.subplot(2, 6, 6)
plt.imshow(img6)
plt.axis('off')

time = np.arange(0, len(flow1[0])*5, 5)/1000
plt.subplot(2, 2, 3)
plt.plot(time, flow1[0], label='u', color='orangered')
plt.plot(time, flow1[1], label='v', color='royalblue')
plt.xlabel('Time [s]', fontsize=f)
plt.ylabel('Optical Flow [px/s]', fontsize=f)
plt.xticks(fontsize=f)
plt.yticks(np.arange(-8, 12, 4), fontsize=f)
# plt.grid()
plt.legend(fontsize=f)

plt.subplot(2, 2, 4)
plt.plot(time[300:1000], flow2[0][300:1000], label='u', color='orangered')
plt.plot(time[300:1000], flow2[1][300:1000], label='v', color='royalblue')
plt.xlabel('Time [s]', fontsize=f)
# plt.ylabel('Optical Flow [pxl/ms]', fontsize=f)
plt.xticks(fontsize=f)
plt.yticks(np.arange(-8, 12, 4), fontsize=f)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True, labelleft=False)
# plt.grid()
plt.legend(fontsize=f)

plt.tight_layout()
plt.savefig('poster_flow2.pdf')
plt.show()
