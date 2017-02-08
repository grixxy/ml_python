import math
import numpy as np
import scipy.misc as msc
import matplotlib.pyplot as plt

def displayData(X, title='', example_width = 40):
# DISPLAYDATA Display 2 D data in a nice grid
# [h, display_array] = DISPLAYDATA(X, example_width) displays 2 D data
# stored in X in a nice grid.It returns the figure handle h and the
# displayed array if requested.

    plt.title(title)
    first_image = X[0:2,:]
    im_size = math.floor(math.sqrt(first_image.shape[1]))
    image = first_image.reshape(-1, im_size).T #cutting on y axis!
    #image_wrong = first_image.reshape(im_size, -1).T cutting on x axis don't work with image size!
    plt.imshow(image, cmap='gray')
    plt.show()


'''


# Gray Image
#colormap(gray)

# Compute rows, cols
    m, n = X.shape
    example_height = (n / example_width)

# Compute number of items to display
    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)

# Between images padding
    pad = 1

# Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

# Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m :
                break

            # Copy the patch
            #  Get the max value of the patch
            max_val = max(abs(X[curr_ex,:]))
            display_array[pad + j * (example_height + pad) + np.arange(example_height), pad + i * (example_width + pad)
                                                                                    + np.arange(example_width)] = X[curr_ex,:].reshape(example_height, -1) #/ max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break


    # Display Image

    #h = msc.imread(ml_dir + 'bird_small.png')
    h = msc.imread(display_array, [-1, 1])
'''