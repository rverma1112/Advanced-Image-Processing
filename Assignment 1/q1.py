import scipy.ndimage
import skimage.color
import skimage.io
import skimage.transform
from matplotlib import pyplot as plt
import numpy as np

img1 = skimage.color.rgb2gray(skimage.io.imread('books.jpg'))
img2 = skimage.color.rgb2gray(skimage.io.imread('building.jpg'))

img1_halved = skimage.transform.rescale(img1, 0.5, anti_aliasing=False)
img2_halved = skimage.transform.rescale(img2, 0.5, anti_aliasing=False)


img1_doubled = skimage.transform.rescale(img1, 2.0, anti_aliasing=False)
img2_doubled = skimage.transform.rescale(img2, 2.0, anti_aliasing=False)

img1_blurred = scipy.ndimage.gaussian_filter(img1, 3)
img2_blurred = scipy.ndimage.gaussian_filter(img2, 3)

img1_rotated = skimage.transform.rotate(img1, 25)
img2_rotated = skimage.transform.rotate(img2, 25)


img1_noisy = skimage.util.random_noise(img1, mode='gaussian')
img2_noisy = skimage.util.random_noise(img2, mode='gaussian')


def extrema_detection(low, mid, up, h, w):
    extrema_points = np.zeros([h, w])

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            value = mid[i, j]
            mid8 = mid[i - 1:i + 2, j - 1:j + 2]
            mid8[1, 1] = value - 1  

            mid8 = mid8 >= value
            up9 = up[i - 1:i + 2, j - 1:j + 2] >= value
            low9 = low[i - 1:i + 2, j - 1:j + 2] >= value

            extrema_flag = np.sum(mid8) + np.sum(up9) + np.sum(low9)

            if extrema_flag == 0:  
                extrema_points[i, j] = 1

            mid8 = mid[i - 1:i + 2, j - 1:j + 2]
            mid8[1, 1] = value + 1

            mid8 = mid8 <= value
            up9 = up[i - 1:i + 2, j - 1:j + 2] <= value
            low9 = low[i - 1:i + 2, j - 1:j + 2] <= value

            extrema_flag = np.sum(mid8) + np.sum(up9) + np.sum(low9)

            if extrema_flag == 0:  
                extrema_points[i, j] = 1

    
    number=0
    extrema_points_image = np.zeros([h, w])
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if extrema_points[i, j] == 1:
                extrema_points_image[i - 1:i + 2, j - 1:j + 2] = extrema_points_image[i - 1:i + 2, j - 1:j + 2] + 1
                number=number+1
    extrema_points_image = extrema_points_image >= 1

    return extrema_points_image,number


def scale_space_extrema(image, sigma, s, k):
    
    gaussians = [None] * s
    
    for i in range(s):
        gaussians[i] = scipy.ndimage.gaussian_filter(image, sigma * (k ** i)) * 255

    DOGs = [None] * (s - 1)
    for i in range(s - 1):
        DOGs[i] = gaussians[i + 1] - gaussians[i]

    extrema_points_matrices = [None] * (s - 3)
    number= [None] * (s - 3)
    for i in range(s - 3):
        extrema_points_matrices[i],number[i] = extrema_detection(DOGs[i].copy(), DOGs[i + 1].copy(), DOGs[i + 2].copy(), image.shape[0],
                                                        image.shape[1])

    for i in range(s - 3):
        plt.subplot(1, 2, i+1)
        plt.imshow(image+extrema_points_matrices[i], cmap='gray')
    print('Number of Extrema Points: ')
    print(number)
    plt.show()
    


if __name__ == "__main__":

    sigma = 1.5
    s = 5  
    k = 2 ** (1 / (s - 1))  
    
    print('Original Image')
    scale_space_extrema(img1, sigma, s, k)
    scale_space_extrema(img2, sigma, s, k)
    
    print('Downscaled Image')
    scale_space_extrema(img1_halved, sigma, s, k)
    scale_space_extrema(img2_halved, sigma, s, k)
    
    print('Upscaled Image')
    scale_space_extrema(img1_doubled, sigma, s, k)
    scale_space_extrema(img2_doubled, sigma, s, k)
    
    print('Gaussian Blurred Image')
    scale_space_extrema(img1_blurred, sigma, s, k)
    scale_space_extrema(img2_blurred, sigma, s, k)
    

    print('Rotated Image')
    scale_space_extrema(img1_rotated, sigma, s, k)
    scale_space_extrema(img2_rotated, sigma, s, k)
    

    print('Gaussian Noise Image')
    scale_space_extrema(img1_noisy, sigma, s, k)
    scale_space_extrema(img2_noisy, sigma, s, k)
    
