from skimage import io, color
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
import skimage
from sklearn import cluster
import numpy as np
import scipy.misc as sp_mi
import cv2


dir_img = ['/Users/georgegkourlias/Downloads/rose1.jpg',
           '/Users/georgegkourlias/Downloads/0.jpg']

# RGB Raster
#   The grayscale image
rgb_image_test = sp_mi.imread(dir_img[1])

#   The RGB image
rgb_image_train = sp_mi.imread(dir_img[0])

# Lab Raster
lab_image_test = color.rgb2lab(rgb_image_test)
lab_image_train = color.rgb2lab(rgb_image_train)

# ******* COLOR SPACE QUANTIZATION *******
def quantize(raster, nColors):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))
    # if we want to cluster only <a,b> from Lab we 'll
    # pass into model the "reshaped_raster[:,1:]"
    # else if we want to cluster and luminance we 'll
    # pass into model the reshaped_raster

    a_b = reshaped_raster[:,1:]

    model = cluster.KMeans(n_clusters=nColors)
    labels = model.fit_predict(a_b)
    palette = model.cluster_centers_

    array_clustered = np.hstack([reshaped_raster[:,:1],palette[labels]])
    quantized_raster = np.reshape(
        array_clustered, (width, height, palette.shape[1]+1)
    )

    return quantized_raster, palette

# Quantize test image into 16 bins->palettes
quantize_image_on_lab, pal = quantize(lab_image_train, 16)

# Convert to RGB for plotting the quantized image
rgb_raster = color.lab2rgb(quantize_image_on_lab)
rgb_raster = (rgb_raster*255).astype('uint8')


# Plot to see the quantized space and before quantize
# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(rgb_image_train)
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(rgb_raster)
# plt.show()



# ******* SLIC ALGORITHM FOR IMAGE SEGMENTATION *******
segments_slic_train = slic(rgb_image_train, n_segments=250, compactness=10, sigma=1)
segments_slic_test = slic(rgb_image_test, n_segments=250, compactness=10, sigma=1)
print("The SLIC segments for train: {}".format(len(np.unique(segments_slic_train))) )
print("The SLIC segments for test: {}".format(len(np.unique(segments_slic_test))) )

# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(mark_boundaries(rgb_image_test, segments_slic_test))
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(mark_boundaries(rgb_image_train, segments_slic_train))
# plt.show()


# ******* SURF FEATURE EXTRACTION *******
surf_algorithm = cv2.xfeatures2d.SURF_create()

# Local features from SURF
def local_surf(image, trainImage):
    # Accessing Individual Super_Pixel Segmentation
    kps = []
    dscs = []
    if(trainImage):
        for (i, segNum) in enumerate(np.unique(segments_slic_train)):
            # print("[x] inspecting segment %d " %i )
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask[segments_slic_train == segNum] = 255

            # Draw current SuperPixel and its mask
            # cv_result = cv2.bitwise_and(image, image, mask=mask)
            # cv2.imshow("Mask", mask)
            # cv2.imshow("Applied", cv_result)
            # cv2.waitKey(0)

            # SURF -> KeyPoints and Descriptors
            keypoints, descriptors = surf_algorithm.detectAndCompute(image, mask)
            kps.append(keypoints)
            dscs.append(descriptors)


            # Print the extracted features per SuperPixel
            # rgb_image_train_1 = cv2.drawKeypoints(image, keypoints, None)
            # cv2.imshow("surf", rgb_image_train_1)
            # cv2.waitKey(0)
    else:
        for (i, segNum) in enumerate(np.unique(segments_slic_test)):
            # print("[x] inspecting segment %d " %i )
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask[segments_slic_test == segNum] = 255

            # Draw current SuperPixel and its mask
            # cv_result = cv2.bitwise_and(image, image, mask=mask)
            # cv2.imshow("Mask", mask)
            # cv2.imshow("Applied", cv_result)
            # cv2.waitKey(0)

            # SURF -> KeyPoints and Descriptors
            keypoints, descriptors = surf_algorithm.detectAndCompute(image, mask)
            kps.append(keypoints)
            dscs.append(descriptors)

            # Print the extracted features per SuperPixel
            # rgb_image_test_1 = cv2.drawKeypoints(image, keypoints, None)
            # cv2.imshow("surf", rgb_image_test_1)
            # cv2.waitKey(0)
    return kps, dscs


# Global features from SURF
def global_surf(image):
    keypoints, descriptors = surf_algorithm.detectAndCompute(image, None)

    # If you want to draw it
    rgb_image_train_1 = cv2.drawKeypoints(image, keypoints, None)
    cv2.imshow("surf", rgb_image_train_1)
    cv2.waitKey(0)
    return  keypoints, descriptors


kp_train_image, ds_train_image = local_surf(rgb_image_train, True)
kp_test_image, ds_test_image = local_surf(rgb_image_test, False)
print("kp: ",kp_train_image, " ds: ",ds_train_image)

# Call global surf
# kp, ds = global_surf(rgb_image_train)

cv2.destroyAllWindows()


# ******* GABOR FEATURES EXTRACTION *******

        # ksize - size of gabor filter (n, n)
        # sigma - standard deviation of the gaussian function
        # theta - orientation of the normal to the parallel stripes
        # lambda - wavelength of the sunusoidal factor
        # gamma - spatial aspect ratio
        # psi - phase offset
        # ktype - type and range of values that each pixel in the gabor kernel can hold

def gabor_feature_extraction(image):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)

    cv2.imshow('filtered image', filtered_image)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)

    # Print filter kernel
    # cv2.imshow('gabor kernel (resized)', g_kernel)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return filtered_image, g_kernel

filter_img_train, kernel_train = gabor_feature_extraction(rgb_image_train)
filter_img_test, kernel_test = gabor_feature_extraction(rgb_image_test)
