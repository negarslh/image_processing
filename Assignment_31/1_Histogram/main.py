import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogramFunc(image):
    hist = [np.sum(image == i) for i in range(256)]
    return hist

def plot_histogram(histogram, plot_type='plot', title="Histogram", xlabel="Pixel Value", ylabel="Frequency", output_path='output_image.jpg'):
    plt.figure()
    if plot_type == 'plot':
        plt.plot(histogram, color='magenta')
    elif plot_type == 'bar':
        plt.bar(range(len(histogram)), histogram, color='orange')
    elif plot_type == 'hist':
        plt.hist(histogram, bins=256, color='blue')
    else:
        raise ValueError("Unsupported plot_type. Use 'plot', 'bar', or 'hist'.")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.show()

image = cv2.imread('1_Histogram/input/image.jpg', cv2.IMREAD_GRAYSCALE)

histogram = histogramFunc(image)

plot_histogram(histogram, plot_type='plot', output_path='1_Histogram/output/plot_image.jpg')
plot_histogram(histogram, plot_type='bar', output_path='1_Histogram/output/bar_image.jpg')
plot_histogram(histogram, plot_type='hist', output_path='1_Histogram/output/hist_image.jpg')
