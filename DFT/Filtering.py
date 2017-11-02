# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import numpy
import copy

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff, order = 0):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""

        mask = numpy.zeros((shape[0],shape[1]))

        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                x = ((j-mask.shape[0]/2)**2 + (i-mask.shape[1]/2)**2)**0.5
                if x <= cutoff:
                    mask[j][i] = 1

        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff, order = 0):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        mask = self.get_ideal_low_pass_filter(shape, cutoff, 0)
        mask = 1 - mask
        
        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        mask = numpy.zeros((shape[0],shape[1]))

        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                x = ((j-mask.shape[0]/2)**2 + (i-mask.shape[1]/2)**2)**0.5
                mask[j][i] = 1/(1 + ((x/cutoff)**(2*order)))

        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        mask = numpy.zeros((shape[0],shape[1]))

        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                x = ((j-mask.shape[0]/2)**2 + (i-mask.shape[1]/2)**2)**0.5
                y = 0
                if x != 0:
                    y = cutoff/x
                mask[j][i] = 1/(1 + ((y)**(2*order)))
        
        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff, order = 0):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""

        mask = numpy.zeros((shape[0],shape[1]))

        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                x = -((j-mask.shape[0]/2)**2 + (i-mask.shape[1]/2)**2)
                mask[j][i] = numpy.exp(x/((2*cutoff)**2))
        
        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff, order = 0):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        mask = self.get_gaussian_low_pass_filter(shape, cutoff, 0)
        mask = 1 - mask
        
        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        image = numpy.fft.ifftshift(image)
        #return image
        image = numpy.fft.ifft2(image)
        #image = numpy.absolute(image)
        #return image
        image = numpy.absolute(numpy.exp(image))

        image_temp = copy.deepcopy(image)

        maxval = image_temp.max()
        minval = image_temp.min()

        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                image_temp[j][i] = ((255)/(maxval - minval))*(image[j][i] - minval)

        image = image_temp.astype(numpy.uint8)

        image = 255 - image

        return image


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """

        fftimage = self.image
        fftimage = numpy.fft.fftshift(numpy.fft.fft2(fftimage))
        fftimage = numpy.log(numpy.absolute(fftimage))
        fftimage_temp = fftimage

        maxval = fftimage.max()
        minval = fftimage.min()

        for j in range(fftimage.shape[0]):
            for i in range(fftimage.shape[1]):
                fftimage_temp[j][i] = ((255)/(maxval - minval))*(fftimage[j][i] - minval)

        fftimage = fftimage_temp.astype(numpy.uint8)

        shape = (fftimage.shape[1], fftimage.shape[0]);
        mask = self.filter(shape, self.cutoff, self.order)

        filteredDFT = fftimage*mask

        new_image = self.post_process_image(filteredDFT)
                
        return [new_image, fftimage, filteredDFT]


##python dip_hw2_filter.py -i Lenna0.jpg -m ideal_l -c 50