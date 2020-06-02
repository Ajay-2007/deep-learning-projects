from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt


class ImageTest:
    def __init__(self, image_path, polygon_size):
        """
        Initializes an instance of the class
        :param image_path: the path of the file containing the reference image
        :param polygon_size: the number of vertices on the polygons used to recreate the image
        """

        self.ref_image = Image.open(image_path)
        self.polygon_size = polygon_size

        self.width, self.height = self.ref_image.size
        self.num_pixels = self.width * self.height
        self.ref_image_cv2 = self.to_cv2(self.ref_image)

    def polygon_data_to_image(self, polygon_data):
        """
        accepts polygon data and creates an image containing these polygons.
        :param polygon_data: a list of polygon parameters. Each item in the list represents the vertices
        locations, color and transparency of the corresponding polygon
        :return: the image containing the polygons ( Pillow format)
        """
        # start with a new image
        image = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(image, "RGBA")

        # divide the polygon_data to chunks, each containing the data for a single polygon
        chunk_size = self.polygon_size * 2 + 4 # (x, y) per vertex + (RGBA)
        polygons = self.list2_chunks(polygon_data, chunk_size)

        # iterate over all polygons and draw each of them into the image
        for poly in polygons:
            index = 0

            # extract the vertices of the current polygon:
            vertices = []
            for vertex in range(self.polygon_size):
                vertices.append(
                    (int(poly[index] * self.width), int(poly[index + 1] * self.height))
                )
                index += 2

            # extract the RGB and alpha values of the current polygon:
            red = int(poly[index] * 255)
            green = int(poly[index + 1] * 255)
            blue = int(poly[index + 2] * 255)
            alpha = int(poly[index + 3] * 255)

            # draw the polygon into the image:
            draw.polygon(vertices, (red, green, blue, alpha))

        # cleanup:
        del draw

        return image

    def get_difference(self, polygon_data, method="MSE"):
        """
        accepts polygon data, creates an image containing these polygons, and calculates the difference between
        this image and the reference image using one of two methods.
        :param polygon_data: a list of polygon parameters. Each item in the list represents the vertices
        locations, color and transparency of the corresponding polygon
        :param method: base method of calculating the difference ("MSE", "SSIM"). larger return value always means
        larger difference
        :return: the calculated difference between the image containing the polygons and the reference image
        """

        # create the image containing the polygons:
        image = self.polygon_data_to_image(polygon_data)

        if method == "MSE":
            return self.get_mse(image)
        else:
            return 1.0 - self.get_ssim(image)

    def plot_image(self, image, header=None):
        """
        creates a "side-by-side" plot of the given image next to the reference image
        :param image: image to be drawn next to reference image (Pillow format)
        :param header: text used as a header for the plot
        :return:
        """

        fig = plt.figure("Image Comparison")
        if header:
            plt.title(header)

        # plot the reference image on the left:
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(self.ref_image)
        self.ticks_off(plt)

        # plot the given image on the right:
        fig.add_subplot(1, 2, 2)
        plt.imshow(image)
        self.ticks_off(plt)

        return plt

    def save_image(self, polygon_data, image_file_path, header=None):
        """
        accepts polygon data, creates an image containing these polygons,
        creates a "side-by-side" plot of this image next to the reference image, and saves the plot to a file
        :param polygon_data: a list of polygon parameters. Each item in the list represents the vertices
        locations, color and transparency of the corresponding polygon
        :param image_file_path: path of file to be used to save the plot to
        :param header: text used as a header for the plot
        """

        # create an image from the polygon data:
        image = self.polygon_data_to_image(polygon_data)

        # plot the image side-by-side with the reference image:
        self.plot_image(image, header)

        # save the plot to file:
        plt.save_fig(image_file_path)


    # utility methods:

    def to_cv2(self, pil_image):
        """converts the given Pillow image to CV2 format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def get_mse(self, image):
        """calculates MSE of difference between the given image and the reference image"""
        return np.sum((self.to_cv2(image).astype("float") - self.ref_image_cv2.astype("float")) ** 2)/float(self.num_pixels)

    def get_ssim(self, image):
        """calculates mean structural similarity index between the given image and the reference image"""
        return structural_similarity(self.to_cv2(image), self.ref_image_cv2, multichannel=True)

    def list2_chunks(self, list, chunk_size):
        """divides a given list to fixed size chunks, returns a generator iterator"""
        for chunk in range(0, len(list), chunk_size):
            yield(list[chunk : chunk + chunk_size])

    def ticks_off(self, plot):
        """turns off ticks on both axes"""
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            top=False,
            right=False,
            labelbottom=False,
            lebelleft=False,
        )