#include <CImg.h>
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>

using cimg_library::CImg;
using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// Constants
const bool displayImages = false;
const bool saveAllImages = true;
const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 4;
const unsigned int CONTRAST_THRESHOLD = 80;
const float filter[] = {	1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f,
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f,
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

extern void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);
extern void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);

extern void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH);
extern void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH);

extern void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);
extern void contrast1DCuda(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);

extern void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter);
extern void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter);


int main(int argc, char *argv[])
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( displayImages ) {
		inputImage.display("Input Image");
	}
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	// Convert the input image to grayscale
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	//rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
	rgb2grayCuda(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());

	if ( displayImages ) {
		grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./grayscale.bmp");
	}

	// Compute 1D histogram
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];

	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);
	//histogram1DCuda(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);

	if ( displayImages ) {
		histogramImage.display("Histogram");
	}
	if ( saveAllImages ) {
		histogramImage.save("./histogram.bmp");
	}

	// Contrast enhancement
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);
	//contrast1DCuda(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);

	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./contrast.bmp");
	}

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	//triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
	triangularSmoothCuda(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);

	if ( displayImages ) {
		smoothImage.display("Smooth Image");
	}

	if ( saveAllImages ) {
		smoothImage.save("./smooth.bmp");
	}

	return 0;
}

