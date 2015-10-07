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


//kernel profiling
double kernelCpuTime [4];
double kernelGpuTime [4];
double kernelSpeedUp [4];

//application profiling 
double totalTimeCpu;
double totalTimeGpu;
double overallSpeedUp;


int main(int argc, char *argv[])
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

    	NSTimer AppTime = NSTimer("AppTime", false, false);
    	NSTimer AppTimeGPU = NSTimer("AppTimeGPU", false, false);


	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( displayImages ) {
		inputImage.display("Input Image");
	}
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	/* CPU Implementation */  

	AppTime.start();
	// Convert the input image to grayscale
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());

	if ( displayImages ) {
		grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./CPU/grayscale.bmp");
	}

	// Compute 1D histogram
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];

	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);

	if ( displayImages ) {
		histogramImage.display("Histogram");
	}
	if ( saveAllImages ) {
		histogramImage.save("./CPU/histogram.bmp");
	}

	// Contrast enhancement
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);

	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./CPU/contrast.bmp");
	}

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);

	if ( displayImages ) {
		smoothImage.display("Smooth Image");
	}

	if ( saveAllImages ) {
		smoothImage.save("./CPU/smooth.bmp");
	}

	AppTime.stop();
	
	totalTimeCpu=AppTime.getElapsed();

	printf("\n\n");	

	/* GPU Implementation */ 

	AppTimeGPU.start();
	// Convert the input image to grayscale
	CImg< unsigned char > grayImageGPU = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	rgb2grayCuda(inputImage.data(), grayImageGPU.data(), inputImage.width(), inputImage.height());

	if ( displayImages ) {
		grayImageGPU.display("Grayscale Image GPU");
	}
	if ( saveAllImages ) {
		grayImageGPU.save("./GPU/grayscale.bmp");
	}

	// Compute 1D histogram
	CImg< unsigned char > histogramImageGPU = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogramGPU = new unsigned int [HISTOGRAM_SIZE];

	histogram1DCuda(grayImageGPU.data(), histogramImageGPU.data(), grayImageGPU.width(), grayImageGPU.height(), histogramGPU, HISTOGRAM_SIZE, BAR_WIDTH);

	if ( displayImages ) {
		histogramImageGPU.display("Histogram GPU");
	}
	if ( saveAllImages ) {
		histogramImageGPU.save("./GPU/histogram.bmp");
	}

	// Contrast enhancement
	contrast1DCuda(grayImageGPU.data(), grayImageGPU.width(), grayImageGPU.height(), histogramGPU, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);

	if ( displayImages ) {
		grayImageGPU.display("Contrast Enhanced Image GPU");
	}
	if ( saveAllImages ) {
		grayImageGPU.save("./GPU/contrast.bmp");
	}

	delete [] histogramGPU;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImageGPU = CImg< unsigned char >(grayImageGPU.width(), grayImageGPU.height(), 1, 1);
	triangularSmoothCuda(grayImageGPU.data(), smoothImageGPU.data(), grayImageGPU.width(), grayImageGPU.height(), filter);
	

	if ( displayImages ) {
		smoothImageGPU.display("Smooth Image GPU");
	}

	if ( saveAllImages ) {
		smoothImageGPU.save("./GPU/smooth.bmp");
	}


	AppTimeGPU.stop();
        totalTimeGpu = AppTimeGPU.getElapsed();


	//Calculate speed-up for kernels and entire app
         cout << fixed << setprecision(3);
	cout<<endl<<endl<<"Evalulating Application"<<endl<<endl;
	for(int i=0; i<4; i++)
	{
         kernelSpeedUp [i]= kernelCpuTime[i]/kernelGpuTime[i] ;
	}
	
       	printf("Speed up RGB2GRAY: x%f \n",kernelSpeedUp[0] );
       	printf("Speed up Histogram: x%f \n",kernelSpeedUp[1] );
       	printf("Speed up Contrast: x%f \n",kernelSpeedUp[2] );
       	printf("Speed up Smooth: x%f \n",kernelSpeedUp[3] );

       	overallSpeedUp = totalTimeCpu/totalTimeGpu ;
       	printf("CPU time: %f \t GPU Time: %f \n", totalTimeCpu,totalTimeGpu );
       	printf("Overall Speed up: x%f \n",overallSpeedUp );

	return 0;
}



