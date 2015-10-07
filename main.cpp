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

//fraction of parallelism for each function
double f [4];


int main(int argc, char *argv[])
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

    	NSTimer AppTime = NSTimer("AppTime", false, false);
    	NSTimer AppTimeGPU = NSTimer("AppTimeGPU", false, false);

	//timer to calculate fraction of parallelism
    	NSTimer f_timer = NSTimer("f", false, false);


	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( displayImages ) {
		inputImage.display("Input Image");
	}
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	/********************
	* cpu implementation 
	*********************/  

	AppTime.start();
	// Convert the input image to grayscale
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

        f_timer.start();
	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
	f_timer.stop();

	//f[0]=f_timer.getElapsed()/kernelCpuTime[0];
	f[0]=kernelCpuTime[0]/f_timer.getElapsed();

	if ( displayImages ) {
		grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./CPU/grayscale.bmp");
	}
	

	// Compute 1D histogram
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];

	f_timer.start();
	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);
	f_timer.stop();

	//f[1]=f_timer.getElapsed()/kernelCpuTime[1];
	f[1]=kernelCpuTime[1]/f_timer.getElapsed();

	if ( displayImages ) {
		histogramImage.display("Histogram");
	}
	if ( saveAllImages ) {
		histogramImage.save("./CPU/histogram.bmp");
	}
	

	// Contrast enhancement
	f_timer.start();
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);
	f_timer.stop();

	//f[2]=f_timer.getElapsed()/kernelCpuTime[2];
	f[2]=kernelCpuTime[2]/f_timer.getElapsed();

	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./CPU/contrast.bmp");
	}

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	f_timer.start();
	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
	f_timer.stop();

	//f[3]=f_timer.getElapsed()/kernelCpuTime[3];
	f[3]=kernelCpuTime[3]/f_timer.getElapsed();

	if ( displayImages ) {
		smoothImage.display("Smooth Image");
	}

	if ( saveAllImages ) {
		smoothImage.save("./CPU/smooth.bmp");
	}

	AppTime.stop();
	
	totalTimeCpu=AppTime.getElapsed();

	printf("\n");	

	
	/********************
	* gpu implementation 
	*********************/  
	

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
	
       /*
	printf("Speed up RGB2GRAY: x%f \n",kernelSpeedUp[0] );
       	printf("Speed up Histogram: x%f \n",kernelSpeedUp[1] );
       	printf("Speed up Contrast: x%f \n",kernelSpeedUp[2] );
       	printf("Speed up Smooth: x%f \n",kernelSpeedUp[3] );

	printf("Function \t f \t Speedup\n");
       	printf("RGB2GRAY: %d\t  x%f \n",(int)(f[0]*100),kernelSpeedUp[0] );
       	printf("Histogram: %d \t  x%f \n",(int)(f[1]*100),kernelSpeedUp[1] );
       	printf("Contrast: %d\t x%f \n",(int)(f[2]*100),kernelSpeedUp[2] );
       	printf("Smooth: %d \t  x%f \n",(int)(f[3]*100),kernelSpeedUp[3] );

	*/

	cout<<"Function"<<"\t"<<"f\%"<<"\t"<<"Speedup"<<endl;
       	cout<<"RGB2GRAY"<<"\t"<<(int)(f[0]*100)<<"\t"<<kernelSpeedUp[0]<<endl;
       	cout<<"Histogram"<<"\t"<<(int)(f[1]*100)<<"\t"<<kernelSpeedUp[1]<<endl;
       	cout<<"Contrast"<<"\t"<<(int)(f[2]*100)<<"\t"<<kernelSpeedUp[2]<<endl;
       	cout<<"Smooth  "<<"\t"<<(int)(f[3]*100)<<"\t"<<kernelSpeedUp[3]<<endl<<endl;

       	overallSpeedUp = totalTimeCpu/totalTimeGpu ;
    
 //	printf("CPU time: %f \t GPU Time: %f \n", totalTimeCpu,totalTimeGpu );
   //    	printf("Overall Speed up: x%f \n",overallSpeedUp );

 	cout<<"Time (cpu): "<<totalTimeCpu<<" sec \t"<<"Time (gpu): "<<totalTimeGpu<<" sec \t"<<"Overall Speed up: x"<<overallSpeedUp<<endl;

	return 0;
}



