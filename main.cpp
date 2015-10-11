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

//keep track of just parallelizable part
double kernelCpuTime [4]={0,0,0,0}; //executime time of on cpu
double kernelGpuTime [4]={0,0,0,0}; //execution time on gpu, excluding communication
double kernelSpeedUp [4]={0,0,0,0}; //speedup of parllelizable part, with out communication



//keep track of execution time of the entire fuction call
double T_old [4]={0,0,0,0}; //executime time of on cpu
double T_new [4]={0,0,0,0}; //execution time on gpu, including communication
double funcOverallSpeedUp [4]={0,0,0,0}; //speedup of parllelizable part, with out communication


//application profiling 
double totalTimeCpu;
double totalTimeGpu;
double overallSpeedUp;

//keep track of setup and communication time of gpu
double time_total_comm[4]={0,0,0,0};


//fraction of parallelism for each function
double f [4]={0,0,0,0};

//keep track of setup and communication time of gpu
//double time_setup_comm[4]={0,0,0,0};
double time_setup_comm[4];

//measure effective bandwidth and gflops 
double bw_effective [4] = {0,0,0,0};
double gflops_effective [4] = {0,0,0,0};


int main(int argc, char *argv[])
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

    	NSTimer timer_totalTimeCpu = NSTimer("timer_totalTimeCpu", false, false);
    	NSTimer timer_totalTimeGpu = NSTimer("timer_totalTimeGpu", false, false);

	//timer to calculate fraction of parallelism
    	NSTimer timerA = NSTimer("f", false, false);


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

	timer_totalTimeCpu.start();
	// Convert the input image to grayscale
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

        timerA.start();
	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
	timerA.stop();
	
	T_old[0]=timerA.getElapsed(); 
	//f[0]=kernelCpuTime[0]/timerA.getElapsed();

	if ( displayImages ) {
		grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./CPU/grayscale.bmp");
	}
	

	// Compute 1D histogram
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];

	timerA.start();
	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);
	timerA.stop();

	T_old[1]=timerA.getElapsed(); 
	
	if ( displayImages ) {
		histogramImage.display("Histogram");
	}
	if ( saveAllImages ) {
		histogramImage.save("./CPU/histogram.bmp");
	}
	

	// Contrast enhancement
	timerA.start();
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);
	timerA.stop();

	T_old[2]=timerA.getElapsed(); 

	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./CPU/contrast.bmp");
	}

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	timerA.start();
	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
	timerA.stop();

	T_old[3]=timerA.getElapsed(); 

	if ( displayImages ) {
		smoothImage.display("Smooth Image");
	}

	if ( saveAllImages ) {
		smoothImage.save("./CPU/smooth.bmp");
	}

	timer_totalTimeCpu.stop();
	totalTimeCpu=timer_totalTimeCpu.getElapsed();

	//calculate the fraction of parallelism 
	for(int i=0; i<4; i++)
	{ 
		f[i]=kernelCpuTime[i]/T_old[i];
	} 


	printf("\n");	

	
	/********************
	* gpu implementation 
	*********************/  


	timer_totalTimeGpu.start();

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


	timer_totalTimeGpu.stop();
        totalTimeGpu = timer_totalTimeGpu.getElapsed();


	//Calculate speed-up for kernels and entire app
         cout << fixed << setprecision(3);
	cout<<endl<<endl<<"Evaluating Application"<<endl<<endl;
	for(int i=0; i<4; i++)
	{
         	kernelSpeedUp [i]= kernelCpuTime[i]/kernelGpuTime[i] ;
		funcOverallSpeedUp[i] = T_old[i]/T_new[i]; 
	}
	
       	//calculate overall speed up of application
	overallSpeedUp = totalTimeCpu/totalTimeGpu ;

	double bw_peak= 86.4;

	cout << fixed << setprecision(6);


	cout<<"---------------------------------------------------------------------------------------"<<endl;
	cout<< "Kernel Analysis (Speed-up)"<<endl; 	
	cout<<"------------------------"<<endl;
	cout<<"Function"<<right<<setw(13)<<"f\%"<<right<<setw(20)<<"Upperbound Sp"<<right<<setw(42)<<"Function overall speedup (comm + kernel)"<<right<<setw(20)<<"Kernel Speedup"<<endl;
       	cout<<"RGB2GRAY"<<right<<setw(13)<<(int)(f[0]*100)<<right<<setw(17)<<1/(1-f[0])<<right<<setw(14)<< funcOverallSpeedUp[0]<<right<<setw(44)<<kernelSpeedUp[0]<<endl;
       	cout<<"Histogram"<<right<<setw(12)<<(int)(f[1]*100)<<right<<setw(17)<<1/(1-f[1])<<right<<setw(14)<< funcOverallSpeedUp[1]<<right<<setw(44)<<kernelSpeedUp[1]<<endl;
       	cout<<"Contrast"<<right<<setw(13)<<(int)(f[2]*100)<<right<<setw(17)<<1/(1-f[2])<<right<<setw(14)<< funcOverallSpeedUp[2]<<right<<setw(44)<<kernelSpeedUp[2]<<endl;
       	cout<<"Smooth  "<<right<<setw(13)<<(int)(f[3]*100)<<right<<setw(17)<<1/(1-f[3])<<right<<setw(14)<< funcOverallSpeedUp[3]<<right<<setw(44)<<kernelSpeedUp[3]<<endl;
	cout<<"---------------------------------------------------------------------------------------"<<endl;
	cout<< "Kernel GPU Analysis ( Communication + kernel)"<<endl; 	
	cout<<"------------------------"<<endl;
	cout<<"Function"<<right<<setw(20)<<"Total time (sec)"<<right<<setw(20)<<"Communication (sec)"<<"\t %"<<"\t"<<" Kernel (sec)"<<"\t"<<"% "<<endl;
       	cout<<"RGB2GRAY"<<right<<setw(14)<<T_new[0]<<right<<setw(20)<<time_total_comm[0]<<right<<setw(20)<<time_total_comm[0]/T_new[0]*100 <<right<<setw(13)<<   kernelGpuTime[0]<<right<<setw(13)<<kernelGpuTime[0]/T_new[0]*100 <<endl;
       	cout<<"Histogram"<<right<<setw(13)<<T_new[1]<<right<<setw(20)<<time_total_comm[1]<<right<<setw(20)<<time_total_comm[1]/T_new[1]*100 <<right<<setw(13)<< kernelGpuTime[1]<<right<<setw(13)<<kernelGpuTime[1]/T_new[1]*100 <<endl;
       	cout<<"Contrast"<<right<<setw(14)<<T_new[2]<<right<<setw(20)<<time_total_comm[2]<<right<<setw(20)<<time_total_comm[2]/T_new[2]*100 <<right<<setw(13)<< kernelGpuTime[2]<<right<<setw(13)<<kernelGpuTime[2]/T_new[2]*100 <<endl;
       	cout<<"Smooth"<<right<<setw(15)<<T_new[3]<<right<<setw(20)<<time_total_comm[3]<<right<<setw(20)<<time_total_comm[3]/T_new[3]*100 <<right<<setw(13)<< kernelGpuTime[3]<<right<<setw(13)<<kernelGpuTime[3]/T_new[3]*100 <<endl;
		
	cout<<"---------------------------------------------------------------------------------------"<<endl;
	cout<< "Bandwidth Analysis"<<endl; 	
	cout<<"------------------------"<<endl;
	cout<<"Function"<<right<<setw(24)<<"Effective Bandwidth (GB/s)"<<right<<setw(20)<<"% of Peak"<<endl;
	cout<<"RGB2GRAY"<<right<<setw(24)<<bw_effective[0]<<right<<setw(20)<<bw_effective[0]/bw_peak * 100 <<endl;
	cout<<"Histogram"<<right<<setw(24)<<bw_effective[1]<<right<<setw(20)<<bw_effective[1]/bw_peak * 100 <<endl;
	cout<<"Contrast"<<right<<setw(24)<<bw_effective[2]<<right<<setw(20)<<bw_effective[2]/bw_peak * 100 <<endl;
	cout<<"Smooth"<<right<<setw(24)<<bw_effective[3]<<right<<setw(20)<<bw_effective[3]/bw_peak * 100 <<endl;
	cout<<"---------------------------------------------------------------------------------------"<<endl;
	cout<< "Application Analysis"<<endl; 	
	cout<<"------------------------"<<endl;
 	cout<<"Time (cpu): "<<totalTimeCpu<<" sec \t"<<"Time (gpu): "<<totalTimeGpu<<" sec \t"<<"Overall Speed up: x"<<overallSpeedUp<<endl;
	cout<<"---------------------------------------------------------------------------------------"<<endl;


	return 0;
}



