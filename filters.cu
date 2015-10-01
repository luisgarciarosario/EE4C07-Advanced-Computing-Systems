#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/* Utility function/macro, used to do error checking.
   Use this function/macro like this:
   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
   And to check the result of a kernel invocation:
   checkCudaCall(cudaGetLastError());
*/
/*
#define checkCudaCall(result) {                                     \
    if (result != cudaSuccess){                                     \
        cerr << "cuda error: " << cudaGetErrorString(result);       \
        cerr << " in " << __FILE__ << " at line "<< __LINE__<<endl; \
        exit(1);                                                    \
    }                                                               \
}
*/
///*


static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}
//*/



__global__ void rgb2grayCudaKernel(unsigned char *d_inputImage, unsigned char *d_grayImage, int ImageSize)
{
        unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
        
	if(index < ImageSize)

           {
                      float grayPix = 0.0f;
                      float r = static_cast< float >(d_inputImage[index]);
                      float g = static_cast< float >(d_inputImage[ImageSize + index]);
                      float b = static_cast< float >(d_inputImage[(2 * ImageSize) + index]);
                      grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);
                      d_grayImage[index] = static_cast< unsigned char >(grayPix);
        }
   }



void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height)
{

        NSTimer kernelTime = NSTimer("kernelTime", false, false);
        memset(reinterpret_cast< void * >(grayImage), 0, width * height * sizeof(unsigned char));

        unsigned char* d_grayImage= NULL;
        unsigned char* d_inputImage= NULL;
        int ImageSize = width * height;
        checkCudaCall(cudaMalloc( (void **) &d_inputImage, (3*ImageSize) ));
        checkCudaCall(cudaMalloc((void **) &d_grayImage, ImageSize));
        checkCudaCall(cudaMemcpy(d_grayImage, grayImage, ImageSize, cudaMemcpyHostToDevice));
        checkCudaCall(cudaMemcpy(d_inputImage, inputImage, 3*ImageSize, cudaMemcpyHostToDevice));

        kernelTime.start();
        dim3 dimBlock(512);
        dim3 dimGrid(ImageSize/(int)dimBlock.x);
        rgb2grayCudaKernel<<<dimGrid, dimBlock>>>(d_inputImage, d_grayImage, ImageSize);
        cudaDeviceSynchronize();
        kernelTime.stop();

        checkCudaCall(cudaMemcpy(grayImage, d_grayImage, ImageSize, cudaMemcpyDeviceToHost));
        checkCudaCall(cudaFree(d_grayImage));
        checkCudaCall(cudaFree(d_inputImage));


        cout << fixed << setprecision(6);
        cout << "rgb2gray (gpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/*
void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			float grayPix = 0.0f;
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

			grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
		}
	}
	// /Kernel
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "rgb2gray (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}
*/
/////////////////////////////////////

__global__ void histogram1DCudaKernel(int ImageSize, unsigned int *device_histogram, unsigned char *d_grayImage)
{
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < ImageSize)
        {
                unsigned char Item = d_grayImage[index];
                atomicAdd (&(device_histogram[Item]), 1);
        }
}



void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height,
                                 unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,
                                 const unsigned int BAR_WIDTH)
{
        dim3 threadBlockSize(512);
        unsigned int max = 0;
        int ImageSize = width * height;
        NSTimer kernelTime = NSTimer("kernelTime", false, false);

        memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));
        unsigned int* device_histogram = NULL;
        checkCudaCall(cudaMalloc((void **) &device_histogram, HISTOGRAM_SIZE* sizeof(unsigned int)));
        checkCudaCall(cudaMemcpy(device_histogram, histogram, HISTOGRAM_SIZE* sizeof(unsigned int), cudaMemcpyHostToDevice));
        unsigned char* d_grayImage = NULL;
        checkCudaCall(cudaMalloc((void **) &d_grayImage, ImageSize));
        checkCudaCall(cudaMemcpy(d_grayImage, grayImage, ImageSize, cudaMemcpyHostToDevice));

        kernelTime.start();
        dim3 BlockNum(width*height/threadBlockSize.x+1);
        histogram1DCudaKernel<<<BlockNum, threadBlockSize>>>(ImageSize, device_histogram, d_grayImage);
        cudaDeviceSynchronize();
        kernelTime.stop();

        checkCudaCall(cudaMemcpy(histogram, device_histogram, HISTOGRAM_SIZE* sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaCall(cudaFree(device_histogram));

	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ )
	{
		if ( histogram[i] > max )
		{
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH )
	{
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ )
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ )
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ )
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ )
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	cout << fixed << setprecision(6);
	cout << "histogram1D (gpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/*
void histogram1D(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height, 
				 unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
				 const unsigned int BAR_WIDTH) 
{
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			histogram[static_cast< unsigned int >(grayImage[(y * width) + x])] += 1;
		}
	}
	// /Kernel
	kernelTime.stop();

	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) 
	{
		if ( histogram[i] > max ) 
		{
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) 
	{
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}
	
	cout << fixed << setprecision(6);
	cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}
*/
/////////////////////////////////////
__global__ void contrast1DKernel(unsigned char *grayImage, const int width, const int height,int min, int max, int diff, int grayImageSize) 
{

  unsigned int index  = blockIdx.x * blockDim.x + threadIdx.x;

 
  //ensure we dont use more threads than image size 
  if(index < grayImageSize)
   {

   
	unsigned char pixel = grayImage[index];

         if ( pixel < min )
        {
        	pixel = 0;
        }
        else if ( pixel > max )
        {
        	pixel = 255;
        }
        else
        {
        	pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
        }
                      
        grayImage[index] =  pixel;
   
   } 
  

}

void contrast1DCuda(unsigned char *grayImage, const int width, const int height, 
				unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
				const unsigned int CONTRAST_THRESHOLD) 
{


      cudaError_t error;

	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i--;
	}
	unsigned int max = i;
	float diff = max - min;


	int threadBlockSize = 512;
       	int grayImageSize= width * height;
       

       	// Allocate device memory for grayImage
       	unsigned char *d_grayImage;
        checkCudaCall(cudaMalloc((void **)&d_grayImage,grayImageSize));
       	
   	// Copy host memory to device 
         checkCudaCall(cudaMemcpy(d_grayImage,grayImage,grayImageSize,cudaMemcpyHostToDevice)); 
	
        // Setup execution parameters 
    	dim3 threads(512);
    	dim3 grid(grayImageSize/threads.x);


	kernelTime.start();
	// Kernel launch
	contrast1DKernel<<<grid,threads>>>(d_grayImage,width,height,min,max,diff,grayImageSize); 
    	cudaDeviceSynchronize();
	kernelTime.stop();


        // Copy result from device to host 
        checkCudaCall(cudaMemcpy(grayImage,d_grayImage,grayImageSize,cudaMemcpyDeviceToHost));
	
	cout << fixed << setprecision(6);
	cout << "contrast1DCUDA (gpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
       
        // clean device memory 
        cudaFree(d_grayImage); 

}

void contrast1D(unsigned char *grayImage, const int width, const int height, 
				unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
				const unsigned int CONTRAST_THRESHOLD) 
{
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for (int x = 0; x < width; x++ ) 
		{
			unsigned char pixel = grayImage[(y * width) + x];

			if ( pixel < min ) 
			{
				pixel = 0;
			}
			else if ( pixel > max ) 
			{
				pixel = 255;
			}
			else 
			{
				pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
			}
			
			grayImage[(y * width) + x] = pixel;
		}
	}
	// /Kernel
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "contrast1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/////////////////////////////////////
/*
__global__ void triangularSmoothKernel
{
}
*/

/*
void triangularSmoothCuda
{
}
*/

void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
					  const float *filter) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			unsigned int filterItem = 0;
			float filterSum = 0.0f;
			float smoothPix = 0.0f;

			for ( int fy = y - 2; fy < y + 3; fy++ ) 
			{
				for ( int fx = x - 2; fx < x + 3; fx++ ) 
				{
					if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) 
					{
						filterItem++;
						continue;
					}

					smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
					filterSum += filter[filterItem];
					filterItem++;
				}
			}

			smoothPix /= filterSum;
			smoothImage[(y * width) + x] = static_cast< unsigned char >(smoothPix);
		}
	}
	// /Kernel
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "triangularSmooth (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}

