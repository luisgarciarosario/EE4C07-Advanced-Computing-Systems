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


#define BLOCK_MAX_THREADSIZE 512

//extern AppProfiler appProf;


extern double kernelCpuTime [4];
extern double kernelGpuTime [4];
extern double kernelSpeedUp [4];

//application profiling 
extern double AppCpuTime;
extern double AppGpuTime;
extern double AppSpeedUp;


static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}



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
	kernelGpuTime[0]= kernelTime.getElapsed(); 

}

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
	kernelCpuTime[0]= kernelTime.getElapsed(); 
}

__global__ void histogram1DCudaKernel(int ImageSize, unsigned int *device_histogram, unsigned char *d_grayImage)
{
	// set the pointer to every element in d_grayImage
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < ImageSize)
        {
                unsigned char Item = d_grayImage[index];
                //use atomic operation to solve problem of memory conflict
                atomicAdd (&(device_histogram[Item]), 1);
        }
}



void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height,
                                 unsigned int *histogram, const unsigned int HISTOGRAM_SIZE,
                                 const unsigned int BAR_WIDTH)
{
	// set the number of threads in a single block
        dim3 threadBlockSize(BLOCK_MAX_THREADSIZE);
        unsigned int max = 0;
        int ImageSize = width * height;
        NSTimer kernelTime = NSTimer("kernelTime", false, false);

        memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));
        // copy histogram to device_histogram
        unsigned int* device_histogram = NULL;
        checkCudaCall(cudaMalloc((void **) &device_histogram, HISTOGRAM_SIZE* sizeof(unsigned int)));
        checkCudaCall(cudaMemcpy(device_histogram, histogram, HISTOGRAM_SIZE* sizeof(unsigned int), cudaMemcpyHostToDevice));
        // copy grayImage to d_grayImage
        unsigned char* d_grayImage = NULL;
        checkCudaCall(cudaMalloc((void **) &d_grayImage, ImageSize));
        checkCudaCall(cudaMemcpy(d_grayImage, grayImage, ImageSize, cudaMemcpyHostToDevice));

        kernelTime.start();
        // set the number of blocks
        dim3 BlockNum(width*height/threadBlockSize.x+1);
        histogram1DCudaKernel<<<BlockNum, threadBlockSize>>>(ImageSize, device_histogram, d_grayImage);
        cudaDeviceSynchronize();
        kernelTime.stop();

        checkCudaCall(cudaMemcpy(histogram, device_histogram, HISTOGRAM_SIZE* sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaCall(cudaFree(device_histogram));
	// find the largest number in histogram
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
	kernelGpuTime[1]= kernelTime.getElapsed(); 

}

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
	kernelCpuTime[1]= kernelTime.getElapsed(); 
}


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

       	int grayImageSize= width * height;
       
       	// Allocate device memory for grayImage
       	unsigned char *d_grayImage;
        checkCudaCall(cudaMalloc((void **)&d_grayImage,grayImageSize));
       	
   	// Copy host memory to device 
         checkCudaCall(cudaMemcpy(d_grayImage,grayImage,grayImageSize,cudaMemcpyHostToDevice)); 
	
        // Setup execution parameters 
    	dim3 threads(BLOCK_MAX_THREADSIZE);
    	dim3 grid(grayImageSize/threads.x);

	// Kernel launch
	kernelTime.start();
	contrast1DKernel<<<grid,threads>>>(d_grayImage,width,height,min,max,diff,grayImageSize); 
    	cudaDeviceSynchronize();
	kernelTime.stop();

        // Copy result from device to host 
        checkCudaCall(cudaMemcpy(grayImage,d_grayImage,grayImageSize,cudaMemcpyDeviceToHost));
	
	cout << fixed << setprecision(6);
	cout << "contrast1DCUDA (gpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	kernelGpuTime[2]= kernelTime.getElapsed(); 
       
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
	kernelCpuTime[2]= kernelTime.getElapsed(); 
}

/////////////////////////////////////
/*
 * step 1. parallelize the triagularSmooth function by just adding the keyword __global__ in front of it
 * step 2. allocate the memory on the GPU and move the data over for the function to execute on
 * step 3. modify the function call in order to enable it to launch on the GPU
 */ 
 
 // 1. parallelize the triagularSmooth functionb by just adding the keyword __global__ in front of it
 // 
 // This function is called a Kernel: when called it is executed N times in parallel by N different
 // CUDA threads
 //
__global__ void triangularSmoothKernel(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
										const float *filter)
{
	// Each thread that executes the kernel is given a unique thread ID that is accessible within 
	// the kernel through the built-in threadIdx variable. 
	// Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz3nMa5Sr53
	
	// 'threadIdx' variable is a 3-component vector used to access the unique thread ID given to each 
	// thread that executes this Kernel
	// Each thread can be identified using a 1D, 2D or 3D thread index to form 1D, 2D or 3D block of threads

	// How many threads does a thread block on our GPU contains? 512 or 1024?
	
	// threadIdx:built-in variable used to access/identify a 1D, 2D or 3D thread index (and the thread-ID)
	// blockIdx: built-in variable used to access/identify a 1D, 2D or 3D block index
	// blockDim: built-in variable used to access a 1D, 2D or 3D thread dimension
	//
	// The total number of threads per block times the number of blocks
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // read-only variable i
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y; // read-only variable j
	
	if(i < width && j < height) /* ...??? */
	{
		unsigned int filterItem = 0;
		float filterSum = 0.0f;
		float smoothPix = 0.0f;

		for ( int fy = j - 2; fy < j + 3; fy++ ) 
		{
			for ( int fx = i - 2; fx < i + 3; fx++ ) 
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
		smoothImage[(j * width) + i] = static_cast< unsigned char >(smoothPix);
	}
}

void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
							const float *filter)
{
	//cudaError_t error;
	
	//.....
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	//int threadBlockSize = 512;
	
	// 2a. Allocate the memory on the GPU
	unsigned char *d_grayImage;
	unsigned char *d_smoothImage;
	float *d_filter;
	checkCudaCall(cudaMalloc((void **)&d_grayImage, width * height));
	checkCudaCall(cudaMalloc((void **)&d_smoothImage, width * height));
	checkCudaCall(cudaMalloc((void **)&d_filter, sizeof(filter)/sizeof(const float)));
	
	// 2b. Move data over (host memory to device memory) for the function to execute
	// cudaMemcpy(void *dst, void *src, size_t nbytes, enum cudaMemcpyKind direction); 
	checkCudaCall(cudaMemcpy((void *)d_grayImage, (void *)grayImage, (cudaMemcpyKind)width*height, cudaMemcpyHostToDevice)); 
	checkCudaCall(cudaMemcpy((void *)d_smoothImage, (void *)smoothImage, (cudaMemcpyKind)width*height, cudaMemcpyHostToDevice));
	checkCudaCall(cudaMemcpy(d_filter, filter, (cudaMemcpyKind)sizeof(filter)/sizeof(const float), cudaMemcpyHostToDevice));	
	
	// 3. Modify the function call in order to enable it to launch on the GPU
	//dim3 threads(512);
	//dim3 grid((width * height)/threads.x);
	//
	// Kernel invocation with one block of width * height * 1 threads
	dim3 threadsPerBlock(16, 16);
	dim3 numberOfBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);
	
	kernelTime.start();
	//Kernel invocation with N threads that executes it
	// Kernel
	//triangularSmoothKernel<<<grid, threads>>>(d_grayImage, d_smoothImage, width, height, d_filter); 
	triangularSmoothKernel<<<numberOfBlocks, threadsPerBlock>>>(d_grayImage, d_smoothImage, width, height, d_filter); 
	cudaDeviceSynchronize();	
	// /Kernel
	kernelTime.stop();
	
	// 4. Move data back over (device memory to host memory)
    checkCudaCall(cudaMemcpy((void *)grayImage, (void *)d_grayImage, (cudaMemcpyKind)width*height, cudaMemcpyDeviceToHost));
	checkCudaCall(cudaMemcpy((void *)smoothImage, (void *)d_smoothImage, (cudaMemcpyKind)width*height, cudaMemcpyDeviceToHost));
	
	cout << fixed << setprecision(6);
	cout << "triangularSmooth (gpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
	
	kernelGpuTime[3]= kernelTime.getElapsed(); 
	// Free up device memory
	cudaFree(d_grayImage); 
	cudaFree(d_smoothImage); 
	cudaFree(d_filter); 
}

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
	kernelCpuTime[3]= kernelTime.getElapsed(); 
}

