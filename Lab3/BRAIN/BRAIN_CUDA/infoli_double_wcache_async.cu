/*
 *
 * Copyright (c) 2012, Neurasmus B.V., The Netherlands,
 * web: www.neurasmus.com email: info@neurasmus.com
 * Copyright (c) 2013, Computer Engineering lab, TU Delft, The Netherlands,
 * web: www.ce.ewi.tudelft.nl
 * Any use or reproduction in whole or in parts is prohibited
 * without the written consent of the copyright owner.
 *
 * All Rights Reserved.
 *
 * Implementation in C by Sebastian Isaza
 * Author: Sebastian Isaza
 * Created: 19-01-2012
 * Modified: 07-08-2012
 *
 * Implementation in CUDA by Du Nguyen Hoang Anh
 * Author: Du Nguyen Hoang Anh
 * Created: 19-04-2013
 * Modified: 09-11-2013
 * Description: Top source file of the Inferior Olive model, originally written
 * in Matlab by Jornt De Gruijl. It contains the implementation of all functions.
 * The main function allocates the necessary memory, initializes the system
 * state and runs the model calculations.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "infoli.h"
#include <cuda_runtime.h>

	typedef unsigned long long timestamp_t;

	static timestamp_t get_timestamp ()
	{
		struct timeval now;
		gettimeofday (&now, NULL);
		return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
	}
	
	//This is function declaration for device's side
	__global__ void neighbor_kernel(double *cellStatePtr, double *cellVDendPtr) ;
	__global__ void compute_kernel(double *cellStatePtr, double *iApp, double *cellVDendPtr) ;
	__device__ int dev_fetch(int j, int k) ;
	__device__ void dev_CompDend(double *cellCompParamsPtr);
	__device__ void dev_DendHCurr(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1);
	__device__ void dev_DendCaCurr(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1);
	__device__ void dev_DendKCurr(double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1);
	__device__ void dev_DendCal(double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1);
	__device__ void dev_DendCurrVolt(double chComps_iC, double *chComps_iApp, double *chComps_vDend, double *chComps_newVDend, double *chComps_vSoma, double *chComps_q, double *chComps_r, double *chComps_s, double *chComps_newI_CaH);
	__device__ double dev_IcNeighbors(double *neighVdend, double prevV_dend);
	__device__ void dev_CompSoma(double *cellCompParamsPtr);
	__device__ void dev_SomaCalcium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1, double *chPrms_newComp2);
	__device__ void dev_SomaSodium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1, double *chPrms_newComp2);
	__device__ void dev_SomaPotassium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1, double *chPrms_newComp2);
	__device__ void dev_SomaPotassiumX(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1);
	__device__ void dev_SomaCurrVolt(double *chComps_g_CaL, double *chComps_vDend, double *chComps_vSoma, double *chComps_newVSoma, double *chComps_vAxon, double *chComps_k, double *chComps_l, double *chComps_m, double *chComps_h, double *chComps_n, double *chComps_x_s);
	__device__ void dev_CompAxon(double *cellCompParamsPtr);
	__device__ void dev_AxonSodium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1, double *chPrms_newComp2);
	__device__ void dev_AxonPotassium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1);
	__device__ void dev_AxonCurrVolt(double *chComps_vSoma, double *chComps_vAxon, double *chComps_newVAxon, double *chComps_m_a, double *chComps_h_a, double *chComps_x_a);
	__device__ int g_i=0;
	texture<int2, 2, cudaReadModeElementType> t_cellVDendPtr;
	static __inline__ __device__ double fetch_double(texture<int2, 2> t, int x, int y);
	
	// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
    #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if( cudaSuccess != err) {
                    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }


int main(int argc, char *argv[]){
    char *inFileName;
    char *outFileName = "InferiorOlive_Output.txt";
    FILE *pInFile;
    FILE *pOutFile;
    char temp[100];//warning: this buffer may overflow
    int inputFromFile = 0;
    int debug=0, print=0;
    timestamp_t t0, t1, t2, t3, t4, t5, secs;
    cudaEvent_t start, stop;
    float time;
    int seedvar, i, b, s;
    cellState initState;
    int simTime, simSteps;
    //GPU implementation's variables:	
    double *dev_cellStatePtr;
    double *dev_iApp;
    double  cellStateInit[STATE_SIZE];
    double  *cellStatePtr;
    double  *iApp;
    double  *cellVDendPtr;
    double  *dev_cellVDendPtr;
    int devID = 0;
    cudaStream_t stream[5];
	for (int i = 0; i < 5; ++i)
    	cudaStreamCreate(&stream[i]);
		
    //Start program
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
    if (print)
    	printf("Inferior Olive Model (%d x %d cell mesh)\n", IO_NETWORK_DIM1, IO_NETWORK_DIM2);
		
    //Open output file
    pOutFile = fopen(outFileName,"w");
    if(pOutFile==NULL){
        printf("Error: Couldn't create %s\n", outFileName);
        exit(EXIT_FAILURE);
    }
    if (debug) {
        sprintf(temp, "#simSteps Time(ms) Input(Iapp) Output(V_axon)\n");
        fputs(temp, pOutFile);
    }
	
    //Process command line arguments
    if(argc == 1){
        inputFromFile = 0;
	if (debug)
            printf("Warning: No input file has been specified. A one-pulse input will be used.\n");
    }else if(argc == 2){
        inputFromFile = 1;
        inFileName = argv[1];//comment out for a hardcoded name
        pInFile = fopen(inFileName,"r");
        if(pInFile==NULL){
            printf("Error: Couldn't open %s\n", inFileName);
            exit(EXIT_FAILURE);
        }
    }else{
        printf("Error: Too many arguments.\nUsage: ./InferiorOlive <Iapp_input_file> or ./InferiorOlive\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors( cudaSetDevice(devID) );
    
    if (debug) {
    	printf("Malloc'ing memory...\n");
    	printf("cellStatePtr: %dB\n", IO_NETWORK_DIM1*IO_NETWORK_DIM2*PARAM_SIZE*sizeof(double));
    }
    cellStatePtr = (double*)malloc(IO_NETWORK_DIM1*IO_NETWORK_DIM2*PARAM_SIZE*sizeof(double));
    if(cellStatePtr==NULL){
        printf("Error: Couldn't malloc for cellStatePtr\n");
        exit(EXIT_FAILURE);
    }
    if (debug)
        printf("cellVDendPtr: %dB\n", IO_NETWORK_DIM1*IO_NETWORK_DIM2*sizeof(double));
    cellVDendPtr = (double*)malloc(IO_NETWORK_DIM1*IO_NETWORK_DIM2*sizeof(double));
    if(cellVDendPtr==NULL){
        printf("Error: Couldn't malloc for cellVDendPtr\n");
        exit(EXIT_FAILURE);
    }
    simTime = SIMTIME; // in miliseconds
    simSteps = ceil(simTime/DELTA);
    if (debug)
        printf("iApp: %dB\n", simSteps*sizeof(double));
    iApp = (double*)malloc(simSteps*sizeof(double));
    if(cellStatePtr==NULL){
        printf("Error: Couldn't malloc for iApp\n");
        exit(EXIT_FAILURE);
    }   
    //Intialize inputs
    for(i=0; i<simSteps; i++){
	  iApp[i] = 0;
	  if(i>20000-1 && i<20500-1){ iApp[i] = 6;} // start @ 1 because skipping initial values
    }

    //Write initial state values
    //Initial dendritic parameters
    initState.dend.V_dend = -60;
    initState.dend.Calcium_r = 0.0112788;// High-threshold calcium
    initState.dend.Potassium_s = 0.0049291;// Calcium-dependent potassium
    initState.dend.Hcurrent_q = 0.0337836;// H current
    initState.dend.Ca2Plus = 3.7152;// Calcium concentration
    initState.dend.I_CaH   = 0.5;// High-threshold calcium current
    //Initial somatic parameters
    initState.soma.g_CaL = 0.68; //default arbitrary value but it should be randomized per cell
    initState.soma.V_soma = -60;
    initState.soma.Sodium_m = 1.0127807;// Sodium (artificial)
    initState.soma.Sodium_h = 0.3596066;
    initState.soma.Potassium_n = 0.2369847;// Potassium (delayed rectifier)
    initState.soma.Potassium_p = 0.2369847;
    initState.soma.Potassium_x_s = 0.1;// Potassium (voltage-dependent)
    initState.soma.Calcium_k = 0.7423159;// Low-threshold calcium
    initState.soma.Calcium_l = 0.0321349;
    // Initial axonal parameters
    initState.axon.V_axon = -60;
    //sisaza: Sodium_m_a doesn't have a state, therefore this assignment doesn'thave any effect
    initState.axon.Sodium_m_a = 0.003596066;// Sodium (thalamocortical)
    initState.axon.Sodium_h_a = 0.9;
    initState.axon.Potassium_x_a = 0.2369847;// Potassium (transient)
    
    //Copy init sate to all cell states
    if (debug)
        printf("Initializing cell states.\n");
		
    cellStateInit[DEND_V]	= initState.dend.V_dend;  
    cellStateInit[DEND_H] 	= initState.dend.Hcurrent_q;
    cellStateInit[DEND_CAL] = initState.dend.Calcium_r;
    cellStateInit[DEND_P] 	= initState.dend.Potassium_s;
    cellStateInit[DEND_I] 	= initState.dend.I_CaH;
    cellStateInit[DEND_CA2] = initState.dend.Ca2Plus;
    cellStateInit[SOMA_G] 	= initState.soma.g_CaL;
    cellStateInit[SOMA_V] 	= initState.soma.V_soma;
    cellStateInit[SOMA_SM] 	= initState.soma.Sodium_m;
    cellStateInit[SOMA_SH] 	= initState.soma.Sodium_h;
    cellStateInit[SOMA_CK] 	= initState.soma.Calcium_k;
    cellStateInit[SOMA_CL] 	= initState.soma.Calcium_l;
    cellStateInit[SOMA_PN] 	= initState.soma.Potassium_n;
    cellStateInit[SOMA_PP] 	= initState.soma.Potassium_p;
    cellStateInit[SOMA_PXS] = initState.soma.Potassium_x_s;
    cellStateInit[AXON_V] 	= initState.axon.V_axon;
    cellStateInit[AXON_SM] 	= initState.axon.Sodium_m_a;
    cellStateInit[AXON_SH] 	= initState.axon.Sodium_h_a;
    cellStateInit[AXON_P] 	= initState.axon.Potassium_x_a;

    //Initialize g_CaL
    seedvar = 1;
    srand(seedvar++);   // use this for debugging, now there is difference
    cellStateInit[SOMA_G] = 0.68;
    for(i=0;i<IO_NETWORK_SIZE;i++){
		for(b=0;b<STATE_SIZE;b++){
	cellStatePtr[i*PARAM_SIZE + STATEADD + b] = cellStateInit[b];
		}
		cellVDendPtr[i] = initState.dend.V_dend;
		//printf("%d: VDend=%.8f\n", i, cellVDendPtr[i]);
    }
	// Uncomment the next two lines to assign different soma conductances to each cell.
    //cellStatePtr[0][j][k].soma.g_CaL = 0.6+(0.2*(rand()%100)/100);
    //cellStatePtr[1][j][k].soma.g_CaL = cellStatePtr[0][j][k].soma.g_CaL;
	
	
    if (debug)
        printf("Setting up device's parameters.\n");
    checkCudaErrors(cudaMalloc( (void**)&dev_iApp, simSteps*sizeof(double) ) );
    checkCudaErrors(cudaMalloc( (void**)&dev_cellStatePtr, IO_NETWORK_DIM1*IO_NETWORK_DIM2*PARAM_SIZE*sizeof(double) ) ); 
    checkCudaErrors(cudaMalloc( (void**)&dev_cellVDendPtr, IO_NETWORK_DIM1*IO_NETWORK_DIM2*sizeof(double) ) ); 
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int2>();
    dim3 blockDim(BLOCKSIZEX, BLOCKSIZEY);
    dim3 gridDim(IO_NETWORK_DIM1/blockDim.x, IO_NETWORK_DIM2/blockDim.y);
    cudaFuncSetCacheConfig(neighbor_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(compute_kernel, cudaFuncCachePreferL1);	

    if (debug)
        printf("Transferring data to device's side.\n");	
    t0 = get_timestamp();
    checkCudaErrors(cudaMemcpy( dev_iApp, iApp, simSteps*sizeof(double), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy( dev_cellStatePtr, cellStatePtr, IO_NETWORK_DIM1*IO_NETWORK_DIM2*PARAM_SIZE*sizeof(double), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy( dev_cellVDendPtr, cellVDendPtr, IO_NETWORK_DIM1*IO_NETWORK_DIM2*sizeof(double), cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaBindTexture2D(NULL, t_cellVDendPtr, dev_cellVDendPtr,  desc, IO_NETWORK_DIM1, IO_NETWORK_DIM2, IO_NETWORK_DIM1*sizeof(double)));
    t1 = get_timestamp();

	
    if (debug)
        printf("Computing ...\n");
    cudaEventRecord(start, 0);
    #pragma unroll 1
    for(i=0;i<simSteps;i++){
		if (debug) {
            sprintf(temp, "%d %.2f %.1f ", i+1, i*0.05, iApp[i]); // start @ 1 because skipping initial values
            fputs(temp, pOutFile);
		}
		s = i%5;
		t4 = get_timestamp();
		neighbor_kernel <<< gridDim, blockDim, 0, stream[s] >>>(dev_cellStatePtr, dev_cellVDendPtr);
		compute_kernel <<< gridDim, blockDim, 0, stream[s] >>>(dev_cellStatePtr, dev_iApp, dev_cellVDendPtr);
		t5 = get_timestamp();
		
		if (debug)
		    printf("Transferring results to host.\n");
		checkCudaErrors(cudaMemcpyAsync( cellStatePtr, dev_cellStatePtr, IO_NETWORK_DIM1*IO_NETWORK_DIM2*PARAM_SIZE*sizeof(double), cudaMemcpyDeviceToHost, stream[s]) );
		if (debug) {		
        	b=0;
        	for (b=0; b<IO_NETWORK_SIZE; b=b+50){
				sprintf(temp, "%.8f ", cellStatePtr[b*PARAM_SIZE + STATEADD + AXON_V]);
				fputs(temp, pOutFile);
	    		//printf("%d - V_AXON final state: %.8f \n", i, cellStatePtr[i*PARAM_SIZE + STATEADD + AXON_V]);
        	}		
            sprintf(temp, "\n ");
            fputs(temp, pOutFile);
		}
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    if (debug) {
        sprintf(temp, "\n");
        fputs(temp, pOutFile);
    }
    t2 = get_timestamp();
	
    //if (debug)
    	//printf("Transferring results to host.\n");
    //checkCudaErrors(cudaMemcpy( cellStatePtr, dev_cellStatePtr, IO_NETWORK_DIM1*IO_NETWORK_DIM2*PARAM_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );
    t3 = get_timestamp();
    
	if (debug) {
    	printf("BlockDim x=%d, y=%d, GridDim x=%d, y=%d \n", blockDim.x, blockDim.y, IO_NETWORK_DIM1/blockDim.x, IO_NETWORK_DIM2/blockDim.y);
        printf("%d ms of brain time in %d simulation steps\n", SIMTIME, 120000);
        secs = (t3 - t0);
        printf("%lld us real time \n", secs);
        cudaEventElapsedTime(&time, start, stop);
        secs = (t2 - t1);
        printf(" %f us kernel time: \n", time*1000);
        printf(" %lld us kernel time \n", secs);
        secs = (t5 - t4);
        printf("   %lld us compute time per timestep to device time \n", secs);
        secs = (t1 - t0);
        printf(" %lld us xfer to device time \n", secs);
        secs = (t3 - t2);
        printf(" %lld us xfer to host time cellState \n", secs);
    }
    if (print) {
        printf("%d ms of brain time in %d simulation steps\n", SIMTIME, 120000);
        secs = (t3 - t0);
        printf("%lld us real time \n", secs);
    }

    //Free up memory and close files
    for (int i = 0; i < 5; ++i)
    	cudaStreamDestroy(stream[i]);
    cudaUnbindTexture(t_cellVDendPtr);
    free(cellStatePtr);
    free(cellVDendPtr);
    free(iApp);
    cudaFree(dev_cellStatePtr);
    cudaFree(dev_cellVDendPtr);
    cudaFree(dev_iApp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    fclose (pOutFile);
    if(inputFromFile){ fclose (pInFile);}

    return EXIT_SUCCESS;
}

__global__ void neighbor_kernel(double *cellStatePtr, double *cellVDendPtr) {

	//int d_simTime, d_simSteps, d_i, k, j, p, q, n = 0, e;
	int j, k, n, p, q;
	//double d_cellCompParams[SIZEOFCOMPPARAM];
	
	k = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;
	
	//Get neighbor V_dend
	n = 0;
	for(p=j-1;p<=j+1;p++){
	    for(q=k-1;q<=k+1;q++){
		//cellStatePtr[j*IO_NETWORK_DIM2*PARAM_SIZE + k*PARAM_SIZE + (n)] = fetch_double(t_cellVDendPtr, j, k);
		//if (((p!=j)||(q!=k)) && (p>=0)&&(q>=0) && (p<IO_NETWORK_DIM1)&&(q<IO_NETWORK_DIM2)){ 
		  cellStatePtr[dev_fetch(j,k) + (n++)] = fetch_double(t_cellVDendPtr, p, q);
		  //cellStatePtr[j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE + (n++)] = fetch_double(t_cellVDendPtr, p, q);
		//}	    
		if(p==j && q==k) n=n-1;
		//n=n+1;
	    }	
	}
	__syncthreads();
	return;
}

__global__ void compute_kernel(double *cellStatePtr, double *iApp, double *cellVDendPtr) {

	int j, k, e, d_i;
	double d_cellCompParams[LOCAL_PARAM_SIZE];
	
	k = blockIdx.x*blockDim.x + threadIdx.x;
	j = blockIdx.y*blockDim.y + threadIdx.y;
	
	//Compute one by one sim step
	    d_i=g_i;  
	    d_cellCompParams[0] = iApp[d_i];
	    #pragma unroll 8
	    for (e = 0; e < STATEADD; e++){
		d_cellCompParams[VNEIGHSTARTADD + e] = cellStatePtr[dev_fetch(j,k) + e];
		//d_cellCompParams[VNEIGHSTARTADD + e] = cellStatePtr[j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE + e];
	    }  
	    #pragma unroll 19
	    for (e = 0; e < STATE_SIZE; e++){
	      	d_cellCompParams[PREVSTATESTARTADD + e] = cellStatePtr[dev_fetch(j,k) + STATEADD + e];
		//d_cellCompParams[VNEIGHSTARTADD + e] = cellStatePtr[j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE + e];
		d_cellCompParams[NEXTSTATESTARTADD + e] = d_cellCompParams[PREVSTATESTARTADD + e];
	    }
	    //Compute one Cell...
	    dev_CompDend(d_cellCompParams);	
	    dev_CompSoma(d_cellCompParams);
	    dev_CompAxon(d_cellCompParams); 
	    //updates
	    __syncthreads();
	    #pragma unroll 19
	    for (e = 0; e < STATE_SIZE; e++){
		cellStatePtr[dev_fetch(j,k) + STATEADD + e] = d_cellCompParams[NEXTSTATESTARTADD + e];
		//cellStatePtr[j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE + STATEADD + e] = d_cellCompParams[NEXTSTATESTARTADD + e];
	    }
	    cellVDendPtr[j*IO_NETWORK_DIM1 + k] = d_cellCompParams[NEXTSTATESTARTADD + DEND_V];
	    if(j==0&&k==0)
		g_i=g_i+1;
	    __syncthreads();
	return;
}

__device__ int dev_fetch(int j, int k) {
	return (j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE);
}
__device__ void dev_CompDend(double *cellCompParamsPtr){

	double *chPrms_v;
	double *chPrms_prevComp1, *chPrms_prevComp2;
	double *chPrms_newComp1;// *chPrms_newComp2;
	double *chComps_iApp;
	double chComps_iC;
	double *chComps_vDend;
	double *chComps_vSoma;
	double *chComps_q, *chComps_r, *chComps_s;
	double *chComps_newVDend;
	double *chComps_newI_CaH;

    //printf("Dendrite ");

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]); //&cellCompParamsPtr->prevCellState->dend.V_dend;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_H]); //&cellCompParamsPtr->prevCellState->dend.Hcurrent_q;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_H]); //&cellCompParamsPtr->newCellState->dend.Hcurrent_q;
    //Compute
    dev_DendHCurr(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]);  //&cellCompParamsPtr->prevCellState->dend.V_dend;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_CAL]); //&cellCompParamsPtr->prevCellState->dend.Calcium_r;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_CAL]); //&cellCompParamsPtr->newCellState->dend.Calcium_r;
    //Compute
    dev_DendCaCurr(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

    //Prepare pointers to inputs/outputs
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_P]); //&cellCompParamsPtr->prevCellState->dend.Potassium_s;
    chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_CA2]); //&cellCompParamsPtr->prevCellState->dend.Ca2Plus;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_P]);  //&cellCompParamsPtr->newCellState->dend.Potassium_s;
    //Compute
    dev_DendKCurr(chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1);

    //Prepare pointers to inputs/outputs
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_CA2]); //&cellCompParamsPtr->prevCellState->dend.Ca2Plus;
    chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_I]); //&cellCompParamsPtr->prevCellState->dend.I_CaH;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_CA2]);  //&cellCompParamsPtr->newCellState->dend.Ca2Plus;
    //Compute
    dev_DendCal(chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1);

    chComps_iC = dev_IcNeighbors(cellCompParamsPtr, cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]);
		//IcNeighbors(cellCompParamsPtr->neighVdend, cellCompParamsPtr->prevCellState->dend.V_dend);
    chComps_iApp = &(cellCompParamsPtr[0]); //&cellCompParamsPtr->iAppIn;
    chComps_vDend = &(cellCompParamsPtr[PREVSTATESTARTADD]); //&cellCompParamsPtr->prevCellState->dend.V_dend;
    chComps_newVDend = &(cellCompParamsPtr[NEXTSTATESTARTADD]); //&cellCompParamsPtr->newCellState->dend.V_dend;
    chComps_vSoma = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma
    chComps_q = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_H]); // &cellCompParamsPtr->newCellState->dend.Hcurrent_q;
    chComps_r = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_CAL]); //&cellCompParamsPtr->newCellState->dend.Calcium_r;
    chComps_s = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_P]); //&cellCompParamsPtr->newCellState->dend.Potassium_s;
    chComps_newI_CaH = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_I]); //&cellCompParamsPtr->newCellState->dend.I_CaH;
    dev_DendCurrVolt(chComps_iC, chComps_iApp, chComps_vDend, chComps_newVDend, chComps_vSoma, chComps_q, chComps_r, chComps_s, chComps_newI_CaH);

    return;
}
__device__ void dev_DendHCurr(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1){

    double q_inf, tau_q, dq_dt, q_local;

    //Get inputs
    double prevV_dend = *chPrms_v; // *chPrms->v;
    double prevHcurrent_q = *chPrms_prevComp1;//*chPrms->prevComp1;

    // Update dendritic H current component
    q_inf = 1 /(1 + exp((prevV_dend + 80) / 4));
    tau_q = 1 /(exp(-0.086 * prevV_dend - 14.6) + exp(0.070 * prevV_dend - 1.87));
    dq_dt = (q_inf - prevHcurrent_q) / tau_q;
    q_local = DELTA * dq_dt + prevHcurrent_q;
    //Put result
    *chPrms_newComp1 = q_local;

    return;
}
__device__ void dev_DendCaCurr(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1){

    double alpha_r, beta_r, r_inf, tau_r, dr_dt, r_local;

    //Get inputs
    double prevV_dend = *chPrms_v; //*chPrms->v;
    double prevCalcium_r = *chPrms_prevComp1; //*chPrms->prevComp1;

    // Update dendritic high-threshold Ca current component
    alpha_r = 1.7 / (1 + exp( -(prevV_dend - 5) / 13.9));
    beta_r = 0.02 * (prevV_dend + 8.5) / (exp((prevV_dend + 8.5) / 5) - 1);
    r_inf = alpha_r / (alpha_r + beta_r);
    tau_r = 5 / (alpha_r + beta_r);
    dr_dt = (r_inf - prevCalcium_r) / tau_r;
    r_local = DELTA * dr_dt + prevCalcium_r;
    //Put result
    *chPrms_newComp1 = r_local; // *chPrms->newComp1

    return;
}

__device__ void dev_DendKCurr(double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1){

    double  alpha_s = 0.01, beta_s, s_inf, tau_s, ds_dt, s_local;

    //Get inputs
    double prevPotassium_s = *chPrms_prevComp1;//*chPrms->prevComp1;
    double prevCa2Plus = *chPrms_prevComp2; //*chPrms->prevComp2;

    // Update dendritic Ca-dependent K current component
    if ((0.00002*prevCa2Plus)<0.01)
    	alpha_s = (0.00002*prevCa2Plus);
    beta_s = 0.015;
    s_inf = alpha_s / (alpha_s + beta_s);
    tau_s = 1 / (alpha_s + beta_s);
    ds_dt = (s_inf - prevPotassium_s) / tau_s;
    s_local = DELTA * ds_dt + prevPotassium_s;
    //Put result
    *chPrms_newComp1 = s_local; //*chPrms->newComp1

    return;
}

//Consider merging DendCal into DendKCurr since DendCal's output doesn't go to DendCurrVolt but to DendKCurr
__device__ void dev_DendCal(double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1){

    double  dCa_dt, Ca2Plus_local;

    //Get inputs
    double prevCa2Plus = *chPrms_prevComp1; //*chPrms->prevComp1;
    double prevI_CaH = *chPrms_prevComp2; //*chPrms->prevComp2;

    // update Calcium concentration
    dCa_dt = -3 * prevI_CaH - 0.075 * prevCa2Plus;
    Ca2Plus_local = DELTA * dCa_dt + prevCa2Plus;
    //Put result
    *chPrms_newComp1 = Ca2Plus_local; //*chPrms->newComp1 //This state value is read in DendKCurr 

    return;
}

__device__ void dev_DendCurrVolt(double chComps_iC, double *chComps_iApp, double *chComps_vDend, double *chComps_newVDend, double *chComps_vSoma, double *chComps_q, double *chComps_r, double *chComps_s, double *chComps_newI_CaH){

    //Loca variables
    double I_sd, I_CaH, I_K_Ca, I_ld, I_h, dVd_dt;

    //Get inputs
    double I_c = chComps_iC; //chComps->iC;
    double I_app = *chComps_iApp; //*chComps->iApp;
    double prevV_dend = *chComps_vDend; //*chComps->vDend;
    double prevV_soma = *chComps_vSoma; //*chComps->vSoma;
    double q = *chComps_q; //*chComps->q;
    double r = *chComps_r; //*chComps->r;
    double s = *chComps_s; //*chComps->s;

    // DENDRITIC CURRENTS

    // Soma-dendrite interaction current I_sd
    I_sd   = (G_INT / (1 - P1)) * (prevV_dend - prevV_soma);
    // Inward high-threshold Ca current I_CaH
    I_CaH  =  G_CAH * r * r * (prevV_dend - V_CA);
    // Outward Ca-dependent K current I_K_Ca
    I_K_Ca =  G_K_CA * s * (prevV_dend - V_K);
    // Leakage current I_ld
    I_ld   =  G_LD * (prevV_dend - V_L);
    // Inward anomalous rectifier I_h
    I_h    =  G_H * q * (prevV_dend - V_H);

    dVd_dt = (-(I_CaH   + I_sd  + I_ld + I_K_Ca + I_c + I_h) + I_app) / C_M;

    //Put result (update V_dend)
    *chComps_newVDend = DELTA * dVd_dt + prevV_dend; //*chComps->newVDend
    *chComps_newI_CaH = I_CaH; //*chComps->newI_CaH //This is a state value read in DendCal
    return;
}

__device__ double dev_IcNeighbors(double *neighVdend, double prevV_dend){

    int i;
    double f, V, I_c;
    //printf("Ic[0]= %f\n", neighVdend[0]);

    I_c = 0;
    #pragma unroll 8
    for(i=0;i<8;i++){
        V = prevV_dend - neighVdend[VNEIGHSTARTADD + i]; //neighVdend[i];
        f = 0.8 * exp(-1*pow(V, 2)/100) + 0.2;    // SCHWEIGHOFER 2004 VERSION
        I_c = I_c + (CONDUCTANCE * f * V);
    }

    return I_c;
}


__device__ void dev_CompSoma(double *cellCompParamsPtr){

	double *chPrms_v;
	double *chPrms_prevComp1, *chPrms_prevComp2;
	double *chPrms_newComp1, *chPrms_newComp2;
	double *chComps_g_CaL;
	double *chComps_vSoma;
	double *chComps_vDend;
	double *chComps_vAxon;
	double *chComps_k, *chComps_l, *chComps_m, *chComps_h, *chComps_n, *chComps_x_s;
	double *chComps_newVSoma;

    // update somatic components
    // SCHWEIGHOFER:

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_CK]); //&cellCompParamsPtr->prevCellState->soma.Calcium_k;
    chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_CL]); //&cellCompParamsPtr->prevCellState->soma.Calcium_l;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CK]); //&cellCompParamsPtr->newCellState->soma.Calcium_k;
    chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CL]); //&cellCompParamsPtr->newCellState->soma.Calcium_l;
    //Compute
    dev_SomaCalcium(chPrms_v, chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1, chPrms_newComp2);

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_SM]); //&cellCompParamsPtr->prevCellState->soma.Sodium_m;
    chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_SH]); //&cellCompParamsPtr->prevCellState->soma.Sodium_h;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SM]); //&cellCompParamsPtr->newCellState->soma.Sodium_m;
    chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SH]); //&cellCompParamsPtr->newCellState->soma.Sodium_h;
    //Compute
    dev_SomaSodium(chPrms_v, chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1, chPrms_newComp2);

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_PN]); //&cellCompParamsPtr->prevCellState->soma.Potassium_n;
    chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_PP]); //&cellCompParamsPtr->prevCellState->soma.Potassium_p;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PN]); //&cellCompParamsPtr->newCellState->soma.Potassium_n;
    chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PP]); //&cellCompParamsPtr->newCellState->soma.Potassium_p;
    //Compute
    dev_SomaPotassium(chPrms_v, chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1, chPrms_newComp2);

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms_prevComp1 =&(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_PXS]); //&cellCompParamsPtr->prevCellState->soma.Potassium_x_s;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PXS]); //&cellCompParamsPtr->newCellState->soma.Potassium_x_s;
    //Compute
    dev_SomaPotassiumX(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

    chComps_g_CaL = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_G]); //&cellCompParamsPtr->prevCellState->soma.g_CaL;
    chComps_vDend = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]); //&cellCompParamsPtr->prevCellState->dend.V_dend;
    chComps_vSoma = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps_newVSoma = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->newCellState->soma.V_soma;
    chComps_vAxon = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]); //&cellCompParamsPtr->prevCellState->axon.V_axon;
    chComps_k = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CK]); //&cellCompParamsPtr->newCellState->soma.Calcium_k;
    chComps_l = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CL]); //&cellCompParamsPtr->newCellState->soma.Calcium_l;
    chComps_m = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SM]); //&cellCompParamsPtr->newCellState->soma.Sodium_m;
    chComps_h = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SH]); //&cellCompParamsPtr->newCellState->soma.Sodium_h;
    chComps_n = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PN]); //&cellCompParamsPtr->newCellState->soma.Potassium_n;
    chComps_x_s = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PXS]); // &cellCompParamsPtr->newCellState->soma.Potassium_x_s;
    dev_SomaCurrVolt(chComps_g_CaL, chComps_vDend, chComps_vSoma, chComps_newVSoma, chComps_vAxon, chComps_k, chComps_l, chComps_m, chComps_h, chComps_n, chComps_x_s);

    return;
}

__device__ void dev_SomaCalcium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1, double *chPrms_newComp2){

    double k_inf, l_inf, tau_k, tau_l, dk_dt, dl_dt, k_local, l_local;

    //Get inputs
    double prevV_soma = *chPrms_v; //*chPrms->v;
    double prevCalcium_k = *chPrms_prevComp1; //*chPrms->prevComp1;
    double prevCalcium_l = *chPrms_prevComp2; //*chPrms->prevComp2;

    k_inf = (1 / (1 + exp(-1 * (prevV_soma + 61)   / 4.2)));
    l_inf = (1 / (1 + exp((     prevV_soma + 85.5) / 8.5)));
    tau_k = 1;
    tau_l = ((20 * exp((prevV_soma + 160) / 30) / (1 + exp((prevV_soma + 84) / 7.3))) +35);
    dk_dt = (k_inf - prevCalcium_k) / tau_k;
    dl_dt = (l_inf - prevCalcium_l) / tau_l;
    k_local = DELTA * dk_dt + prevCalcium_k;
    l_local = DELTA * dl_dt + prevCalcium_l;
    //Put result
    *chPrms_newComp1= k_local; //*chPrms->newComp1
    *chPrms_newComp2= l_local; //*chPrms->newComp2

    //return;
}

__device__ void dev_SomaSodium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1, double *chPrms_newComp2){

    double m_inf, h_inf, tau_h, dh_dt, m_local, h_local;

    //Get inputs
    double prevV_soma = *chPrms_v; //*chPrms->v;
    //mod_prec prevSodium_m = *chPrms->prevComp1;
    double prevSodium_h = *chPrms_prevComp2; //*chPrms->prevComp2;

    // RAT THALAMOCORTICAL SODIUM:
    m_inf   = 1 / (1 + (exp((-30 - prevV_soma)/ 5.5)));
    h_inf   = 1 / (1 + (exp((-70 - prevV_soma)/-5.8)));
    tau_h   =       3 * exp((-40 - prevV_soma)/33);
    dh_dt   = (h_inf - prevSodium_h)/tau_h;
    m_local       = m_inf;
    h_local       = prevSodium_h + DELTA * dh_dt;
    //Put result
    *chPrms_newComp1 = m_local; //*chPrms->newComp1
    *chPrms_newComp2 = h_local; //*chPrms->newComp2

    return;
}

__device__ void dev_SomaPotassium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_prevComp2, double *chPrms_newComp1, double *chPrms_newComp2){

    double n_inf, p_inf, tau_n, tau_p, dn_dt, dp_dt, n_local, p_local;

    //Get inputs
    double prevV_soma = *chPrms_v; //*chPrms->v;
    double prevPotassium_n = *chPrms_prevComp1; //*chPrms->prevComp1;
    double prevPotassium_p = *chPrms_prevComp2; //*chPrms->prevComp2;

    // NEOCORTICAL
    n_inf = 1 / (1 + exp( ( -3 - prevV_soma) /  10));
    p_inf = 1 / (1 + exp( (-51 - prevV_soma) / -12));
    tau_n =   5 + (  47 * exp( -(-50 - prevV_soma) /  900));
    tau_p = tau_n;
    dn_dt = (n_inf - prevPotassium_n) / tau_n;
    dp_dt = (p_inf - prevPotassium_p) / tau_p;
    n_local = DELTA * dn_dt + prevPotassium_n;
    p_local = DELTA * dp_dt + prevPotassium_p;
    //Put result
    *chPrms_newComp1 = n_local; //*chPrms->newComp1
    *chPrms_newComp2 = p_local; //*chPrms->newComp2

    return;
}

__device__ void dev_SomaPotassiumX(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1){

    double alpha_x_s, beta_x_s, x_inf_s, tau_x_s, dx_dt_s, x_s_local;

    //Get inputs
    double prevV_soma = *chPrms_v; //*chPrms->v;
    double prevPotassium_x_s = *chPrms_prevComp1; //*chPrms->prevComp1;

    // Voltage-dependent (fast) potassium
    alpha_x_s = 0.13 * (prevV_soma + 25) / (1 - exp(-(prevV_soma + 25) / 10));
    beta_x_s  = 1.69 * exp(-0.0125 * (prevV_soma + 35));
    x_inf_s   = alpha_x_s / (alpha_x_s + beta_x_s);
    tau_x_s   =         1 / (alpha_x_s + beta_x_s);
    dx_dt_s   = (x_inf_s - prevPotassium_x_s) / tau_x_s;
    x_s_local       = 0.05 * dx_dt_s + prevPotassium_x_s;
    //Put result
    *chPrms_newComp1 = x_s_local; //*chPrms->newComp1

    return;
}

__device__ void dev_SomaCurrVolt(double *chComps_g_CaL, double *chComps_vDend, double *chComps_vSoma, double *chComps_newVSoma, double *chComps_vAxon, double *chComps_k, double *chComps_l, double *chComps_m, double *chComps_h, double *chComps_n, double *chComps_x_s){

    //Local variables
    double I_ds, I_CaL, I_Na_s, I_ls, I_Kdr_s, I_K_s, I_as, dVs_dt;

    //Get inputs
    double g_CaL = *chComps_g_CaL; //*chComps->g_CaL;
    double prevV_dend = *chComps_vDend; //*chComps->vDend;
    double prevV_soma = *chComps_vSoma; //*chComps->vSoma;
    double prevV_axon = *chComps_vAxon; //*chComps->vAxon;
    double k = *chComps_k; //*chComps->k;
    double l = *chComps_l; //*chComps->l;
    double m = *chComps_m; //*chComps->m;
    double h = *chComps_h; //*chComps->h;
    double n = *chComps_n; //*chComps->n;
    double x_s = *chComps_x_s; //*chComps->x_s;

    // SOMATIC CURRENTS

    // Dendrite-soma interaction current I_ds
    I_ds  = (G_INT / P1) * (prevV_soma - prevV_dend);
    // Inward low-threshold Ca current I_CaL
    I_CaL = g_CaL * k * k * k * l * (prevV_soma - V_CA); //k^3
    // Inward Na current I_Na_s
    I_Na_s  = G_NA_S * m * m * m * h * (prevV_soma - V_NA);
    // Leakage current I_ls
    I_ls  = G_LS * (prevV_soma - V_L);
    // Outward delayed potassium current I_Kdr
    I_Kdr_s = G_KDR_S * n * n * n * n * (prevV_soma - V_K); // SCHWEIGHOFER
    // I_K_s
    I_K_s   = G_K_S * pow(x_s, 4) * (prevV_soma - V_K);
    // Axon-soma interaction current I_as
    I_as    = (G_INT / (1 - P2)) * (prevV_soma - prevV_axon);

    dVs_dt = (-(I_CaL   + I_ds  + I_as + I_Na_s + I_ls   + I_Kdr_s + I_K_s)) / C_M;
    *chComps_newVSoma = DELTA * dVs_dt + prevV_soma; // *chComps->newVSoma

    return;
}

__device__ void dev_CompAxon(double *cellCompParamsPtr){

	double *chPrms_v;
	double *chPrms_prevComp1;// *chPrms_prevComp2;
	double *chPrms_newComp1, *chPrms_newComp2;
	double *chComps_vSoma;
	double *chComps_vAxon;
	double *chComps_m_a, *chComps_h_a, *chComps_x_a;
	double *chComps_newVAxon;

    // update somatic components
    // SCHWEIGHOFER:

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]);//prevCellState->axon.V_axon;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_SH]);//prevCellState->axon.Sodium_h_a;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SH]);//&cellCompParamsPtr->newCellState->axon.Sodium_h_a;
    chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SM]);//&cellCompParamsPtr->newCellState->axon.Sodium_m_a;
    //Compute
    dev_AxonSodium(chPrms_v, chPrms_prevComp1, chPrms_newComp1, chPrms_newComp2);

    //Prepare pointers to inputs/outputs
    chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]);//&cellCompParamsPtr->prevCellState->axon.V_axon;
    chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_P]);//&cellCompParamsPtr->prevCellState->axon.Potassium_x_a;
    chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_P]);//&cellCompParamsPtr->newCellState->axon.Potassium_x_a;
    //Compute
    dev_AxonPotassium(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

    //Get inputs
    chComps_vSoma = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]);//&cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps_vAxon = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]);//&cellCompParamsPtr->prevCellState->axon.V_axon;
    chComps_newVAxon = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_V]);//&cellCompParamsPtr->newCellState->axon.V_axon;
    chComps_m_a = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SM]);//&cellCompParamsPtr->newCellState->axon.Sodium_m_a;
    chComps_h_a = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SH]);//&cellCompParamsPtr->newCellState->axon.Sodium_h_a;
    chComps_x_a = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_P]);//&cellCompParamsPtr->newCellState->axon.Potassium_x_a;
    dev_AxonCurrVolt(chComps_vSoma, chComps_vAxon, chComps_newVAxon, chComps_m_a, chComps_h_a, chComps_x_a);

    return;
}

__device__ void dev_AxonSodium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1, double *chPrms_newComp2){

    double m_inf_a, h_inf_a, tau_h_a, dh_dt_a, m_a_local, h_a_local;

    //Get inputs
    double prevV_axon = *chPrms_v; //*chPrms->v;
    double prevSodium_h_a = *chPrms_prevComp1; //*chPrms->prevComp1;

    // Update axonal Na components
    // NOTE: current has shortened inactivation to account for high
    // firing frequencies in axon hillock
    m_inf_a   = 1 / (1 + (exp((-30 - prevV_axon)/ 5.5)));
    h_inf_a   = 1 / (1 + (exp((-60 - prevV_axon)/(-5.8))));
    tau_h_a   =     1.5 * exp((-40 - prevV_axon)/33);
    dh_dt_a   = (h_inf_a - prevSodium_h_a)/tau_h_a;
    m_a_local = m_inf_a;
    h_a_local = prevSodium_h_a + DELTA * dh_dt_a;
    //Put result
    *chPrms_newComp1 = h_a_local; //*chPrms->newComp1
    *chPrms_newComp2 = m_a_local; //*chPrms->newComp2

    return;
}

__device__ void dev_AxonPotassium(double *chPrms_v, double *chPrms_prevComp1, double *chPrms_newComp1){

    double alpha_x_a, beta_x_a, x_inf_a, tau_x_a, dx_dt_a, x_a_local;

    //Get inputs
    double prevV_axon = *chPrms_v; //*chPrms->v;
    double prevPotassium_x_a = *chPrms_prevComp1; //*chPrms->prevComp1;

    // D'ANGELO 2001 -- Voltage-dependent potassium
    alpha_x_a = 0.13 * (prevV_axon + 25) / (1 - exp(-(prevV_axon + 25) / 10));
    beta_x_a  = 1.69 * exp(-0.0125 * (prevV_axon + 35));
    x_inf_a   = alpha_x_a / (alpha_x_a + beta_x_a);
    tau_x_a   =         1 / (alpha_x_a + beta_x_a);
    dx_dt_a   = (x_inf_a - prevPotassium_x_a) / tau_x_a;
    x_a_local = 0.05 * dx_dt_a + prevPotassium_x_a;
    //Put result
    *chPrms_newComp1 = x_a_local; //*chPrms->newComp1

    return;
}

__device__ void dev_AxonCurrVolt(double *chComps_vSoma, double *chComps_vAxon, double *chComps_newVAxon, double *chComps_m_a, double *chComps_h_a, double *chComps_x_a){

    //Local variable
    double I_Na_a, I_la, I_sa, I_K_a, dVa_dt;

    //Get inputs
    double prevV_soma = *chComps_vSoma; //*chComps->vSoma;
    double prevV_axon = *chComps_vAxon; //*chComps->vAxon;
    double m_a = *chComps_m_a; //*chComps->m_a;
    double h_a = *chComps_h_a; //*chComps->h_a;
    double x_a = *chComps_x_a; //*chComps->x_a;

    // AXONAL CURRENTS
    // Sodium
    I_Na_a  = G_NA_A  * m_a * m_a * m_a * h_a * (prevV_axon - V_NA);
    // Leak
    I_la    = G_LA    * (prevV_axon - V_L);
    // Soma-axon interaction current I_sa
    I_sa    = (G_INT / P2) * (prevV_axon - prevV_soma);
    // Potassium (transient)
    //I_K_a   = G_K_A * pow(x_a, 4) * (prevV_axon - V_K);
    I_K_a   = G_K_A * x_a * x_a * x_a * x_a * (prevV_axon - V_K);
    dVa_dt = (-(I_K_a + I_sa + I_la + I_Na_a)) / C_M;
    *chComps_newVAxon = DELTA * dVa_dt + prevV_axon; //*chComps->newVAxon
    return;
}

int ReadFileLine(char *iAppBuf, int iAppBufSize, FILE *pInFile, double *iAppArray){
    //FIXME: make this function more robust
    char *strNumber;
    int i = 0;
    //Get one line
    if(fgets(iAppBuf, iAppBufSize, pInFile)){
        //Convert the ASCII string of one element to a double precision floating point value
        strNumber = strtok(iAppBuf," ");
        i = 0;
        //printf("Line:\n");
        while ((strNumber != NULL) && (i<IO_NETWORK_SIZE)){
            iAppArray[i] = atof(strNumber);//atof() should change if using integers or fixed point
            //printf ("(%s) %0.2f ", strNumber, iAppArray[i]);
            strNumber = strtok(NULL, " ");
            i++;
        }
        //printf("i: %d\n", i);
        if(i<IO_NETWORK_SIZE){
            //BUG: if only one element is missing but the line ends in a space, the error is not detected
            printf("Error: Input line doesn't have enough elements, only %d\n", i);
            exit(EXIT_FAILURE);
        }
        return 1;//success
    }else{
        if(!feof(pInFile)){
        printf("Error: Reading from input file didn't finish successfully\n");
        exit(EXIT_FAILURE);
        }
        return 0;//end of file
    }
}

static __inline__ __device__ double fetch_double(texture<int2, 2, cudaReadModeElementType> t, int x, int y)
{
int2 v = tex2D(t, x, y);
return __hiloint2double(v.y, v.x);
}
