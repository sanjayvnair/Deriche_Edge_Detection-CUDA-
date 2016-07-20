#include <stdio.h>				//standard includes
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>


#include <npp.h>				//cuda includes
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define complexity 49	//complexity multiplication constant of 5x5 kernel (24 additions, 25 multiplications)
#define x 14		//sets width and height of image to 2^width
#define P 1000			// # of trials


//function for initializing unsigned char matrix, modes ['0': init with zeros] ['1': init with 1,2,3,4] ['2': init with random values between 0 and 255]
void randomMat (const char name[], float *h_data, char mode, int rows, int columns);

double timerval ();

using namespace std;

int main(int argc, char* argv[])
{
	int dev=0;								// print cuda capable device name
	cudaDeviceProp prop;
	cudaSetDevice(dev);
	cudaGetDeviceProperties(&prop, dev);
	printf("device: %s\n\n",prop.name);
	
	int rows=pow(2,x);
	int columns=rows;
	int step = columns*sizeof(float);
	float *d_data, *d_ix;
	float *h_data, *h_ix;
						
	int mallocsize=rows*columns*sizeof(float);		

	h_data = (float*) malloc(mallocsize);
	h_ix	=  (float*) malloc(mallocsize);							//allocating host memory
	printf("host mallocked\n");
	
	randomMat ("h_data", h_data, '2', rows, columns);		//initializing host memory
	randomMat ("h_ix",   h_ix,   '0', rows, columns);		//initializing host memory
	printf("host initialized\n");
	
	int i;
	float minE=0,maxE=0;		
	for(i=0;i<rows*columns;i++)					//get max and min values h_ix matrix
	{
		if(h_ix[i]<minE)	minE=h_ix[i];
		if(h_ix[i]>maxE)	maxE=h_ix[i];
	}
	printf("h_ix   minE : %f    maxE : %f\n",minE,maxE);  
	
	minE=0;maxE=0;
	for(i=0;i<rows*columns;i++)					//get max and min values h_ix matrix
	{
		if(h_data[i]<minE)	minE=h_data[i];
		if(h_data[i]>maxE)	maxE=h_data[i];
	}
	printf("h_data minE : %f    maxE : %f\n",minE,maxE);  //maxE usually ~ 650
	
	Npp32f *d_Kx;
					   
	Npp32f h_Kx[25]= {-2,-2,-2,-2,-2, 
					   -1,-1,-1,-1,-1, 
						0, 0, 0, 0, 0, 
						1, 1, 1, 1, 1,
						2, 2, 2, 2, 2}; 

	//Npp32s divisor	= 1;		//npp initializations
	NppStatus e;
	NppiSize ksize	= {5,5};
	NppiPoint anchor= {3,3};
	NppiSize roi	= {columns-5+1, rows-5+1};
	
	cudaMalloc <float> (&d_Kx ,5*5*sizeof(Npp32f));	//allocating device memory
	cudaMalloc <float> (&d_ix ,mallocsize);
	cudaMalloc <float> (&d_data ,mallocsize);
	printf("device mallocked\n");

	cudaMemcpy(d_Kx,	h_Kx,5*5*sizeof(Npp32s),	cudaMemcpyHostToDevice);	//initializing device mem via mem transfer
	cudaMemcpy(d_ix, 	h_ix, mallocsize, 	cudaMemcpyHostToDevice);
	cudaMemcpy(d_data,	h_data,mallocsize,  cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();		//waiting for mem transfer to complete
	cudaThreadSynchronize();
	printf("device initialized\n");

	printf("running nppiFilter_32f_C1R %d times\n",P);
	double t=timerval();
	for(int p=0;p<P;p++)
	{
	e = nppiFilter_32f_C1R(d_data, step, d_ix, step, roi, d_Kx, ksize, anchor);		//data x conv
	}
	cudaDeviceSynchronize();
	
	t=timerval()-t;
	t/=P;	//CORRECTING FOR P trials
	
	printf("error status is %s\n",e);	
	
	cudaMemcpy(h_ix, 	d_ix,mallocsize, 	cudaMemcpyDeviceToHost);	//transfering result from device to host
	
	float gflops;
	gflops =(float)(complexity*(float)columns*(float)rows)/(t*pow(10,9));  //9 for GFLOP/s, 12 for TFLOP/s
	printf("pixels        : %dx%d=%d\n",rows,columns,columns*rows);
	printf("time of exec  : %f\n",t);
	printf("GFLOP/s       : %f\n",gflops);		
	
	free(h_data);
	free(h_ix);
	cudaFree(d_data);
	cudaFree(d_ix);
	cudaFree(d_Kx);
	
	
}

void randomMat(const char name[], float *h_data, char mode, int rows, int columns)
{		// initializes a matrix with random float entries.
	for (int i = 0; i < rows*columns; ++i)
	{
		switch(mode)
		{
			case '0': h_data[i]=(unsigned char)0;							break;
			case '1': h_data[i]=(unsigned char)i;							break;
			case '2': h_data[i]=(((float)rand())*255/(float)RAND_MAX);	break;
			default : h_data[i]=99;
		}
	}
//	printMat(name, h_data);
}

double timerval () 
{
	return (double)(clock()/CLOCKS_PER_SEC);
}
