#include <stdio.h>              //standard includes
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#include <npp.h>                //cuda includes
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16
#define CLAMP_TO_EDGE 1
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define TH_LOW 200
#define TH_HIGH 250
#define HYST_MAX_SIZE 128
#define SOBEL_SMEM_HEIGHT 18
#define SOBEL_SMEM_WIDTH 18
#define HYS_COUNT 2

#define IMAGE_SIZE (width * height * sizeof(float))
#define SIZE_INT (width * height * sizeof(int))

float *h_img = NULL;
float *h_result = NULL;
float *d_img = NULL;
float *d_img_x =  NULL;
float *d_img_y =  NULL;
float *d_temp = NULL;
float *d_result = NULL;
float *d_hys_result = NULL;
float *d_nms_result = NULL;
unsigned int *gradientDirection = NULL;float *gradientStrength = NULL;float *gradientStrengthOut = NULL;


float sigma = 1.0f;
int order = 2;
int nthreads = 64;  // number of threads per block
    
void pgmwrite(const char fname[20], float *h_data, int gscale, char version[10], char comment[100], int height, int width);
void initCudaBuffers(int width, int height);
void cleanup();

int iDivUp(int a, int b);
void transpose(float *d_src, float *d_dest, int width, int height);
void deriche_gaussian_filer(float *d_src, float *d_dest, float *d_temp, int width, int height, float sigma, int order, int nthreads);
void cuda_hysteris(float *gradStrength,float *gradStrengthOut,
                    unsigned short int iWidth,unsigned short int iHeight,
                    float thresholdLow, float thresholdHigh);
void non_max_suppr(  float *gradX,float *gradY,float *gradStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight);



//BLOCK_CONVERT

__global__ void cannyBlockConverter(float *gradStrength,
                                    float *outputImage,
                                    unsigned short iWidth,unsigned short iHeight){
    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned char bx=blockIdx.x;
    unsigned short int x=BLOCK_WIDTH*bx+tx;
    unsigned short int y=BLOCK_HEIGHT*blockIdx.y+ty;
    unsigned int ref2=iWidth*y+x;

    __shared__ unsigned int sharedStrength[BLOCK_HEIGHT][BLOCK_WIDTH][3];
    ////Load global data into shared
    //Load center
    __syncthreads();
    float str=gradStrength[ref2];

    
    bool isLine=(str==-2);

    sharedStrength[ty][tx][0]=255*isLine;
    sharedStrength[ty][tx][1]=255*isLine;
    sharedStrength[ty][tx][2]=255*isLine;
    

    x=(ty)*BLOCK_WIDTH*3+tx*4;
    __syncthreads();
    if(tx<12){
        *((unsigned int *)(outputImage+3*(iWidth*y+BLOCK_WIDTH*bx)+tx*4))=\
              *(sharedStrength[0][0]+x+0)\
            +((*(sharedStrength[0][0]+x+1))<<8)\
            +((*(sharedStrength[0][0]+x+2))<<16)\
            +((*(sharedStrength[0][0]+x+3))<<24);
    }

}
__global__ void cannyBlockConverter8(float *gradStrength,
                                        float *outputImage,
                                        unsigned short iWidth,unsigned short iHeight){
    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned char bx=blockIdx.x;
    unsigned short int x=BLOCK_WIDTH*bx+tx;
    unsigned short int y=BLOCK_HEIGHT*blockIdx.y+ty;
    unsigned int ref2=iWidth*y+x;

    __shared__ unsigned int sharedStrength[BLOCK_HEIGHT][BLOCK_WIDTH][3];
    ////Load global data into shared
    //Load center
    __syncthreads();
    float str=gradStrength[ref2];

    
    bool isLine=(str==-2);

    sharedStrength[ty][tx][0]=255*isLine;
    sharedStrength[ty][tx][1]=255*isLine;
    sharedStrength[ty][tx][2]=255*isLine;
    

    x=(ty)*BLOCK_WIDTH+tx*4;
    __syncthreads();
    if(tx<12){
        *((unsigned int *)(outputImage+(iWidth*y+BLOCK_WIDTH*bx)+tx*4))=\
               *(sharedStrength[0][0]+x+0)\
            +((*(sharedStrength[0][0]+x+1))<<8)\
            +((*(sharedStrength[0][0]+x+2))<<16)\
            +((*(sharedStrength[0][0]+x+3))<<24);
    }

}
// GAUSSIAN KERNEL


__global__ void
deriche_kernel(float *id, float *od, int w, int h, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= w) return;
    
    id += x;    // advance pointers to correct column
    od += x;

     // forward pass
    float xp,yp,yb ;

    for (int y = 0; y < h; y++) {
        float xc = *id;
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
        *od = (float)yc;
        id += w; od += w;    // move to next row
        xp = xc; yb = yp; yp = yc; 
    }


    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    
    float xn,xa,yn,ya;

    for (int y = h-1; y >= 0; y--) {
        float xc = (*id);
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn; xn = xc; ya = yn; yn = yc;
        *od += (float)yc;
        id -= w; od -= w;  // move to previous row
    }
}



// --------------- TRANSPOSE KERNEL----------------


__global__ void d_transpose(float *odata, float *idata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
    
    // reading the matrix tile into shared memory

    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}


// -------------------- HYSTERISIS BLOCK KERNEL ---------------------
__device__ unsigned int angleToDirection(float y,float x){
    //Calculate direction

    float dir=((x==0&&y==0)?0:atan2(y,x));
      
    if((dir>0.39269908169872414&&dir<=1.1780972450961724)||(dir>=-2.748893571891069&&dir<-1.9634954084936207)) return 45;
    else if((dir>1.1780972450961724&&dir<=1.9634954084936207)||(dir>=-1.9634954084936207&&dir<-1.1780972450961724)) return 90;
    else if((dir>1.9634954084936207&&dir<=2.748893571891069)||(dir>=-1.1780972450961724&&dir<-0.39269908169872414)) return 135;
    else return 0;
}

__global__ void cudaHysteresisBlock(   float *gradStrength,float *gradStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight,
                                        float thresholdLow, float thresholdHigh)

{
    __shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
    unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
    
    unsigned int ref2=iWidth*y+x;
    
    //Load center
    sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];

    if(ty==0){//Load top
        sharedGradStrength[0][tx+1]=(y>0) ? (gradStrength[ref2-iWidth]): (0);
    }else if(ty==BLOCK_HEIGHT-1){//Load bottom
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+iWidth]): (0);
    }

    __syncthreads();

    if(tx==0){//Load left 
        sharedGradStrength[ty+1][0]=(x>0) ? (gradStrength[ref2-1]) : (0);
    }else if(tx==BLOCK_WIDTH-1){//Load right
        sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth) ? (gradStrength[ref2+1]) : (0);
    }

    __syncthreads();

    //Corners
    if(tx==0&&ty==0){
        sharedGradStrength[0][0]=(x>0&&y>0) ? (gradStrength[ref2-(iWidth+1)]) : (0);//TL
    }else if(tx==BLOCK_WIDTH-1&&ty==0){
        sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y>0 ) ? (gradStrength[ref2-(iWidth-1)]) : (0);//TR
    }else if(tx==0&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+(iWidth-1)]) : (0);//BL
    }else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y+BLOCK_HEIGHT<iHeight)? (gradStrength[ref2+(iWidth+1)]) : (0);//BR
    }
    
    __syncthreads();
    //Initialization part
    //Check if neighbors are edge pixels
    float str=sharedGradStrength[ty+1][tx+1];
    __syncthreads();

    if(str>thresholdHigh)       str=-2;
    else if(str>thresholdLow)   str=-1;
    else if(str>0)              str=0;

    sharedGradStrength[ty+1][tx+1]=str;

    __syncthreads();
    
    unsigned char list[HYST_MAX_SIZE][2];   //Dump into Local memory
    unsigned short listOff=0;

    if(str==-1){
        //Search neighbors
        //Seed list
        if( sharedGradStrength[ty][tx]==-2||sharedGradStrength[ty][tx+1]==-2||sharedGradStrength[ty][tx+2]==-2||
            sharedGradStrength[ty+1][tx]==-2||sharedGradStrength[ty+1][tx+2]==-2||
            sharedGradStrength[ty+2][tx]==-2||sharedGradStrength[ty+2][tx+1]==-2||sharedGradStrength[ty+2][tx+2]==-2){

            list[listOff][0]=ty+1;
            list[listOff++][1]=tx+1;
        }
    }

    unsigned char txReplace,tyReplace;
    __syncthreads();
    

    //Grow an edge and set potential edges
    for(x=0;x<listOff;++x){//While potential edge
        
        ty=list[x][0];
        tx=list[x][1];

        sharedGradStrength[ty][tx]=-2;//Set as definite edge
        
        //Check neighbors
        if(listOff<HYST_MAX_SIZE){
            if(sharedGradStrength[ty][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
                list[listOff][0]=ty;
                list[listOff++][1]=txReplace;
            }
            
            if(sharedGradStrength[ty][txReplace=max(tx-1,0)]==-1){
                list[listOff][0]=ty;
                list[listOff++][1]=txReplace;
            }
            
            if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][tx]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=tx;
            }
            
            if(sharedGradStrength[tyReplace=max(ty-1,0)][tx]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=tx;
            }

            if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }

            if(sharedGradStrength[tyReplace=max(ty-1,0)][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }

            if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][txReplace=max(tx-1,0)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }

            if(sharedGradStrength[tyReplace=max(ty-1,0)][txReplace=max(tx-1,0)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }
        }
        
    }

    tx=threadIdx.x;
    ty=threadIdx.y;
    __syncthreads();
    gradStrengthOut[ref2]=sharedGradStrength[ty+1][tx+1];
}


// ----------------------  NON MAX SUPRESSION ----------------


__global__ void nonmaxSupression(  unsigned int *gradDirection,float *gradStrength,float *gradStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight){
    __shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
    unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
    unsigned int ref2=iWidth*y+x;
    
    ////Load gradient strength data
    ////And 1 pixel apron

    //Load center (implicit coalesce reads from data format)
    sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];

    if(ty==0){//Load top (Implicit coalesce as center)
        sharedGradStrength[0][tx+1]=(y>0?gradStrength[ref2-iWidth]:0);
    }
    if(ty==BLOCK_HEIGHT-1){//Load bottom
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y<iHeight-1?gradStrength[ref2+iWidth]:0);
    }

    if(tx==0){
        //Load leftmost column (uncoalesced but only 1 thread per halfwarp)
        sharedGradStrength[ty+1][0]=(x>0?gradStrength[ref2-1]:0);
        //Load rightmost column (coalesced but only 1 thread per halfwar)
        sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth?gradStrength[ref2+BLOCK_WIDTH]:0);
    }

    //Corners
    if(tx==0&&ty==0){
        sharedGradStrength[0][0]=(x>0&&y>0?gradStrength[ref2-(iWidth+1)]:0);//TL
    }else if(tx==BLOCK_WIDTH-1&&ty==0){
        sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x<iWidth-1&&y>0?gradStrength[ref2-(iWidth-1)]:0);//TR
    }else if(tx==0&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y<iHeight-1?gradStrength[ref2+(iWidth-1)]:0);//BL
    }else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x<iWidth-1&&y<iHeight-1?gradStrength[ref2+(iWidth+1)]:0);//BR
    }

    __syncthreads();


    
    //Suppress gradient in all nonmaximum pixels
  
    gradStrengthOut[ref2]=sharedGradStrength[ty+1][tx+1];

}


// ----------grad strength and non max ------------

/*__device__ unsigned int angleToDirection(float y,float x){
    //Calculate direction

    float dir=((x==0&&y==0)?0:atan2(y,x));
      
    if((dir>0.39269908169872414&&dir<=1.1780972450961724)||(dir>=-2.748893571891069&&dir<-1.9634954084936207)) return 45;
    else if((dir>1.1780972450961724&&dir<=1.9634954084936207)||(dir>=-1.9634954084936207&&dir<-1.1780972450961724)) return 90;
    else if((dir>1.9634954084936207&&dir<=2.748893571891069)||(dir>=-1.1780972450961724&&dir<-0.39269908169872414)) return 135;
    else return 0;
}*/

/*
calculate gradient magnitude and direction
*/
__global__ void GradientStrengthDirection(float *gradX,float *gradY,
                                                unsigned short int iWidth, unsigned short int iHeight,
                                                float *gradStrength,unsigned int *gradDirection){
    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned short int x=BLOCK_WIDTH*blockIdx.x+tx;
    unsigned short int y=BLOCK_HEIGHT*blockIdx.y+ty;
    unsigned int ref2=iWidth*y+x;

    //get magnitude while loading to reuse space
    float strength,strength2,strengthMag;

    //Load gradX into shared mem
    //Load gradY into shared mem
    strength=gradX[ref2];
    strength2=gradY[ref2];

    //Calculate and coalesce write the most "salient" gradient magnitude to global mem
    strengthMag=sqrt(strength*strength+strength2*strength2);

    __syncthreads();
    gradStrength[ref2]=strengthMag;
    gradDirection[ref2]=angleToDirection(strength2,strength);

}


// -----------------------//

__global__ void NonmaxSupression(  unsigned int *gradDirection,float *gradStrength,float *gradStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight){
    __shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
    unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
    unsigned int ref2=iWidth*y+x;
    
    ////Load gradient strength data
    ////And 1 pixel apron

    //Load center (implicit coalesce reads from data format)
    sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];

    if(ty==0){//Load top (Implicit coalesce as center)
        sharedGradStrength[0][tx+1]=(y>0?gradStrength[ref2-iWidth]:0);
    }
    if(ty==BLOCK_HEIGHT-1){//Load bottom
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y<iHeight-1?gradStrength[ref2+iWidth]:0);
    }

    if(tx==0){
        //Load leftmost column (uncoalesced but only 1 thread per halfwarp)
        sharedGradStrength[ty+1][0]=(x>0?gradStrength[ref2-1]:0);
        //Load rightmost column (coalesced but only 1 thread per halfwar)
        sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth?gradStrength[ref2+BLOCK_WIDTH]:0);
    }

    //Corners
    if(tx==0&&ty==0){
        sharedGradStrength[0][0]=(x>0&&y>0?gradStrength[ref2-(iWidth+1)]:0);//TL
    }else if(tx==BLOCK_WIDTH-1&&ty==0){
        sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x<iWidth-1&&y>0?gradStrength[ref2-(iWidth-1)]:0);//TR
    }else if(tx==0&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y<iHeight-1?gradStrength[ref2+(iWidth-1)]:0);//BL
    }else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x<iWidth-1&&y<iHeight-1?gradStrength[ref2+(iWidth+1)]:0);//BR
    }

    __syncthreads();

    unsigned int f=gradDirection[ref2];
    
    x=(f==135?-1:(f==90?0:1));
    y=(f==135||f==45?-1:(f==0?0:1));

    //Is thread a maximum? //High chance of bank conflict
    bool a=(sharedGradStrength[ty+1][tx+1]>max(sharedGradStrength[ty+1+y][tx+1+x],sharedGradStrength[ty+1-y][tx+1-x]));
    
    //Suppress gradient in all nonmaximum pixels
    __syncthreads();
    gradStrengthOut[ref2]=sharedGradStrength[ty+1][tx+1]*a;

}


__global__ void cannyHysteresisBlock(   float *gradStrength,float *gradStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight,
                                        float thresholdLow, float thresholdHigh){
    __shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
    unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
    
    unsigned int ref2=iWidth*y+x;
    
    //Load center
    sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];

    if(ty==0){//Load top
        sharedGradStrength[0][tx+1]=(y>0) ? (gradStrength[ref2-iWidth]): (0);
    }else if(ty==BLOCK_HEIGHT-1){//Load bottom
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+iWidth]): (0);
    }

    __syncthreads();

    if(tx==0){//Load left 
        sharedGradStrength[ty+1][0]=(x>0) ? (gradStrength[ref2-1]) : (0);
    }else if(tx==BLOCK_WIDTH-1){//Load right
        sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth) ? (gradStrength[ref2+1]) : (0);
    }

    __syncthreads();

    //Corners
    if(tx==0&&ty==0){
        sharedGradStrength[0][0]=(x>0&&y>0) ? (gradStrength[ref2-(iWidth+1)]) : (0);//TL
    }else if(tx==BLOCK_WIDTH-1&&ty==0){
        sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y>0 ) ? (gradStrength[ref2-(iWidth-1)]) : (0);//TR
    }else if(tx==0&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+(iWidth-1)]) : (0);//BL
    }else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y+BLOCK_HEIGHT<iHeight)? (gradStrength[ref2+(iWidth+1)]) : (0);//BR
    }
    
    __syncthreads();
    //Initialization part
    //Check if neighbors are edge pixels
    float str=sharedGradStrength[ty+1][tx+1];
    __syncthreads();

    if(str>thresholdHigh)       str=-2;
    else if(str>thresholdLow)   str=-1;
    else if(str>0)              str=0;

    sharedGradStrength[ty+1][tx+1]=str;

    __syncthreads();
    
    unsigned char list[HYST_MAX_SIZE][2];   //Dump into Local memory
    unsigned short listOff=0;

    if(str==-1){
        //Search neighbors
        //Seed list
        if( sharedGradStrength[ty][tx]==-2||sharedGradStrength[ty][tx+1]==-2||sharedGradStrength[ty][tx+2]==-2||
            sharedGradStrength[ty+1][tx]==-2||sharedGradStrength[ty+1][tx+2]==-2||
            sharedGradStrength[ty+2][tx]==-2||sharedGradStrength[ty+2][tx+1]==-2||sharedGradStrength[ty+2][tx+2]==-2){

            list[listOff][0]=ty+1;
            list[listOff++][1]=tx+1;
        }
    }

    unsigned char txReplace,tyReplace;
    __syncthreads();
    

    //Grow an edge and set potential edges
    for(x=0;x<listOff;++x){//While potential edge
        
        ty=list[x][0];
        tx=list[x][1];

        sharedGradStrength[ty][tx]=-2;//Set as definite edge
        
        //Check neighbors
        if(listOff<HYST_MAX_SIZE){
            if(sharedGradStrength[ty][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
                list[listOff][0]=ty;
                list[listOff++][1]=txReplace;
            }
            
            if(sharedGradStrength[ty][txReplace=max(tx-1,0)]==-1){
                list[listOff][0]=ty;
                list[listOff++][1]=txReplace;
            }
            
            if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][tx]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=tx;
            }
            
            if(sharedGradStrength[tyReplace=max(ty-1,0)][tx]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=tx;
            }

            if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }

            if(sharedGradStrength[tyReplace=max(ty-1,0)][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }

            if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][txReplace=max(tx-1,0)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }

            if(sharedGradStrength[tyReplace=max(ty-1,0)][txReplace=max(tx-1,0)]==-1){
                list[listOff][0]=tyReplace;
                list[listOff++][1]=txReplace;
            }
        }
        
    }

    tx=threadIdx.x;
    ty=threadIdx.y;
    __syncthreads();
    gradStrengthOut[ref2]=sharedGradStrength[ty+1][tx+1];
}


__device__ float reduce256(volatile float smem256[],unsigned short tID){
    __syncthreads();
    if(tID<128) smem256[tID]+=smem256[tID+128];
    __syncthreads();
    if(tID<64)  smem256[tID]+=smem256[tID+64];
    __syncthreads();
    if(tID<32){
        smem256[tID]+=smem256[tID+32];
        smem256[tID]+=smem256[tID+16];
        smem256[tID]+=smem256[tID+8];
        smem256[tID]+=smem256[tID+4];
        smem256[tID]+=smem256[tID+2];
        smem256[tID]+=smem256[tID+1];
    }
    __syncthreads();
    return smem256[0];
}


__global__ void cannyHysteresisBlockShared( float *gradStrength,float *gradStrengthOut,
                                            unsigned short iWidth,unsigned short iHeight,
                                            float thresholdLow, float thresholdHigh){
    
    __shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

    unsigned char tx=threadIdx.x;
    unsigned char ty=threadIdx.y;
    unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
    unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
    unsigned int ref2=iWidth*y+x;
    
    //Load center
    sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];


    if(ty==0){//Load top
        sharedGradStrength[0][tx+1]=(y>0) ? (gradStrength[ref2-iWidth]): (0);
    }else if(ty==BLOCK_HEIGHT-1){//Load bottom
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+iWidth]): (0);
    }

    __syncthreads();

    if(tx==0){//Load left 
        sharedGradStrength[ty+1][0]=(x>0) ? (gradStrength[ref2-1]) : (0);
    }else if(tx==BLOCK_WIDTH-1){//Load right
        sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth) ? (gradStrength[ref2+1]) : (0);
    }

    __syncthreads();

    //Corners
    if(tx==0&&ty==0){
        sharedGradStrength[0][0]=(x>0&&y>0) ? (gradStrength[ref2-(iWidth+1)]) : (0);//TL
    }else if(tx==BLOCK_WIDTH-1&&ty==0){
        sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y>0 ) ? (gradStrength[ref2-(iWidth-1)]) : (0);//TR
    }else if(tx==0&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+(iWidth-1)]) : (0);//BL
    }else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
        sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y+BLOCK_HEIGHT<iHeight)? (gradStrength[ref2+(iWidth+1)]) : (0);//BR
    }


    
    __syncthreads();
    //Initialization part
    //Check if neighbors are edge pixels
    float str=sharedGradStrength[ty+1][tx+1];
    __syncthreads();

    if(str>thresholdHigh)       str=-2;     //Definite edge
    else if(str>thresholdLow)   str=-1;     //Potential edge
    else if(str>0)              str=0;      //not an edge

    ++tx;
    ++ty;

    __syncthreads();

    sharedGradStrength[ty][tx]=str;

    __shared__ int sharedModfied[BLOCK_HEIGHT][BLOCK_WIDTH];

    for(unsigned short a=0; a<HYST_MAX_SIZE; ++a){
        sharedModfied[ty-1][tx-1]=0;
        __syncthreads();
        //If potential edge, search neighbors for definite edge... if found, mark as definite edge 
        bool e=false;
        if(sharedGradStrength[ty][tx]==-1){
            e=e || (sharedGradStrength[ty][max(tx-1,0)]==-2);                   //Left      
            e=e || (sharedGradStrength[ty][min(tx+1,SOBEL_SMEM_WIDTH-1)]==-2);  //Right
            e=e || (sharedGradStrength[max(ty-1,0)][tx]==-2);                   //Top
            e=e || (sharedGradStrength[min(ty+1,SOBEL_SMEM_HEIGHT-1)][tx]==-2); //Bot
            e=e || (sharedGradStrength[max(ty-1,0)][max(tx-1,0)]==-2);                      //Top left
            e=e || (sharedGradStrength[max(ty-1,0)][min(tx+1, SOBEL_SMEM_WIDTH-1)]==-2);    //Top right
            e=e || (sharedGradStrength[min(ty+1,SOBEL_SMEM_HEIGHT-1)][max(tx-1,0)]==-2);                    //Bot left
            e=e || (sharedGradStrength[min(ty+1,SOBEL_SMEM_HEIGHT-1)][min(tx+1, SOBEL_SMEM_WIDTH-1)]==-2);  //Bot right
        }
        __syncthreads();

        if(e){
            sharedGradStrength[ty][tx]=-2;
            sharedModfied[ty-1][tx-1]=1;
        }
        int modified=reduce256((float*)&(sharedModfied[0][0]), ty*BLOCK_WIDTH+tx);
        
        if(modified==0) break;
    }

    __syncthreads();
    gradStrengthOut[ref2]=sharedGradStrength[ty][tx];
}


////////////////////////////////////////////////////////////////////////////////
// Program main  ----- DERICHE ALGORITHM
////////////////////////////////////////////////////////////////////////////////

using namespace std;
int main(int argc, char* argv[])
{
    
    
    
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice( &device );
    cudaGetDeviceProperties( &prop, device );
    printf("Device is %s\n", prop.name );   
    cudaError_t err; 
    NppStatus e;

    ifstream infile("./images/lenanandan.pgm");              // open a stream to input image
    
    char line[100], version[10],comment[100],c;
    int height,width,gscale;
    
    infile.getline(version,10,'\n');        // get version
    infile.getline(comment,100,'\n');       // get comment
    infile.getline(line,100,' ');           // get height     {notice ' ' instead of '\n' refer pmg image format in the beginning of this program}
    width=atoi(line);
    infile.getline(line,100,'\n');          // get width
    height=atoi(line);
    infile.getline(line,100,'\n');          // get gray scale
    gscale=atoi(line);
    
    NppiSize roiFull= {width , height };                             
    int malloc_size=height*width*sizeof(float);      //get image h_data from stream
    
    h_img = (float*) malloc(malloc_size);
    h_result = (float*) malloc(malloc_size);
    
    int i=0;
    if(version[1]=='2')
    {
    while(i<height*width){
        infile.getline(line,100,' ');
        int dat= atoi(line);
        if(dat!=0){
            h_img[i]= (float) dat;
            
            i++;
        }
    }
    }
    else if(version[1]=='5')
    {
        version[1]='2';
        while(i<height*width)
        {   
            infile.get(c);
            h_img[i]= (float) c;
            i++;
        }
    }
    else 
    {
        printf("pgm version unknown\n");
        return 0;
    }


    float maxe=0,mine=0;
    for(int i=0;i<height*width;i++)
    {
        if(h_img[i]>maxe) maxe=h_img[i];
        if(h_img[i]<mine) mine=h_img[i];   
    }

    printf("%20s   -   mine : %f  maxe: %f\n", "SOMETHING", mine, maxe);


    printf("version %s\n", version);
    printf("comment %s\n", comment);
    printf("height of the image is %d\n",    height);
    printf("width  of the image is %d\n", width);
    printf("gscale of the image is %d\n",  gscale);     //print all the values
    printf("%d of %d pixels read\n",i,height*width);


      

    initCudaBuffers(width,height);

    err = cudaMemcpy( d_img, h_img, IMAGE_SIZE, cudaMemcpyHostToDevice);
   

/*    if(err != cudaSuccess){
        printf("%s ---- \n", cudaGetErrorSting(err));
    } */

    deriche_gaussian_filer(d_img, d_result, d_temp, width, height, sigma, order, nthreads);

    width*=4;
    
    e = nppiFilterSobelVert_32f_C1R (d_result, width, d_img_y, width, roiFull);

  

    e = nppiFilterSobelHoriz_32f_C1R (d_result, width, d_img_x, width, roiFull);

   

    //e = nppiAdd_32f_C1R (d_img_x, width, d_img_y, width, d_img, width, roiFull);

     width/=4;

    //err = cudaMemcpy( h_result, d_result, IMAGE_SIZE, cudaMemcpyDeviceToHost);

    //e = nppiFilterSobelVert_32f_C1R ((Npp32f *)d_img, width, (Npp32f *)d_img_x, width, roiFull);

    non_max_suppr(d_img_x, d_img_y,d_img,width,height);

    
    //------------//
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid((width/(float)block.x),(height/(float)block.y), 1);
    for(int a=0; a< 1;++a){
        
        /*
        //Slow local memory intensive version
        cannyHysteresisBlock<<<grid,block>>>(   gradientStrengthOut,gradientStrength,
                                                iWidth,iHeight,
                                                thresholdLow, thresholdHigh);
                                    */

        //Fast version using shared memory only and parallel reduction
        cannyHysteresisBlockShared<<<grid,block>>>( d_img,d_hys_result,
                                                width,height,
                                                TH_LOW, TH_HIGH);

        cudaMemcpy(d_img,d_hys_result,width*height*sizeof(float),cudaMemcpyDeviceToDevice);
        cudaThreadSynchronize();
    //------------//
        
        cannyBlockConverter<<<grid,block>>>(d_hys_result,gradientStrength,width,height);
        //cannyBlockConverter8<<<grid,block>>>(d_hys_result,d_result,width,height);

    /*cuda_hysteris(d_img,d_hys_result,width,height,TH_LOW,TH_HIGH);*/
    //err = cudaMemcpy( h_result, d_img, IMAGE_SIZE, cudaMemcpyDeviceToHost);

     err = cudaMemcpy( h_result, gradientStrength, IMAGE_SIZE, cudaMemcpyDeviceToHost);

    pgmwrite("result.pgm",h_result,gscale,version,comment,height,width); 

    cleanup();
    cudaThreadExit();

}

}


void initCudaBuffers(int width, int height)
{
    

    // allocate device memory
    cudaMalloc( (void**) &d_img, IMAGE_SIZE);
    cudaMalloc( (void**) &d_img_x, IMAGE_SIZE);
    cudaMalloc( (void**) &d_img_y, IMAGE_SIZE);
    cudaMalloc( (void**) &d_temp, IMAGE_SIZE);
    cudaMalloc( (void**) &d_result, IMAGE_SIZE);
    cudaMalloc( (void**) &gradientStrength, IMAGE_SIZE);
    cudaMalloc( (void**) &gradientStrengthOut, IMAGE_SIZE);
    cudaMalloc( (void**) &gradientDirection, SIZE_INT);
    cudaMalloc( (void**) &d_hys_result, IMAGE_SIZE);
    cudaMalloc( (void**) &d_nms_result, IMAGE_SIZE);
    
}


void cleanup()
{
  
    if (!h_img) {
        free(h_img);
    }

    // de - allocating the cuda buffers

    cudaFree(d_img);
    cudaFree(d_img_x);
    cudaFree(d_img_y);
    cudaFree(d_temp);
    cudaFree(d_result);
    cudaFree(d_hys_result);
    cudaFree(d_nms_result);
    cudaFree(gradientDirection);
    cudaFree(gradientStrength);
    cudaFree(gradientStrengthOut);

}

void pgmwrite(const char fname[20], float *h_data, int gscale, char version[10], char comment[100], int height, int width)
{
    char location[30] = "./images/";
    FILE *outfile=fopen(strcat(location,fname), "w");
    fprintf(outfile,"%s\n%s\n%d  %d\n%d", version,comment, width, height,gscale);
    float maxe=0,mine=0;
    for(int i=0;i<height*width;i++)
    {
        if(h_data[i]>maxe) maxe=h_data[i];
        if(h_data[i]<mine) mine=h_data[i];   
    }
    int i=0;
    printf("%20s   -   mine : %f  maxe: %f\n", fname, mine, maxe);
    while(i<height*width)
    {
        if(i%12!=0) fprintf(outfile," ");
        else fprintf(outfile,"\n");
        //h_data[i]=((h_data[i]-mine)/(maxe-mine))*(float)255;
        if(h_data[i]<0) h_data[i]=0;
        fprintf(outfile,"%.0f ",h_data[i]);
        i++;
    }
    fclose(outfile);
    
}




//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*
    Transpose a 2D array (see SDK transpose example)
*/
void transpose(float *d_src, float *d_dest, int width, int height)
{
    dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose<<< grid, threads >>>(d_dest, d_src, width, height);
 
}


void deriche_gaussian_filer(float *d_src, float *d_dest, float *d_temp, int width, int height, float sigma, int order, int nthreads)
{
    // compute filter coefficients
    const float
        nsigma = sigma < 0.1f ? 0.1f : sigma,
        alpha = 1.695f / nsigma,
        ema = (float)std::exp(-alpha),
        ema2 = (float)std::exp(-2*alpha),
        b1 = -2*ema,
        b2 = ema2;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;
    switch (order) {
    case 0: {
        const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
        a0 = k;
        a1 = k*(alpha-1)*ema;
        a2 = k*(alpha+1)*ema;
        a3 = -k*ema2;
    } break;

    case 1: {
        const float k = (1-ema)*(1-ema)/ema;
        a0 = k*ema;
        a1 = a3 = 0;
        a2 = -a0;
    } break;

    case 2: {
        const float
            ea = (float)std::exp(-alpha),
            k = -(ema2-1)/(2*alpha*ema),
            kn = (-2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea));
        a0 = kn;
        a1 = -kn*(1+k*alpha)*ema;
        a2 = kn*(1-k*alpha)*ema;
        a3 = -kn*ema2;
    } break;

    default:
        fprintf(stderr, "gaussianFilter: invalid order parameter!\n");
        return;
    }
    coefp = (a0+a1)/(1+b1+b2);
    coefn = (a2+a3)/(1+b1+b2);

    // process columns

    deriche_kernel<<< iDivUp(width, nthreads), nthreads >>>(d_src, d_temp, width, height, a0, a1, a2, a3, b1, b2, coefp, coefn);    

    transpose(d_temp, d_dest, width, height);

    // process rows

    deriche_kernel<<< iDivUp(height, nthreads), nthreads >>>(d_dest, d_temp, height, width, a0, a1, a2, a3, b1, b2, coefp, coefn);

    transpose(d_temp, d_dest, height, width);
}

/*
void cuda_hysteris(float *gradStrength,float *gradStrengthOut,
                    unsigned short int iWidth,unsigned short int iHeight,
                    float thresholdLow, float thresholdHigh)

{
    dim3 grid(iDivUp(iWidth, BLOCK_DIM), iDivUp(iHeight, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    cudaHysteresisBlock<<<grid,threads>>>(gradStrength,gradStrengthOut,iWidth,iHeight,thresholdLow,thresholdHigh);
}*/

void cuda_hysteris(float *gradStrength,float *gradStrengthOut,
                    unsigned short int iWidth,unsigned short int iHeight,
                    float thresholdLow, float thresholdHigh)

{
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid((iWidth/(float)block.x),(iHeight/(float)block.y), 1);
    for(int a=0; a< HYS_COUNT;++a){
        
        /*
        //Slow local memory intensive version
        cannyHysteresisBlock<<<grid,block>>>(   gradientStrengthOut,gradientStrength,
                                                iWidth,iHeight,
                                                thresholdLow, thresholdHigh);
                                    */

        //Fast version using shared memory only and parallel reduction
        cannyHysteresisBlockShared<<<grid,block>>>( gradStrengthOut,gradStrength,
                                                iWidth,iHeight,
                                                thresholdLow, thresholdHigh);

        cudaMemcpy(gradStrengthOut,gradStrength,iWidth*iHeight*sizeof(float),cudaMemcpyDeviceToDevice);
        cudaThreadSynchronize();
    }

}



/*void non_max_suppr(  unsigned int *gradDirection,float *gradStrength,float *gradStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight)

{
    dim3 grid(iDivUp(iWidth, BLOCK_DIM), iDivUp(iHeight, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    unsigned int gradDir[4] = {0,45,90,135};
    nonmaxSupression<<<grid,threads>>>(gradDir,gradStrength,gradStrengthOut,iWidth,iHeight);
}*/


void non_max_suppr(  float *gradX,float *gradY,float *gradientStrengthOut,
                                        unsigned short int iWidth,unsigned short int iHeight)

{   

    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid((iWidth/(float)block.x),(iHeight/(float)block.y), 1);
    //Find magnitude and direction
    GradientStrengthDirection<<<grid,block>>>(  gradX,gradY,
                                                    iWidth,iHeight,
                                                    gradientStrength,gradientDirection);


    //Find nonmaximum supression (thin pixels)
    NonmaxSupression<<<grid,block>>>( gradientDirection,gradientStrength,gradientStrengthOut,
                                            iWidth,iHeight);

}



