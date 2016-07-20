#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

/*
// Copyright 2015 2016 Intel Corporation All Rights Reserved.
//
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title
// to such Material remains with Intel Corporation or its suppliers or
// licensors. The Material contains proprietary information of Intel
// or its suppliers and licensors. The Material is protected by worldwide
// copyright laws and treaty provisions. No part of the Material may be used,
// copied, reproduced, modified, published, uploaded, posted, transmitted,
// distributed or disclosed in any way without Intel's prior express written
// permission. No license under any patent, copyright or other intellectual
// property rights in the Material is granted to or conferred upon you,
// either expressly, by implication, inducement, estoppel or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing,
// you may not remove or alter this notice or any other notice embedded in
// Materials by Intel or Intel's suppliers or licensors in any way.
*/

// A simple example of the edge detection algorithm implemented with Intel IPP functions:
//     ippiFilterSobelNegVertBorder_8u16s_C1R
//     ippiFilterSobelHorizBorder_8u16s_C1R
//     ippiCanny_16s8u_C1R


#include <stdio.h>
#include "ipp.h"
#include "ippi.h"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define WIDTH  128  /* image width */
#define HEIGHT  64  /* image height */
#define THRESH_LOW    50.f /* Low   threshold for edges detection */
#define THRESH_HIGHT 150.f /* Upper threshold for edges detection */
#define BORDER_VAL 0

/* Next two defines are created to simplify code reading and understanding */
/* Next two defines are created to simplify code reading and understanding */
#define EXIT_MAIN exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; /* Go to Exit if IPP function returned status different from ippStsNoErr */


//function for initializing unsigned char matrix, modes ['0': init with zeros] ['1': init with 1,2,3,4] ['2': init with random values]
void randomMat(const char name[], unsigned char *h_data, char mode, int rows, int columns);

//function for writing a pgm image (of version 2 only) into file
void pgmwrite(const char fname[20], Ipp8u* h_data, int gscale, char version[10], char comment[100], int rows, int columns);

/* Results of ippMalloc() are not validated because IPP functions perform bad arguments check and will return an appropriate status  */


using namespace std;


int main(void)
{
	
	//system("start C:/Users/svgn1/Desktop/Proj_files/pgm_pic_read_write/pgm_pic_read_write/pgmtrial/images/lena.pgm");					// open input image

	ifstream infile("./images/im3.pgm");				// open a stream to input image

	char line[100], version[10], comment[100];
	char c;
	int rows, columns, gscale;

	infile.getline(version, 10, '\n');		// get version
	infile.getline(comment, 100, '\n');		// get comment
	infile.getline(line, 100, ' ');			// get rows		{notice ' ' instead of '\n' refer pmg image format in the beginning of this program}
	columns = atoi(line);
	infile.getline(line, 100, '\n');			// get columns
	rows = atoi(line);
	infile.getline(line, 100, '\n');			// get gray scale
	gscale = atoi(line);
	
	//----------------------------//


	IppStatus status = ippStsNoErr;
	Ipp8u* pSrc = NULL, *pDst = NULL;  /* Pointers to source/destination images */
	int srcStep = 0, dstStep = 0;      /* Steps, in bytes, through the source/destination images */
	IppiSize roiSize = { columns, rows }; /* Size of source/destination ROI in pixels */

	int iTmpBufSizeSobV = 0;
	int iTmpBufSizeSobH = 0;
	int iTmpBufSizeCanny = 0;
	int iTmpBufSize = 0;         /* Common work buffer size */
	int  dxStep = 0, dyStep = 0; /* Steps, in bytes, through buffer_dx/buffer_dy */
	Ipp8u *buffer = NULL;        /* Pointer to the common work buffer */
	Ipp16s* buffer_dx = NULL, *buffer_dy = NULL; /* Pointer to the buffer for first derivatives with respect to X / Y */

	pSrc = ippiMalloc_8u_C1(roiSize.width, roiSize.height, &srcStep);
	pDst = ippiMalloc_8u_C1(roiSize.width, roiSize.height, &dstStep);


	int i = 0;
	if (version[1] == '2')						// read data for v2 images
	{
		while (i<rows*columns){
			infile.getline(line, 100, ' ');
			int dat = atoi(line);
			if (dat != 0){
				//h_data[i] = (float)dat;
				pSrc[i] = (float)dat;
				//printf("%d\n",h_data[i]);
				i++;
			}
		}
	}
	else if (version[1] == '5')				// read data for v5 images
	{
		version[1] = '2';
		while (i<rows*columns)
		{
			infile.get(c);
			//h_data[i] = (float)c;
			pSrc[i] = (float)c;
			i++;
		}
	}
	else
	{
		printf("pgm version unknown\n");
		return 0;
	}

	printf("version %s\n", version);
	printf("comment %s\n", comment);
	printf("height of the image is %d\n", rows);
	printf("width  of the image is %d\n", columns);
	printf("gscale of the image is %d\n", gscale);		//print all the values
	printf("%d of %d pixels read\n", i, rows*columns);

	/* Start Edge Detection algorithm */
	{
		buffer_dx = ippiMalloc_16s_C1(roiSize.width, roiSize.height, &dxStep);
		buffer_dy = ippiMalloc_16s_C1(roiSize.width, roiSize.height, &dyStep);

		check_sts(status = ippiFilterSobelVertBorderGetBufferSize(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, &iTmpBufSizeSobV))
			check_sts(status = ippiFilterSobelHorizBorderGetBufferSize(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, &iTmpBufSizeSobH))

			check_sts(ippiCannyGetSize(roiSize, &iTmpBufSizeCanny))

			/* Find maximum buffer size */
			iTmpBufSize = (iTmpBufSizeSobV > iTmpBufSizeSobH) ? iTmpBufSizeSobV : iTmpBufSizeSobH;
		iTmpBufSize = (iTmpBufSize > iTmpBufSizeCanny) ? iTmpBufSize : iTmpBufSizeCanny;
		buffer = ippsMalloc_8u(iTmpBufSize);

		check_sts(status = ippiFilterSobelNegVertBorder_8u16s_C1R(pSrc, srcStep, buffer_dx, dxStep, roiSize, ippMskSize3x3, ippBorderRepl, BORDER_VAL, buffer))

			check_sts(status = ippiFilterSobelHorizBorder_8u16s_C1R(pSrc, srcStep, buffer_dy, dyStep, roiSize, ippMskSize3x3, ippBorderRepl, BORDER_VAL, buffer))

			check_sts(status = ippiCanny_16s8u_C1R(buffer_dx, dxStep, buffer_dy, dyStep, pDst, dstStep, roiSize, THRESH_LOW, THRESH_HIGHT, buffer))
	}
	
	
	pgmwrite("inv.pgm", pDst, gscale, version, comment, rows, columns);

	infile.close();

	EXIT_MAIN
	ippsFree(buffer);
	ippiFree(buffer_dx);
	ippiFree(buffer_dy);
	ippiFree(pSrc);
	ippiFree(pDst);
	printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
	return (int)status;
}


void pgmwrite(const char fname[20], Ipp8u* h_data, int gscale, char version[10], char comment[100], int rows, int columns)
{
	char location[100] = "./images/";
	FILE *outfile = fopen("./images/inv.pgm", "w");
	fprintf(outfile, "%s\n%s\n%d  %d\n%d", version, comment, columns, rows, gscale);
	int i = 0;

	while (i< (rows*columns))
	{
		if (i % 12 != 0) fprintf(outfile, " ");
		else fprintf(outfile, "\n");
		fprintf(outfile, "%d ", h_data[i]);
		i++;
	}
	fclose(outfile);
	//char cmd[500] = "start ";
	//system(strcat(cmd, location));
}

void randomMat(const char name[], unsigned char *h_data, char mode, int rows, int columns){						// Allocates a matrix with random float entries.
	for (int i = 0; i < rows*columns; ++i){
		switch (mode){
		case '0': h_data[i] = (unsigned char)0;							break;
		case '1': h_data[i] = (unsigned char)i;							break;
		case '2': h_data[i] = (unsigned char)(rand() * 100 / (float)RAND_MAX);	break;
		default: h_data[i] = 99;
		}
	}
	//	printMat(name, h_data);
}