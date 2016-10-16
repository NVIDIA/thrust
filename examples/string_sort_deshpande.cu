#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <math.h>

#define MAX_THREADS_PER_BLOCK 1024

/*
* Author : Aditya Deshpande (aditya12agd5@gmail.com)
*
* CUDA/Thrust Implementation of String Sort algorithm given in IEEE High Performance 
* Computing 2013 paper:
*
* "Can GPUs Sort Strings Efficiently?",
* By: Aditya Deshpande and P J Narayanan
* 
* Copyright (c) 2013 International Institute of Information Technology - Hyderabad. 
* All rights reserved.
*  
* Permission to use, copy, modify and distribute this software and its documentation for 
* educational purpose is hereby granted without fee, provided that the above copyright 
* notice and this permission notice appear in all copies of this software and that you do 
* not sell the software.
*  
* THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESSED, IMPLIED OR 
* OTHERWISE.
*
*
* File: stringSortCvit.cu
* 	- Code that performs string sort using thrust sort, scatter, scan
*	- Refer to the paper for algorithm
*	- To Execute: ./stringSortCvit <input file> <write output flag>
*/



using namespace std;

//#define __PRINT_DEBUG__ 

int printSortedOutput(thrust::host_vector<unsigned int> valuesSorted, thrust::host_vector<unsigned char> stringVals, int numElements, int stringSize, 
		char inputFile[500]) {
	int retval = 0;
	char outFile[500];

	sprintf(outFile,"%s_string_sort_output",inputFile);

	printf("[DEBUG] Writing Output to file %s\n", outFile);
	FILE *fp = fopen(outFile,"w");

	for(unsigned int i = 0; i < numElements; ++i) {
		unsigned int index = valuesSorted[i];
		while(true) { 
			char ch;
			ch = (char)(stringVals[index]);
			if(ch == '\0') break;
			fprintf(fp,"%c",ch);
			index++;
		}
		fprintf(fp,"\n");
	}	
	return retval;
}

double calculateDiff (struct timespec t1, struct timespec t2) { 
	return (((t1.tv_sec - t2.tv_sec)*1000.0) + (((t1.tv_nsec - t2.tv_nsec)*1.0)/1000000.0));
}

__global__ void findSuccessor( unsigned char *d_array_stringVals, unsigned long long int *d_array_segment_keys,  
		unsigned int *d_array_valIndex, unsigned long long int *d_array_segment_keys_out,  unsigned int numElements, 
		unsigned int stringSize, unsigned int charPosition, unsigned int segmentBytes) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
	if(threadID > numElements) return;
	d_array_segment_keys_out[threadID] = 0;

	if(threadID > 0) { 
		if(d_array_segment_keys[threadID] != d_array_segment_keys[threadID-1]) { 
			d_array_segment_keys_out[threadID] = ((unsigned long long int)(1) << 56);
		}
	}

	unsigned int stringIndex = d_array_valIndex[threadID];
	unsigned long long int currentKey = (d_array_segment_keys[threadID] << (segmentBytes*8));
	unsigned char ch;
	int i = 0;
	unsigned int end = 0;

	for(i = 7; i >= ((int)segmentBytes); i--) { 
		ch = (unsigned char)(currentKey >> (i*8));
		if(ch == '\0') { 
			end = 1;
			break;
		}
	}
	
    if(end == 0) {
		unsigned int startPosition = charPosition;
		if( stringIndex + startPosition < stringSize ) { 
			unsigned long long int a = (unsigned long long int)(d_array_stringVals + stringIndex + startPosition);
			unsigned long long int *addr1 = (unsigned long long int *)(a & 0xfffffffffffffff8);
			unsigned long long int *addr2 = addr1 + 1;
			unsigned long long int part1 = *addr1;
			unsigned long long int part2 = *addr2;

			unsigned long long int offsetAddressVal = (a & 0x000000000000000f);
			unsigned long long int lowerMaskAddressVal = (a & 0x0000000000000008);

			long int diff = (offsetAddressVal - lowerMaskAddressVal);
			part1 >>= (diff * 8);
			part2 <<= ((8 - diff) * 8);

			unsigned long long int ans = (part1 | part2);
			for(i = 0; i < 7; i++) { 
				unsigned char ch = (ans >> (i*8));
				d_array_segment_keys_out[threadID] |= ((unsigned long long int) ch << ((6-i)*8)); 
				if(ch == '\0') break;
			}
		}
		
	} else { 
		d_array_segment_keys_out[threadID] = ((unsigned long long int)(1) << 56);
	}
}

__global__ void  eliminateSingleton(unsigned int *d_array_output_valIndex, unsigned int *d_array_valIndex, unsigned int *d_array_static_index, 
	unsigned int *d_array_map, unsigned int *d_array_stencil, int currentSize) {

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
	if(threadID >= currentSize) return;

	d_array_stencil[threadID] = 1;

        if(threadID == 0 && (d_array_map[threadID + 1] == 1)) { 
		d_array_stencil[threadID] = 0; 
	} else if( (threadID == (currentSize-1)) && (d_array_map[threadID] == 1) ) {
		d_array_stencil[threadID] = 0;  
	} else if( (d_array_map[threadID] == 1) && (d_array_map[threadID + 1] == 1)) { 
		d_array_stencil[threadID] = 0; 
	}
	
	if(d_array_stencil[threadID] == 0) { 
		d_array_output_valIndex[ d_array_static_index[threadID] ] = d_array_valIndex[threadID]; 
	}
}

__global__ void rearrangeSegMCU(unsigned long long int *d_array_segment_keys, unsigned long long int *d_array_segment_keys_out, 
		unsigned int *d_array_segment, unsigned int segmentBytes, unsigned int numElements) { 

	int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
	if(threadID >= numElements) return;
	
	unsigned long long int currentKey = (d_array_segment_keys_out[threadID] << 8);
	unsigned long long int segmentID  = (unsigned long long int) d_array_segment[threadID];
	d_array_segment_keys[threadID] = (segmentID << ((8-segmentBytes)*8));
	d_array_segment_keys[threadID] |= (currentKey >> (segmentBytes*8));
	return;
}

struct get_segment_bytes {
	__host__ __device__
	unsigned int operator()(const unsigned long long int& x) const { 
		return (unsigned int)(x >> 56);
	}
};

void print_chars(unsigned long long int val, unsigned int segmentBytes) { 
	printf("printing keys \t");
	int shift = 56;
	if(segmentBytes > 0) { 
		printf("segment number %d \t", (unsigned int)(val>>((8-segmentBytes)*8)));
		shift-=(segmentBytes*8);
	}
	while(shift>=0) {
		char ch = (char)(val>> shift);
		printf("%c", ch);
		shift-=8;
		if(ch == '\0') printf("*");
	}
	printf(" ");
}

int main(int argc, char** argv) {

	printf("*****************************************************************************\n");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0;

	if(argc!=3) { 
		printf("[DEBUG] Correct usage : ./stringSort [filename] [write_output]\n");
		exit(EXIT_FAILURE);
	}

	char inputFile[500];
	sprintf(inputFile,"%s",argv[1]);
	printf("[DEBUG] InputFile : %s\n", inputFile);

	int writeOutput = atoi(argv[2]);
	printf("[DEBUG] writeOutput : %d\n", writeOutput);

	if (dev < 0) dev = 0;
	if (dev > deviceCount-1) dev = deviceCount - 1;
	cudaSetDevice(dev);

	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
				prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
				(int)prop.minor, (int)prop.clockRate);
	}

	unsigned int maxStringLength = 35;
	unsigned int stringSize = 0;
	unsigned int numElements = 40000000;
	unsigned int MAXBYTES = numElements*maxStringLength;

	thrust::host_vector<unsigned long long int> h_keys;
	thrust::host_vector<unsigned int> h_valIndex;
	thrust::host_vector<unsigned char> h_stringVals;

	struct timespec tread1, tread2; 
	clock_gettime(CLOCK_MONOTONIC, &tread1);

	FILE *fp = fopen(inputFile,"rb");
	if(fp == NULL) { 
		printf("[DEBUG] cannot load %s\n", inputFile);
		exit(EXIT_FAILURE);
	}

	char* INBUF = (char*)malloc((sizeof(char)*MAXBYTES)+2);
	MAXBYTES = fread(INBUF, 1, MAXBYTES, fp);
	printf("[DEBUG] read bytes %u\n", MAXBYTES);
	unsigned int index = 0;
	numElements = 0;
	unsigned int i = 0;
	
	while(i < MAXBYTES) { 
		h_valIndex.push_back(index);
		unsigned int prefixLen = 0;
		unsigned long long int firstKey = 0;
		while(true) {
			unsigned char ch = (unsigned char) INBUF[i];
			if(ch == '\n') {
				h_stringVals.push_back('\0');
				index++;
				i++;
				break;
			}
			if(prefixLen < 8) { 
				firstKey |= (((unsigned long long int)ch) << ((7-prefixLen)*8));
				prefixLen++;
			}
			h_stringVals.push_back(ch);
			index++;
			i++;
		}
		h_keys.push_back(firstKey);
		numElements++;
	}
	stringSize = index;

	clock_gettime(CLOCK_MONOTONIC, &tread2);
	printf("[DEBUG] file read time (ms) : %lf\n", calculateDiff(tread2, tread1));
	printf("[DEBUG] number of elements read %d\n", numElements); 

	struct timespec tsetup1, tsetup2;
	clock_gettime(CLOCK_MONOTONIC, &tsetup1);
	
	thrust::device_vector<unsigned char> d_stringVals = h_stringVals;
	thrust::device_vector<unsigned long long int> d_segment_keys = h_keys;
	thrust::device_vector<unsigned int> d_valIndex = h_valIndex;
	thrust::device_vector<unsigned int> d_static_index(numElements);
	thrust::sequence(d_static_index.begin(), d_static_index.begin() + numElements);
	thrust::device_vector<unsigned int> d_output_valIndex(numElements);

	clock_gettime(CLOCK_MONOTONIC, &tsetup2);
	printf("[TIME] memory setup time (ms) : %lf\n", calculateDiff(tsetup2, tsetup1));
	double totalSetup = calculateDiff(tsetup2, tsetup1);

	unsigned int charPosition = 8;
	unsigned int originalSize = numElements;
	unsigned int segmentBytes = 0;
	unsigned int lastSegmentID = 0;
	unsigned long long int totalKeysSorted = 0;
	
	double totalSortTime = 0.0;
	double totalOtherThrust = 0.0;
	double totalCudaKernel = 0.0;
	unsigned int numSorts = 0;
	struct timespec t1, t2;
	clock_gettime(CLOCK_MONOTONIC, &t1);

	while(true) { 

		struct timespec tsort1, tsort2;
		clock_gettime(CLOCK_MONOTONIC, &tsort1);
		thrust::sort_by_key (
				d_segment_keys.begin(),
				d_segment_keys.begin() + numElements,
				d_valIndex.begin()
		); 
		clock_gettime(CLOCK_MONOTONIC, &tsort2);
		totalSortTime += calculateDiff(tsort2, tsort1);
		numSorts++;
		totalKeysSorted += numElements;

		thrust::device_vector<unsigned long long int> d_segment_keys_out(numElements, 0);

		unsigned char *d_array_stringVals = thrust::raw_pointer_cast(&d_stringVals[0]); 
		unsigned int *d_array_valIndex = thrust::raw_pointer_cast(&d_valIndex[0]);
		unsigned int *d_array_static_index = thrust::raw_pointer_cast(&d_static_index[0]);
		unsigned int *d_array_output_valIndex = thrust::raw_pointer_cast(&d_output_valIndex[0]);

		unsigned long long int *d_array_segment_keys_out = thrust::raw_pointer_cast(&d_segment_keys_out[0]);
		unsigned long long int *d_array_segment_keys = thrust::raw_pointer_cast(&d_segment_keys[0]); 
		
		int numBlocks = 1;
		int numThreadsPerBlock = numElements/numBlocks;

		if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
			numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
			numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
		}
		dim3 grid(numBlocks, 1, 1);
		dim3 threads(numThreadsPerBlock, 1, 1); 

		struct timespec tcuda1, tcuda2;
		clock_gettime(CLOCK_MONOTONIC, &tcuda1);
		cudaThreadSynchronize();
		findSuccessor<<<grid, threads, 0>>>(d_array_stringVals, d_array_segment_keys, d_array_valIndex, 
				d_array_segment_keys_out, numElements, stringSize, charPosition, segmentBytes);
		cudaThreadSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &tcuda2);
		totalCudaKernel += calculateDiff(tcuda2, tcuda1);

		charPosition+=7;

		struct timespec tother1, tother2; 
		clock_gettime(CLOCK_MONOTONIC, &tother1);
		thrust::device_vector<unsigned int> d_temp_vector(numElements);
		thrust::device_vector<unsigned int> d_segment(numElements);
		thrust::device_vector<unsigned int> d_stencil(numElements);
		thrust::device_vector<unsigned int> d_map(numElements);
	
		unsigned int *d_array_temp_vector = thrust::raw_pointer_cast(&d_temp_vector[0]);
		unsigned int *d_array_segment = thrust::raw_pointer_cast(&d_segment[0]);
		unsigned int *d_array_stencil = thrust::raw_pointer_cast(&d_stencil[0]);
		
		thrust::transform(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElements, d_temp_vector.begin(), get_segment_bytes());

#ifdef __PRINT_DEBUG__
		thrust::device_vector<unsigned int>::iterator itr;
		thrust::device_vector<unsigned long long int>::iterator itr2;
		thrust::device_vector<unsigned long long int>::iterator itr3;

		
		itr2 = d_segment_keys_out.begin();
		itr3 = d_segment_keys.begin();
		
		for(itr = d_temp_vector.begin(); itr!=d_temp_vector.end(); ++itr) { 
			cout << *itr << " ";
			print_chars(*itr3, segmentBytes);
			cout << " ";
			print_chars(*itr2, 1);
			++itr2;
			++itr3;
			cout << endl;
		}
#endif
		thrust::inclusive_scan(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_segment.begin());
		clock_gettime(CLOCK_MONOTONIC, &tother2);
		totalOtherThrust += calculateDiff(tother2, tother1);

		clock_gettime(CLOCK_MONOTONIC, &tcuda1);
		cudaThreadSynchronize(); 
	        eliminateSingleton<<<grid, threads, 0>>>(d_array_output_valIndex, d_array_valIndex, d_array_static_index, 
			d_array_temp_vector, d_array_stencil, numElements); 
		cudaThreadSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &tcuda2);
		totalCudaKernel += calculateDiff(tcuda2, tcuda1);

#ifdef __PRINT_DEUBG__
		cout << "Stencil values are ";
		for( itr = d_stencil.begin(); itr != d_stencil.end(); ++itr) { 
			cout << *itr << " ";
		}
		cout << endl;
#endif
		clock_gettime(CLOCK_MONOTONIC, &tother1);
		
		thrust::exclusive_scan(d_stencil.begin(), d_stencil.begin() + numElements, d_map.begin());

		thrust::scatter_if(d_segment.begin(), d_segment.begin() + numElements, d_map.begin(), 
				d_stencil.begin(), d_temp_vector.begin());
		thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_segment.begin()); 

		thrust::scatter_if(d_valIndex.begin(), d_valIndex.begin() + numElements, d_map.begin(), 
				d_stencil.begin(), d_temp_vector.begin());
		thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_valIndex.begin()); 

		thrust::scatter_if(d_static_index.begin(), d_static_index.begin() + numElements, d_map.begin(), 
				d_stencil.begin(), d_temp_vector.begin());
		thrust::copy(d_temp_vector.begin(), d_temp_vector.begin() + numElements, d_static_index.begin()); 

		thrust::scatter_if(d_segment_keys_out.begin(), d_segment_keys_out.begin() + numElements, d_map.begin(), 
				d_stencil.begin(), d_segment_keys.begin());
		thrust::copy(d_segment_keys.begin(), d_segment_keys.begin() + numElements, d_segment_keys_out.begin()); 


		numElements = *(d_map.begin() + numElements - 1) + *(d_stencil.begin() + numElements - 1); 
		if(numElements != 0) { 
			lastSegmentID = *(d_segment.begin() + numElements - 1);
		}
		
		d_temp_vector.clear();
		d_temp_vector.shrink_to_fit();
		
		d_stencil.clear();
		d_stencil.shrink_to_fit();

		d_map.clear();
		d_map.shrink_to_fit();
		
		clock_gettime(CLOCK_MONOTONIC, &tother2);
		totalOtherThrust += calculateDiff(tother2, tother1);
		
		if(numElements == 0) {
			thrust::copy(d_output_valIndex.begin(), d_output_valIndex.begin() + originalSize, h_valIndex.begin());
			break;
		}
	
		segmentBytes = (int) ceil(((float)(log2((float)lastSegmentID+2))*1.0)/8.0);
		charPosition-=(segmentBytes-1);

#ifdef __PRINT_DEBUG__
		printf("[DEBUG] numElements %d, charPosition %d, lastSegmentID %d, segmentBytes %d\n", numElements, 
				charPosition, lastSegmentID, segmentBytes );
#endif

		int numBlocks1 = 1;
		int numThreadsPerBlock1 = numElements/numBlocks1;

		if(numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) { 
			numBlocks1 = (int)ceil(numThreadsPerBlock1/(float)MAX_THREADS_PER_BLOCK);
			numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
		}
		dim3 grid1(numBlocks1, 1, 1);
		dim3 threads1(numThreadsPerBlock1, 1, 1); 

		clock_gettime(CLOCK_MONOTONIC, &tcuda1);
		cudaThreadSynchronize();
		rearrangeSegMCU<<<grid1, threads1, 0>>>(d_array_segment_keys, d_array_segment_keys_out, d_array_segment, 
				segmentBytes, numElements);
		cudaThreadSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &tcuda2);
		totalCudaKernel += calculateDiff(tcuda2, tcuda1);

#ifdef __PRINT_DEBUG__		
		printf("---------- new keys are --------\n");
		itr2 = d_segment_keys.begin();
		unsigned int ct = 0;
		for( ct = 0; ct < numElements; ct++ ) { 
			print_chars(*itr2, segmentBytes);
			printf("\n");
			++itr2;
		}
		printf("----\n");
#endif
	}
	clock_gettime(CLOCK_MONOTONIC, &t2);

	printf("[TIME] Total (ms) : %lf, Sort (ms) : %lf, Iterations : %d, Thrust Other (ms) : %lf, Cuda Kernels : %lf, total calc : %lf\n", 
			calculateDiff(t2, t1), totalSortTime, numSorts, totalOtherThrust, totalCudaKernel, 
			totalSetup+totalSortTime+totalOtherThrust+totalCudaKernel);
	printf("[INFO] Throughput : %lf MKeys/sec\n", (totalKeysSorted / 1000000.0)/(calculateDiff(t2, t1) / 1000.0));

	if(writeOutput == 1) {
		struct timespec tout1, tout2;
		clock_gettime(CLOCK_MONOTONIC, &tout1);
		printSortedOutput(h_valIndex, h_stringVals, originalSize, stringSize, inputFile);
		clock_gettime(CLOCK_MONOTONIC, &tout2);
		printf("[DEBUG] file write time (ms) : %lf\n", calculateDiff(tout2, tout1));
	}
	printf("*****************************************************************************\n");
	return 0;
}
