/*
This program accepts a folder of .dta files and percentage data 
to be retained. Output is generated in the form of a .ms2 file


Copyright (C) Muaaz Gul Awan and Fahad Saeed 
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/



#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <sstream>
#include <algorithm>
#include <unistd.h>
#include <string>
#include "specFile.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include <thrust/scan.h>
#include <time.h>
#include <ctime>
#include <iomanip> 
#include<random>
#include <thrust/execution_policy.h>
#include <fstream>
#define elesPerBucket 20
#define sampleRateSo 10

using namespace std;

template <typename mType>
struct dataArrays{
	mType* dataList;
	int *prefixArray;
};

spectrum stringToSpectrum(string input);

float* headLineTokenizer(string line);

int* fileNameTokenizer(string line);

void ms2FileWriter (vector< specFile> listOfFiles, int *binSpecs, vector< dataArrays<int> >  DataVec, int arrayPerPass, int passes, int maxSize);

vector<string> listFilesOfFolder (string dir);

template <class type> 
__device__ void  swapD (type &a, type &b);

template <class type> 

__device__ void insertionSort(type *input, int begin, int end);

__device__ void mergesort (int beg, int *a, int *b, long num);

__device__ void getMinMax(int input[], int beginPtr, int endPtr, int *ret);
__global__ void getNumOfBuckets(int *prefixSumArray, int *numOfBucketsArray, int offset, int arraysPerPass);

template <typename mType>
__device__ void getSplitters (mType *data, mType *splittersArray, int sample[], int beginPtr, int endPtr, int arraySize, int *prefixBucketsArray, int offset);

template <typename mType>
__global__ void splitterKer(mType *data, mType *splittersArray, int *prefixSizeArray, int *prefixBucketsArray, int offset, int arraysPerPass);

template <typename mType>
__device__ void getBuckets(mType *input, mType *splitters, int beginPtr, int endPtr, int *bucketsSize, int *prefixBucketsArray, int offset, int maxSize, int maxBucks, int arraysPerPass);

template <typename mType>
__global__ void bucketKernel(mType *data, mType *splittersArray, int *prefixSizeArray, int *prefixBucketsArray, int *bucketSizes, int offset, int maxSize, int maxBucks, int arraysPerPass);

template <typename mType>	
__global__ void sortBuckets(mType *buckets, int *bucketsSize, int *prefixBucketsArray, int *prefixSizeArray, int offset, int arraysPerPass);

__global__ void rangeKernel(int *intens, int *scanSizes, unsigned long int *d_localRanges, int *maxAvgs, int arraysPerPass);

__device__ void quantization(int *localSpec, int size, int quanta, int *quantizedSpec, int *maxAvgs, int *quantaSizes, int *quantaBegPtr);

__device__ bool notInArray(int num,int *arr, int size);

__device__ void randomSample(int peaksReq, int *randsArray, int max);

__device__ void sampling(int *sampledSpec, int *quantizedSpec, int size, int *quantaSizes, int *quantaBegPtr, int peaksReq, int quanta, int *sampledSpecSize);

__global__ void classKernel(int *intens, int *sizesScanned, unsigned long int *localRanges, int *globalRange, int *maxAvgs, int *sampledSpecSize, int *d_sampleRate, int arraysPerPass);

template <typename mType>		
int* gpuArraySort(vector< dataArrays<mType> > newData, int numOfStreams, long long int totArrays, int maxSize, int *h_sampleRate);

vector<specFile> retrieveSpectraFromFiles(vector<string> listOfFiles, string dir, int start, int end, int countX);

int getTotalPeaks(vector<specFile> listOfFiles);

int getMaxSpecSize(vector<specFile> listofSpecFiles);


 

int main (int argc, char* argv[])
{
 

 string dirn = argv[1];
 //int *d_sampleRate;
 int *sampleRate =new int[1];
 int timesX = stoi(argv[3]);
  *sampleRate = stoi(argv[2]);
  vector<string> listOfFileNames = listFilesOfFolder(dirn);
  vector<specFile> listofSpecFiles = retrieveSpectraFromFiles(listOfFileNames, dirn, 0, listOfFileNames.size(),timesX);
 int totalPeaks = getTotalPeaks(listofSpecFiles);
  int *h_testPtr = new int[totalPeaks];
  int *h_resultSizes = new int [listofSpecFiles.size()];
  int *h_globalRange = new int[1];
int maxSpecSize = getMaxSpecSize(listofSpecFiles);
 
  thrust::device_vector<int> d_intensity;
  thrust::device_vector<int> d_intensityNew;
  thrust::host_vector<int> h_keys(totalPeaks);
  thrust::device_vector<int> d_keys;
  int *d_sizes = new int[listofSpecFiles.size()];
 // long int index = 0;
  int *after = new int[listOfFileNames.size()];
  int *before = new int[listOfFileNames.size()];
  size_t f, t, size_heap, size_stack;

  float *header = headLineTokenizer(listofSpecFiles.at(0).headerLine);
  int *namer = fileNameTokenizer(listofSpecFiles.at(0).fileName);
  
cudaSetDevice(0);
cudaMemGetInfo(&f,&t);
cudaDeviceSetLimit(cudaLimitStackSize, 102400);
cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);

  

 long long int totArrays = listofSpecFiles.size();
 //long int estMemReq = (5*totArrays*sizeof(unsigned long int)+ totArrays*maxSpecSize*sizeof(int));
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
//	cout <<"memReq:"<<estMemReq<<endl;
	//long int GPUMem = prop.totalGlobalMem*0.6;
	int estPasses = 4;//ceil((double)estMemReq/(GPUMem));
	int arraysPerPass = totArrays/estPasses;
	int arraysLastPass = arraysPerPass + totArrays%estPasses;
	
   vector< dataArrays<int> > DataVec;
   //int sizesTemp[4];
for(int p = 0; p < estPasses-1; p++)
{
   dataArrays<int> data;//data.prefixArray = 
   cudaMallocHost((void**) &data.prefixArray, (1+arraysPerPass)*sizeof(int));//new int[numOfArrays+1]; //exclusive prefix scan
   cudaMallocHost((void**) &data.dataList, (arraysPerPass*maxSpecSize)*sizeof(int));
   
  int prefixSum = 0, q = 0, preIndex=0;
  for(int i = p*arraysPerPass; i < (p+1)*arraysPerPass; i++){
      data.prefixArray[preIndex] = prefixSum;
	  for(int j = 0; j < listofSpecFiles.at(i).listVals.size(); j++){
         data.dataList[q] = listofSpecFiles.at(i).listVals.at(j).intensity;
      
		q++;
      }
	  prefixSum += listofSpecFiles.at(i).listVals.size();
	  preIndex++;
  }
   data.prefixArray[preIndex] = prefixSum;
   DataVec.push_back(data);
   
}

//for the last pass
dataArrays<int> data;//data.prefixArray = 
   cudaMallocHost((void**) &data.prefixArray, (1+arraysPerPass)*sizeof(int));//new int[numOfArrays+1]; //exclusive prefix scan
   cudaMallocHost((void**) &data.dataList, (arraysPerPass*maxSpecSize)*sizeof(int));
   
  int prefixSum = 0, q = 0, preIndex=0;
  for(int i = (estPasses-1)*arraysPerPass; i < (estPasses-1)*arraysPerPass + arraysLastPass; i++){
      data.prefixArray[preIndex] = prefixSum;
	  for(int j = 0; j < listofSpecFiles.at(i).listVals.size(); j++){
         data.dataList[q] = listofSpecFiles.at(i).listVals.at(j).intensity;
        // h_keys[index] = i;
        // index++;
		q++;
      }
	  prefixSum += listofSpecFiles.at(i).listVals.size();
	  preIndex++;
  }
  //sizesTemp[3] = q;
   data.prefixArray[preIndex] = prefixSum;
   DataVec.push_back(data);

	for(int i = 0; i < estPasses-1; i++){
		//cout <<"for Sizes:"<<sizesTemp[i]<<"prefix:"<<DataVec.at(i).prefixArray[arraysPerPass]<<endl;
		
		}
	//	cout <<"for Sizes:"<<sizesTemp[3]<<"prefix:"<<DataVec.at(3).prefixArray[arraysLastPass]<<endl;
		
  // cout<<"total spectra::::::"<<totArrays<<endl;
   
  //  cout <<"*********file reading done calling sort**********"<<endl;
   int *h_result = gpuArraySort<int>(DataVec, 1, listofSpecFiles.size(), maxSpecSize, sampleRate );
	
  ///writing takes time so for now not writing.
	//  ms2FileWriter (listofSpecFiles, h_result, DataVec[0].prefixArray);
    ms2FileWriter (listofSpecFiles, h_result, DataVec, arraysPerPass, estPasses, maxSpecSize);

  
 

 // cudaFree(d_localRanges);
 // cudaFree(d_maxAvgs);
 // cudaFree(d_globalRange);
  //cudaFree(d_sampledSpecSizes);

  return 0;

}







//converts string to spectrum class
spectrum stringToSpectrum(string input) {
	spectrum newSpectrum;
	int point = 0;
	point = input.find(" ");
	newSpectrum.m_z = stof(input.substr(0, point));
	newSpectrum.intensity = stof(input.substr(point + 1, input.find('\0')));


	return newSpectrum;
}

//tokenizes the header line 
float* headLineTokenizer(string line)
{
	float *headerFile = new float[2];
	
	headerFile[0] = stof(line.substr(0, line.find(" ")));
	headerFile[1] = stof(line.substr(line.find(" ")+1, line.find('\0')));
	
	return headerFile;
}

//tokenizes the file name
int* fileNameTokenizer(string line)
{
	int *fileNameTokens = new int[1];
	
	fileNameTokens[0] = stoi(line.substr(line.find(".")+1, line.find(".5", line.find(".")+1, 1)-1));
	
	return fileNameTokens;
}

//writes to .ms2 file
void ms2FileWriter (vector< specFile> listOfFiles, int *binSpecs, vector< dataArrays<int> >  DataVec, int arrayPerPass, int passes, int maxSize)
{
	
	ofstream myFileWriter;
	string targetPath = "reducedSpec.ms2";
	myFileWriter.open(targetPath.c_str());
	for(int p = 0; p < passes; p++)
	{
	for(int i = 0; i < arrayPerPass; i++)
	{
		float *header = headLineTokenizer(listOfFiles.at(arrayPerPass*p + i).headerLine);
		int *namer = fileNameTokenizer(listOfFiles.at(arrayPerPass*p + i).fileName);
		float prec_mass_peptide = (float) ((header[0]) + (header[1] - 1)) / (header[1]);
		int specStart = DataVec.at(p).prefixArray[i];
		int specEnd = DataVec.at(p).prefixArray[i+1];
		int specSize = specEnd - specStart;
		myFileWriter << setprecision(8);
		myFileWriter << "S\t"<<namer[0]<<"\t"<<namer[0]<<"\t"<<prec_mass_peptide<<endl;
		myFileWriter << "Z\t"<<header[1]<<"\t"<<header[0]<<endl;
				
			
		for(int j = 0; j < specSize; j++)
		{
			if(binSpecs[p*arrayPerPass*maxSize + specStart + j] == 1)
			{
				myFileWriter << listOfFiles.at(arrayPerPass*p + i).listVals.at(j).m_z<<"\t"<<listOfFiles.at(arrayPerPass*p + i).listVals.at(j).intensity<<endl;
			}
		}
		delete[] header;
		delete[] namer;
	}
	}
}

//lists file for folder
vector<string> listFilesOfFolder (string dir)
{
	
	vector<string> listOfFiles;
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error opening the file " << dir << endl;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if(string(dirp->d_name).compare("..") == 0 || string(dirp->d_name).compare(".") == 0)
			continue;
		else
        listOfFiles.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return listOfFiles;
}



//swap function for Insertion sort
template <class type> __device__ void  swapD (type &a, type &b)

{
    type temp;
        temp=a;
	a=b;
        b=temp;
}

//insertion sort
template <class type> __device__ void insertionSort(type *input, int begin, int end){
	int i, j; 
	for (i = begin+1; i < end; i++) {
		j = i;
	while (j > begin && input[j - 1] > input[j]) {
		swapD(input[j], input[j-1]);
		j--;
		}
	}
}


// mergesort code from stackoverflow
__device__ void mergesort (int beg, int *a, int *b, long num)
{
    int rght, rend;
    int i,j,m;

    for (int k=1; k < num; k *= 2 ) {       
        for (int left=beg+0; left+k < num; left += k*2 ) {
            rght = left + k;        
            rend = rght + k;
            if (rend > num) rend = num; 
            m = left; i = left; j = rght; 
            while (i < rght && j < rend) { 
                if (a[i] <= a[j]) {         
                    b[m] = a[i]; i++;
                } else {
                    b[m] = a[j]; j++;
                }
                m++;
            }
            while (i < rght) { 
                b[m]=a[i]; 
                i++; m++;
            }
            while (j < rend) { 
                b[m]=a[j]; 
                j++; m++;
            }
            for (m=left; m < rend; m++) { 
                a[m] = b[m]; 
            }
        }
    }
}




__device__ void getMinMax(int input[], int beginPtr, int endPtr, int *ret){
          int min = input[beginPtr];
          int max = 0;
          for(int i = beginPtr; i < endPtr; i++){
              if(min > input[i])
                  min = input[i];
              if (max < input[i])
                  max = input[i];     
            }

     ret[0] = min;
     ret[1] = max;

}


//kernel for obtaining num of buckets for each array
__global__ void getNumOfBuckets(int *prefixSumArray, int *numOfBucketsArray, int offset, int arraysPerPass){
	int id = offset + blockIdx.x; 
	
	if(id < arraysPerPass)
		numOfBucketsArray[id] = (prefixSumArray[id+1] - prefixSumArray[id])/elesPerBucket;
}

template <typename mType>
__device__ void getSplitters (mType *data, mType *splittersArray, int sample[], int beginPtr, int endPtr, int arraySize, int *prefixBucketsArray, int offset){
           int SAMPLED = (sampleRateSo*arraySize)/100;           
		   mType *mySamples = &sample[SAMPLED];
		   mType *mySamples_temp = &mySamples[SAMPLED];
		   int bid = blockIdx.x + offset;
		   
			// calculating samples for this array
			int numOfSamples = ((float)sampleRateSo/100)*(arraySize);
			//calculating the number of buckets for this array
			int numOfBuckets = (bid == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[bid] - prefixBucketsArray[bid-1]);
			
            for(int i = 0; i < numOfSamples; i++)
	           mySamples[i] = data[beginPtr+sample[i]];

	         mergesort (0, mySamples, mySamples_temp, numOfSamples);
	        //calculate splitter index for this array 
            int splitterIndex = ((bid == 0)? 1 : (prefixBucketsArray[bid-1]+1)+1); //the other plus one is for leaving space for smallest splitter(added later)
            int splittersSize=0;
	        for(int i = (numOfSamples)/(numOfBuckets); splittersSize < numOfBuckets-1; i +=(numOfSamples)/(numOfBuckets)){
                 splittersArray[splitterIndex] = mySamples[i];
                 splitterIndex++;
                 splittersSize++;
             }
			int bits = 8*sizeof(mType);
            mType min = -(1 << (bits-1));
            mType max = (1 << (bits - 1)) - 1;
            splittersArray[((bid == 0)? 0 : (prefixBucketsArray[bid-1]+1))] = min;//smaller than min value
            splittersArray[((bid == 0)? prefixBucketsArray[0] : (prefixBucketsArray[bid]))] = max;//larger than max value;
      
}

//kernel for obtaining splitters
template <typename mType>
__global__ void splitterKer(mType *data, mType *splittersArray, int *prefixSizeArray, int *prefixBucketsArray, int offset, int arraysPerPass){
          int bid = blockIdx.x + offset;
		  if(bid < arraysPerPass){
             int id = offset + blockIdx.x;
	         extern __shared__ int majorArray[];
	         int *sampleSh = majorArray;
			 int arraySize = prefixSizeArray[id+1] - prefixSizeArray[id];
			// calculating samples for this array
			int numOfSamples = ((float)sampleRateSo/100)*(arraySize);
            int max = arraySize;
            int  sam = numOfSamples;
            int stride = max/sam;
	        int sampleVal = 0;
            for( int i = 0; i < numOfSamples; i++)
            {
               sampleSh[i] = sampleVal;
               sampleVal += stride; 
            }
			 
	        getSplitters(data, splittersArray, sampleSh, prefixSizeArray[id], prefixSizeArray[id+1], prefixSizeArray[id+1] - prefixSizeArray[id], prefixBucketsArray, offset);

           }
        }

		
template <typename mType>
__device__ void getBuckets(mType *input, mType *splitters, int beginPtr, int endPtr, int *bucketsSize, int *prefixBucketsArray, int offset, int maxSize, int maxBucks, int arraysPerPass){
     int bid = blockIdx.x + offset;
	 int numOfBuckets = (bid == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[bid] - prefixBucketsArray[bid-1]);
        
	if(blockIdx.x < arraysPerPass && threadIdx.x < numOfBuckets){
	  int *localSizes = &splitters[maxBucks+2];
	  int id = threadIdx.x;
	  int sizeOffset = (bid == 0) ? (0+threadIdx.x) : (prefixBucketsArray[bid-1] + threadIdx.x);  //blockIdx.x*BUCKETS+threadIdx.x;
	 // int sizeOffsetBlock = (bid == 0) ? (0) : (prefixBucketsArray[bid-1]);
      int bucketSizeOff = sizeOffset+1;
	  //__shared__ int my
      mType myBucket[750]; 
      int indexSum=0;
      localSizes[threadIdx.x] = 0;
	  
     for(int i = beginPtr; i < endPtr ; i++)
	 {
         if(input[i] > splitters[id] && input[i] <= splitters[id+1])
		 {
			myBucket[localSizes[threadIdx.x]] = input[i];
			localSizes[threadIdx.x]++;
         }
     }
       
    __syncthreads();
         //prefix sum for bucket sizes of current array
     for(int j = 0; j < threadIdx.x; j++)
        indexSum += localSizes[j];

     bucketsSize[bucketSizeOff] = localSizes[threadIdx.x];
         //writing back current buckt back to the input memory
	 for(int i = 0; i < localSizes[threadIdx.x]; i++)
             input[indexSum+beginPtr+i] = myBucket[i];
		
	}
     

}
		
//kernel for obtaining buckets
template <typename mType>
__global__ void bucketKernel(mType *data, mType *splittersArray, int *prefixSizeArray, int *prefixBucketsArray, int *bucketSizes, int offset, int maxSize, int maxBucks, int arraysPerPass){
    
	int numOfBuckets = (offset + blockIdx.x == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[offset + blockIdx.x] - prefixBucketsArray[(blockIdx.x + offset)-1]);
        
	if(blockIdx.x < arraysPerPass && threadIdx.x < numOfBuckets){
        bucketSizes[0] = 0;
		extern __shared__ int majorArray[];
		int bid = offset + blockIdx.x;
        int arrBegin = prefixSizeArray[bid];
        int arrEnd = prefixSizeArray[bid+1];
		    
        int splitterIndexSt = ((bid == 0)? 0 : (prefixBucketsArray[bid-1]+1));//blockIdx.x*(BUCKETS+1);
        mType *splitters = majorArray;
 
        //int j = 0;
		
	   	if(threadIdx.x == numOfBuckets - 1)
			{
             splitters[threadIdx.x] = splittersArray[splitterIndexSt+threadIdx.x];
             splitters[threadIdx.x+1] = splittersArray[splitterIndexSt+threadIdx.x+1];
		    } 	
            else	
           	 splitters[threadIdx.x] = splittersArray[splitterIndexSt+threadIdx.x];	
		
	   
		 __syncthreads();
	    getBuckets(data, splitters, arrBegin, arrEnd, bucketSizes, prefixBucketsArray, offset, maxSize, maxBucks, arraysPerPass);

	}
}		
		
	
		
		
		
//sorting kernel	
template <typename mType>	
__global__ void sortBuckets(mType *buckets, int *bucketsSize, int *prefixBucketsArray, int *prefixSizeArray, int offset, int arraysPerPass){
	int bid = blockIdx.x + offset;
	int numOfBuckets = (bid == 0) ? prefixBucketsArray[0] : (prefixBucketsArray[bid] - prefixBucketsArray[bid-1]);
     
	
       if(bid < arraysPerPass && threadIdx.x < numOfBuckets){
		int sizeOffset = (bid == 0) ? (0+threadIdx.x) : (prefixBucketsArray[bid-1] + threadIdx.x); 
        int sizeOffsetBlock = (bid == 0) ? (0) : (prefixBucketsArray[bid-1]);
       
      //  int tid = threadIdx.x;
		//int arraySize = prefixSizeArray[bid+1] - prefixSizeArray[bid];
	   
        int indexSum = 0;
    

        //prefix sum for bucket sizes of current array
        
     	  for(int j = 0; j < threadIdx.x; j++)
              indexSum += bucketsSize[sizeOffsetBlock+j+1];

 
          insertionSort(buckets, prefixSizeArray[bid] + indexSum,prefixSizeArray[bid] + indexSum + bucketsSize[sizeOffset+1]);
		
   		__syncthreads();
 
}


}

__global__ void rangeKernel(int *intens, int *scanSizes, unsigned long int *d_localRanges, int *maxAvgs, int arraysPerPass){
       int totalSpecs = arraysPerPass;
       int prefixSum = 0;
       int min3[10];
       int max3[10];
       int localRange =0, minAvg=0, maxAvg=0;
       int size = 0;
   
          size = scanSizes[blockIdx.x+1] - scanSizes[blockIdx.x];
          prefixSum = scanSizes[blockIdx.x];

       

       
       if(blockIdx.x < totalSpecs){
       
       for(int i = prefixSum, index=0; i < prefixSum+10;i++,index++)
            min3[index] = intens[i]; 
       for(int i = prefixSum+(size-1), index=0; i > (prefixSum+size)-11;i--,index++)
            max3[index] = intens[i]; 
       for(int i = 0; i < 10; i++){
           minAvg += min3[i];
           maxAvg += max3[i];
        }
        
      minAvg = minAvg/10;
      maxAvg = maxAvg/10;

      maxAvgs[blockIdx.x] = maxAvg;
      localRange = maxAvg - minAvg;
      d_localRanges[blockIdx.x] = localRange; 
       }
}


__device__ void quantization(int *localSpec, int size, int quanta, int *quantizedSpec, int *maxAvgs, int *quantaSizes, int *quantaBegPtr){
           float jumpInc = (float) 1/quanta, jump = 0, jumpLag = 0;
           float refVal = maxAvgs[blockIdx.x];
           int indexPtr = 0;
           int quantumSize = 0;
           for(int i = 0; i < quanta; i++){ 
                quantumSize = 0;
                if(i == 0){
               jumpLag = 0;
               jump = jumpInc;
                }

               for(int j = 0; j < size; j++){
                   if( i == quanta -1){
               // we add only the indexes of the intensities in the final array
                      if(localSpec[j] >= (refVal*jumpLag)){
                         quantizedSpec[indexPtr] = j;
                         indexPtr++;
                         quantumSize++;
                       }
                      else{}
           
                    }
                   else{
                      if(localSpec[j] >= (refVal*jumpLag) && (localSpec[j]< (refVal*jump))){
                         quantizedSpec[indexPtr] = j;
                         indexPtr++;   
                         quantumSize++;  
                      }
                   }
                }
                quantaSizes[i] = quantumSize;
                quantaBegPtr[i] = indexPtr-quantumSize; 
                jump = jump + jumpInc;
                jumpLag += jumpInc;


           }



    }

__device__ bool notInArray(int num,int *arr, int size){
                bool var = true;
                for(int i = 0; i < size; i++){
                  if(arr[i] == num)
                     var = false;
               }
       return var;

    }

// generate random samples on device
__device__ void randomSample(int peaksReq, int *randsArray, int max){
             curandState_t state;
             curand_init(1234, blockIdx.x, 0, &state);
             int count = 0;
                        
             while(count < peaksReq){
                float randF = curand_uniform(&(state));
                randF *= ((max-1) - 0 + 0.999999);
                randF += 0;
                int randI = (int)truncf(randF);
                if( notInArray(randI, randsArray, count)){
                    randsArray[count] = randI;
                    count++;
                   }
                 }

}

__device__ void sampling(int *sampledSpec, int *quantizedSpec, int size, int *quantaSizes, int *quantaBegPtr, int peaksReq, int quanta, int *sampledSpecSize){
                           
        int tolPeaks = 5;
        int *rands = new int[peaksReq];
        if( (quantaSizes[quanta-1] >= (peaksReq - tolPeaks)  && (quantaSizes[quanta-1] <= (peaksReq+tolPeaks)))){
           for(int i = quantaBegPtr[quanta-1]; i <(quantaBegPtr[quanta-1]+ quantaSizes[quanta-1]); i++){
            sampledSpec[quantizedSpec[i]] = 1;
         
           }
          }

         else if( (quantaSizes[quanta-1] > (peaksReq + tolPeaks))){
              //generate random nums array
            int max = quantaSizes[quanta-1];
               randomSample (peaksReq, rands,max);
               
               for(int i = 0; i < peaksReq; i++){
                 sampledSpec[quantizedSpec[quantaBegPtr[quanta-1]+rands[i]]] = 1;
                
               }
         
          }

         else{
              while((peaksReq > quantaSizes[quanta-1]) || ((quantaSizes[quanta-1] >= (peaksReq - tolPeaks)) && (quantaSizes[quanta-1] <= (peaksReq + tolPeaks)))){
                for(int i = quantaBegPtr[quanta-1]; i < (quantaBegPtr[quanta-1]+quantaSizes[quanta-1]);i++){
                     sampledSpec[quantizedSpec[i]] = 1;
                  

                   }
                 peaksReq = peaksReq - quantaSizes[quanta-1];
                 quanta--;
                }
             int max = quantaSizes[quanta-1];
                randomSample (peaksReq, rands,max);
                
                for(int i = 0; i < peaksReq; i++){
                 sampledSpec[quantizedSpec[quantaBegPtr[quanta-1]+rands[i]]] = 1;
                 
              }

           }
    delete[] rands;
   }

    

    


__global__ void classKernel(int *intens, int *sizesScanned, unsigned long int *localRanges, int *globalRange, int *maxAvgs, int *sampledSpecSize, int *d_sampleRate, int arraysPerPass){
        int totalSpecs = arraysPerPass;
        int sample_rate = *d_sampleRate;
        int size = 0;
        int prefixSum =0;
       
           size = sizesScanned[blockIdx.x+1]-sizesScanned[blockIdx.x];
           prefixSum = sizesScanned[blockIdx.x];

        
        if( blockIdx.x < totalSpecs){
          float relativeRange = ((float)localRanges[blockIdx.x]/(*globalRange))*100;
          int peaksReq = ((float)sample_rate/100)*size;
          int *localSpec = new int[size];
          int *quantizedSpec = new int[size]; 
          int *sampledSpec = new int[size];
          int *quantaSizes = new int[11];
          int *quantaBegPtr = new int[11];

          for(int i = 0; i < size; i++)
             sampledSpec[i] = 0;       

          for(int i = 0; i < size; i++)
             localSpec[i] = intens[prefixSum+i];
        

           if(relativeRange < 25){
              quantization (localSpec, size, 5, quantizedSpec, maxAvgs, quantaSizes,quantaBegPtr);
              sampling (sampledSpec, quantizedSpec, size, quantaSizes,quantaBegPtr, peaksReq, 5, sampledSpecSize);
            }
            else if ((relativeRange >= 25) && (relativeRange < 50)){
              quantization (localSpec, size, 7, quantizedSpec, maxAvgs, quantaSizes, quantaBegPtr);
             sampling (sampledSpec, quantizedSpec, size, quantaSizes,quantaBegPtr, peaksReq, 7, sampledSpecSize);

            }  
            
            else if ((relativeRange >= 50) && (relativeRange < 75)){
              quantization (localSpec, size, 9, quantizedSpec, maxAvgs, quantaSizes, quantaBegPtr);
             sampling (sampledSpec, quantizedSpec, size, quantaSizes,quantaBegPtr, peaksReq, 9, sampledSpecSize);

            }   
            
            else if (relativeRange >= 75){
              quantization (localSpec, size, 11, quantizedSpec, maxAvgs, quantaSizes, quantaBegPtr);
             sampling (sampledSpec, quantizedSpec, size, quantaSizes,quantaBegPtr, peaksReq, 11, sampledSpecSize);

            }

       // int indexPtr = 0;

		for(int i = 0; i < size; i++)
		{
			intens[prefixSum+i] = sampledSpec[i];
		}
	

      
   
       delete[] localSpec;
       delete[] quantizedSpec;
       delete[] quantaSizes;
       delete[] quantaBegPtr;
       delete[] sampledSpec;
   }
}


template <typename mType>		
int* gpuArraySort(vector< dataArrays<mType> > newData, int numOfStreams, long long int totArrays, int maxSize, int *h_sampleRate  ){
	//long int estMemReq = (5*totArrays*sizeof(unsigned long int)+ totArrays*maxSize*sizeof(int));
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
//	long int GPUMem = prop.totalGlobalMem*0.6;
	int estPasses = 4;
	long long int arraysPerPass = totArrays/estPasses;
	long long int arraysPerStream = arraysPerPass/numOfStreams;

	int *h_result = new int[totArrays*maxSize];
	int *h_offsetArray = new int[numOfStreams];
	int *d_prefixSum, *d_numOfBuckets;
    mType *d_inputData, *d_splitters, *d_bucketSizes;
	mType *h_sortedData;// = new mType[newData.prefixArray[totArrays]];
	int SAMPLED = (sampleRateSo*maxSize)/100;
    int maxBuckets = (maxSize/elesPerBucket);
	 size_t size_heap, size_stack;
    //setting stack size limit
    cudaDeviceSetLimit(cudaLimitStackSize,20480);
    cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
    cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
   int *h_localRange, *h_maxAvgs;
	cudaMallocHost((void**)&h_sortedData, sizeof(mType)*totArrays*4*maxSize);//newData.prefixArray[totArrays]);
    cudaMallocHost((void**)&h_localRange, sizeof(int)*totArrays);//newData.prefixArray[totArrays]);
	cudaMallocHost((void**)&h_maxAvgs, sizeof(int)*totArrays);
    cudaMalloc((void**) &d_prefixSum, (arraysPerPass+1)*sizeof(int));
    cudaMalloc((void**) &d_numOfBuckets, (arraysPerPass+1)*sizeof(int));
    cudaMalloc((void**) &d_inputData, sizeof(mType)*arraysPerPass*maxSize);//prefixSum[arraysPerPass]*
	
	cudaMalloc((void**) &d_splitters, (arraysPerPass*maxBuckets+2*arraysPerPass)*sizeof(mType));
    cudaMalloc((void**) &d_bucketSizes, (arraysPerPass*maxBuckets)*sizeof(int)); 
	
	
	unsigned long int *d_localRanges;
	int *d_maxAvgs, *d_globalRange, *d_sampledSpecSizes, *d_sampleRate;
	
    cudaMalloc((void**) &d_localRanges, sizeof(unsigned long int)*arraysPerPass);
    cudaMalloc((void**) &d_maxAvgs, sizeof(int)*arraysPerPass);
    cudaMalloc((void**) &d_globalRange, sizeof(int));
	cudaMalloc((void**) &d_sampleRate, sizeof(int));
	cudaMalloc((void**) &d_sampledSpecSizes, sizeof(int));
	 
	cudaStream_t stream[numOfStreams];
	for (int i = 0; i < numOfStreams; ++i)
          cudaStreamCreate(&stream[i]) ;
	
	//creating events
	cudaEvent_t start, stop, stream1, stream2, stream3, stream4, copyStart, copyStop, synch[numOfStreams];
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&copyStart);
	cudaEventCreate(&copyStop);
	cudaEventCreate(&stream1);
	cudaEventCreate(&stream2);
	cudaEventCreate(&stream3);
	cudaEventCreate(&stream4);
	
	for(int i = 0; i < numOfStreams; i++)
		cudaEventCreate(&synch[i]);
	
   
   //  int errCout = 0;
   float copyTime = 0;
   
   thrust::device_ptr<int> prefixNumBuckets = thrust::device_pointer_cast(d_numOfBuckets);
	 
    unsigned long int *range = new unsigned long int[1];
	unsigned long int *temp_range = new unsigned long int[estPasses];
	  cudaEventRecord(start); 
   for(int p = 0; p <estPasses; p++){

   //copying prefixSums to Device
   
     for(int i = 0; i < numOfStreams -1; i++)   
        cudaMemcpyAsync(&d_prefixSum[i*arraysPerStream], &newData[p].prefixArray[i*arraysPerStream], sizeof(int)*(arraysPerStream), cudaMemcpyHostToDevice, stream[i]);
     cudaMemcpyAsync(&d_prefixSum[(numOfStreams-1)*arraysPerStream], &newData[p].prefixArray[(numOfStreams-1)*arraysPerStream],
	 sizeof(int)*(arraysPerStream+1), cudaMemcpyHostToDevice, stream[numOfStreams-1]);
    
	//copying input data to device 
	
	  for(int i = 0; i < numOfStreams; i++)   
        cudaMemcpyAsync(&d_inputData[newData[p].prefixArray[i*arraysPerStream]], &newData[p].dataList[newData[p].prefixArray[i*arraysPerStream]],
	    sizeof(int)*(newData[p].prefixArray[(i+1)*arraysPerStream]-newData[p].prefixArray[i*arraysPerStream]), cudaMemcpyHostToDevice, stream[i]);
	
   //clculating buckets on GPU
   int offset = 0;
   for(int i = 0; i < numOfStreams; i++){
	   offset = i*arraysPerStream;
       getNumOfBuckets<<<arraysPerStream, 1, 0, stream[i]>>>(d_prefixSum, d_numOfBuckets, offset,arraysPerPass);
   }
 


   for(int i = 0; i < numOfStreams; i++){
	   if( i > 0)
	    cudaStreamWaitEvent(stream[i], synch[i-1], 0);
	
   thrust::inclusive_scan(thrust::cuda::par.on(stream[i]), prefixNumBuckets+(i*arraysPerStream)+(i==0?0:-1), 
  prefixNumBuckets+(i+1)*arraysPerStream , prefixNumBuckets+(i*arraysPerStream)+(i==0?0:-1));
      cudaEventRecord(synch[i],stream[i]);
	 
   }
   
	
  
   // cout<< "**** Generating Splitters ****" << endl;
      offset = 0;
    for(int i = 0; i < numOfStreams; i++){  
	   offset = i*arraysPerStream;
       splitterKer<<<arraysPerStream, 1, 3*SAMPLED*sizeof(int), stream[i]>>>(d_inputData, d_splitters, d_prefixSum, d_numOfBuckets, offset,arraysPerPass);
	   
	}
	

	//cout<< "**** Generating Buckets ****" << endl;
	
    offset = 0;
      for(int i = 0; i < numOfStreams; i++){  
	   offset = i*arraysPerStream;
       bucketKernel<<<arraysPerStream, maxBuckets, (2*maxBuckets + 2)*sizeof(int), stream[i]>>>(d_inputData, d_splitters, d_prefixSum, d_numOfBuckets, d_bucketSizes, offset, maxSize,maxBuckets,arraysPerPass);
   
	}
	

	  
	//cout<< "**** Sorting Buckets ****" << endl;
   
	offset = 0;
	   for(int i = 0; i < numOfStreams; i++){  
	   offset = i*arraysPerStream;
       sortBuckets<<<arraysPerStream, maxBuckets, 0, stream[i]>>>(d_inputData, d_bucketSizes,d_numOfBuckets, d_prefixSum, offset,arraysPerPass);
	
	}
	
		cudaDeviceSynchronize();

	//cudaFree(d_bucketSizes);
	//cudaFree(d_splitters);
	//cudaFree(d_numOfBuckets);
	
	

    rangeKernel<<<arraysPerPass,1>>>(d_inputData, d_prefixSum, d_localRanges, d_maxAvgs,arraysPerPass);
	
	
    thrust::device_ptr<unsigned long int> dPtrRange = thrust::device_pointer_cast(d_localRanges); 
  
	temp_range[p] = thrust::reduce(dPtrRange, dPtrRange+arraysPerPass);
	cudaMemcpy (&h_localRange[p*arraysPerPass], d_localRanges, sizeof(unsigned long int)*arraysPerPass, cudaMemcpyDeviceToHost); 
	cudaMemcpy (&h_maxAvgs[p*arraysPerPass], d_maxAvgs, sizeof(int)*arraysPerPass, cudaMemcpyDeviceToHost); 
  
     }// for ends
	 
	  unsigned long int number = 0;
	 
	  for(int i = 0; i < estPasses; i++)
	  {
		  number += temp_range[i];
	  }
	   number = number/totArrays;
	  *range = number;
	  
     cudaMemcpy(d_globalRange, range, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_sampleRate, h_sampleRate, sizeof(int), cudaMemcpyHostToDevice);
	 
	 
	  //reamining part of the algorithm
	 for(int p = 0; p < estPasses; p++)
	 {
		for(int i = 0; i < numOfStreams; i++)   
        cudaMemcpyAsync(&d_inputData[newData[p].prefixArray[i*arraysPerStream]], &newData[p].dataList[newData[p].prefixArray[i*arraysPerStream]],
	    sizeof(int)*(newData[p].prefixArray[(i+1)*arraysPerStream]-newData[p].prefixArray[i*arraysPerStream]), cudaMemcpyHostToDevice, stream[i]);
	  
	  
	    for(int i = 0; i < numOfStreams -1; i++)   
        cudaMemcpyAsync(&d_prefixSum[i*arraysPerStream], &newData[p].prefixArray[i*arraysPerStream], sizeof(int)*(arraysPerStream), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_prefixSum[(numOfStreams-1)*arraysPerStream], &newData[p].prefixArray[(numOfStreams-1)*arraysPerStream],
	    sizeof(int)*(arraysPerStream+1), cudaMemcpyHostToDevice, stream[numOfStreams-1]);
    
	  //copying the ranges back
	    cudaMemcpy(d_localRanges, &h_localRange[p*arraysPerPass], sizeof(unsigned long int)*arraysPerPass, cudaMemcpyHostToDevice);
        cudaMemcpy(d_maxAvgs, &h_maxAvgs[p*arraysPerPass], sizeof(int)*arraysPerPass, cudaMemcpyHostToDevice);
	
	  
	    classKernel<<<arraysPerPass,1>>>(d_inputData, d_prefixSum, d_localRanges, d_globalRange, d_maxAvgs, d_sampledSpecSizes, d_sampleRate,arraysPerPass);

	  
	    for(int i = 0; i < numOfStreams; i++)   
        cudaMemcpyAsync(&h_result[(newData[p].prefixArray[i*arraysPerStream]) + p*arraysPerPass*maxSize], &d_inputData[newData[p].prefixArray[i*arraysPerStream]],
	    sizeof(mType)*(newData[p].prefixArray[(i+1)*arraysPerStream]-newData[p].prefixArray[i*arraysPerStream]), cudaMemcpyDeviceToHost, stream[i]);
	
		 
	 }
	 
       float milliseconds = 0;
	   cudaEventRecord(stop);
	   cudaEventSynchronize(stop);
	   cudaEventElapsedTime(&milliseconds, start, stop);
	


	cudaFreeHost(newData[0].dataList);
	cudaFreeHost(newData[1].dataList);
	cudaFreeHost(newData[2].dataList);
	cudaFreeHost(newData[3].dataList);
	cudaFreeHost(h_sortedData);
	cudaFree(d_prefixSum);
	
	cudaFree(d_inputData);
	cout<<" time taken:"<<(milliseconds-copyTime)<<" total spectra:"<<totArrays<<endl;
	return h_result;
} 


//extracting spectra from file and converting them into specFile type.

vector<specFile> retrieveSpectraFromFiles(vector<string> listOfFiles, string dir, int start, int end, int countX) {
	vector<specFile> listofSpecFiles;
	spectrum tempSpectrum;
	string filePath;
	string line;
	int p = 0;
	for(int times = 0; times < countX; times++)
	{
		for (int i = start; i < end; i++) {
			p = 0;
			specFile tempFile;
			filePath = dir + listOfFiles.at(i);
			tempFile.fileName = listOfFiles.at(i);
			ifstream myfile(filePath.c_str());
			if (myfile.is_open())
			{
				while (getline(myfile, line))
				{
					if (p == 0) {
						p++;
						tempFile.headerLine = line;
					}
					else {
						tempSpectrum = stringToSpectrum(line);
						tempFile.listVals.push_back(tempSpectrum);
					}
				}
				myfile.close();
				listofSpecFiles.push_back(tempFile);
			}
		}
	}
	return listofSpecFiles;
}

int getTotalPeaks(vector<specFile> listOfFiles){
    int totalSize = 0;
    for(int i = 0; i < listOfFiles.size(); i++){
       totalSize = totalSize + listOfFiles.at(i).listVals.size();
    }

    return totalSize;
 }


int getMaxSpecSize(vector<specFile> listofSpecFiles)
{
	int largest = 0;
	
	for (int i = 0; i < listofSpecFiles.size(); i++)
	{
		if (largest < listofSpecFiles.at(i).listVals.size())
			largest = listofSpecFiles.at(i).listVals.size();
	}
	return largest;
}

