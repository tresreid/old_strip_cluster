//  nvcc -arch=sm_35 -rdc=true cuda-strip-cluster.cu -o cuda_strip 

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <mm_malloc.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#define IDEAL_ALIGNMENT 64
using detId_t = uint32_t;
using fedId_t = uint16_t;
using fedCh_t = uint8_t;

//#define ChannelThreshold 2.0
#define SeedThreshold 3.0
#define ClusterThresholdSquared 25.0
//#define MaxSequentialHoles 0
#define MaxSequentialBad 1
#define MaxAdjacentBad 0
#define minGoodCharge 1620.0
#define RemoveApvShots true
//float ChannelThreshold = 2.0, SeedThreshold = 3.0, ClusterThresholdSquared = 25.0;
//uint8_t MaxSequentialHoles = 0, MaxSequentialBad = 1, MaxAdjacentBad = 0;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void clusterChecker(int nSeedStripsNC,int* clusterLastIndexLeft, int* clusterLastIndexRight, uint16_t* adc, float* clusterNoiseSquared,float* gain, uint8_t* clusterADCs, bool* trueCluster)
{
  unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
  while(i<nSeedStripsNC){
    int left=clusterLastIndexLeft[i];
    int right=clusterLastIndexRight[i];
    int size=right-left+1;
    int adcsum = 0;
    for (int j=0; j<size; j++) {
      adcsum += (int)adc[left+j];
    }
    bool noiseSquaredPass = clusterNoiseSquared[i]*ClusterThresholdSquared <= ((float)(adcsum)*float(adcsum));
    bool chargePerCMPass = (float)(adcsum)/0.047f > minGoodCharge;
    if (noiseSquaredPass&&chargePerCMPass) {
      for (int j=0; j<size; j++){
      uint8_t adc_j = adc[left+j];
      float gain_j = gain[left+j];
      auto charge = int( float(adc_j)/gain_j + 0.5f );
      if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
      clusterADCs[j*nSeedStripsNC+i] = adc_j;
      }
      trueCluster[i] = true;
    }
  i += blockDim.x*gridDim.x;
  }
}


__global__ void findBoundries(const int nStrips,const int nSeedStripsNC,const int* seedStripsNCIndex, float* clusterNoiseSquared,int* clusterLastIndexLeft,int* clusterLastIndexRight, const uint16_t* stripId, const float* noise,const uint16_t* adc)
{
  const float ChannelThreshold = 2.0;
  const uint8_t MaxSequentialHoles = 0;
  int i = threadIdx.x + blockIdx.x*blockDim.x;
 // printf("0i: %d\n", i);
  int index, IndexLeft, IndexRight, testIndexL, testIndexR;
  //float noise_i;
  while(i<nSeedStripsNC){
    clusterNoiseSquared[i] = 0.0;
    //clusterLastIndexLeft[i] = 0;
    //clusterLastIndexRight[i] = 0;
    index = seedStripsNCIndex[i];
    IndexLeft = index;
    IndexRight = index;
    //clusterLastIndexLeft[i] = index;
    //clusterLastIndexRight[i] = index;
    //float noise_i = noise[index];
    float noise_i = noise[index];
    //printf("index: %d, noise %d\n",index, noise_i);
    //printf("index %d noise %f\n",i, noise[i]);
    clusterNoiseSquared[i] += noise_i*noise_i;
    // find left boundary
    testIndexL=index-1;
    //while(testIndexL>0&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndexL]-1)>=0)&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndexL]-1)<=MaxSequentialHoles)){
    while(testIndexL>0&&((stripId[IndexLeft]-stripId[testIndexL]-1)>=0)&&((stripId[IndexLeft]-stripId[testIndexL]-1)<=MaxSequentialHoles)){

      float testNoise = noise[testIndexL];
      uint8_t testADC = adc[testIndexL];
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
        //--clusterLastIndexLeft[i];
        --IndexLeft;
        clusterNoiseSquared[i] += testNoise*testNoise;
//	printf("pass: index: %d, testADC %u, testNoise %f, result %d\n",index,testADC, testNoise, IndexLeft);
      }
      --testIndexL;
  //      printf("index %d, indexL %d, testIndexL %d, testNoise: %f, testADC: %u\n",index, testIndexL,IndexLeft,testNoise,testADC); 
    }
   // printf("xxindex %d, testIndexL %d, 1: %d, 2:%d, stripID_indexL: %d, stripID_testIndex: %d\n",index, testIndexL,(stripId[IndexLeft]-stripId[testIndexL]-1)>=0,(stripId[IndexLeft]-stripId[testIndexL]-1)<=MaxSequentialHoles, stripId[IndexLeft], stripId[testIndexL]);

    // find right boundary
    testIndexR=index+1;
    //while(testIndexR<nStrips&&((stripId[testIndexR]-stripId[clusterLastIndexRight[i]]-1)>=0)&&((stripId[testIndexR]-stripId[clusterLastIndexRight[i]]-1)<=MaxSequentialHoles)){
    while(testIndexR<nStrips&&((stripId[testIndexR]-stripId[IndexRight]-1)>=0)&&((stripId[testIndexR]-stripId[IndexRight]-1)<=MaxSequentialHoles)){

      float testNoise = noise[testIndexR];
      uint8_t testADC = static_cast<uint8_t>(adc[testIndexR]);
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
        ++IndexRight;
        //++clusterLastIndexRight[i];
        clusterNoiseSquared[i] += testNoise*testNoise;
      }
      ++testIndexR;
    }
    clusterLastIndexLeft[i] = IndexLeft;
    clusterLastIndexRight[i] = IndexRight;
  i += blockDim.x*gridDim.x;
  }
}

/*
__global__ void getNCSeedStrips(const int nStrips,const float* noise_d,const uint16_t* adc,int* nSeedStripsNC_old)
{

  // find the seed strips
   unsigned int in = threadIdx.x + blockIdx.x*blockDim.x+1;
while(in<nStrips){
    float noise_i = noise_d[in];
    uint8_t adc_i = static_cast<uint8_t>(adc[in]);
    float noise_iR = noise_d[in-1];
    uint8_t adc_iR = static_cast<uint8_t>(adc[in-1]);
    nSeedStripsNC_old[in] = ((adc_iR >= static_cast<uint8_t>( noise_iR * SeedThreshold)) && (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)));
   // printf("xxx: %d %d\n",in, ((adc_iR >= static_cast<uint8_t>( noise_iR * SeedThreshold)) && (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold))));

    in = in + blockDim.x*gridDim.x;
  }
}

*/
__global__ void getNCSeedStrips(const int nStrips,const uint16_t* adc,const float* noise,const uint16_t* stripId, int* seedStripMask, int* seedStripNCMask)
{
  unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
  
  while(i<nStrips){
    float noise_i = noise[i];
	//printf("strips noise %d %f\n",i,noise[i]);
    uint8_t adc_i = static_cast<uint8_t>(adc[i]);
    seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? true:false;
  i += blockDim.x*gridDim.x;
  }
}
__global__ void getNCSeedStrips1(const int nStrips,const uint16_t* stripId, int* seedStripMask, int* seedStripNCMask)
{
  unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i== 0){
    seedStripNCMask[0] = seedStripMask[0];
  }
  else{ 
    while(i<nStrips){
    //seedStripNCMask[i] == false;
      if (seedStripMask[i] == true) {
        if (stripId[i]-stripId[i-1]!=1||((stripId[i]-stripId[i-1]==1)&&!seedStripMask[i-1])) {
          seedStripNCMask[i] = true;
        }
      }
    i += blockDim.x*gridDim.x;
    }
  }
}



int main()
{

  double start, end;
  struct timeval timecheck;

  int max_strips = 1400000;
  detId_t *detId = (detId_t *)_mm_malloc(max_strips*sizeof(detId_t), IDEAL_ALIGNMENT);
  fedId_t *fedId = (fedId_t *)_mm_malloc(max_strips*sizeof(fedId_t), IDEAL_ALIGNMENT);
  fedCh_t *fedCh = (fedCh_t *)_mm_malloc(max_strips*sizeof(fedCh_t), IDEAL_ALIGNMENT);
  uint16_t *stripId = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  uint16_t *adc = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  float *noise = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  float *gain = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  bool *bad = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);

  // read in the data
  std::ifstream digidata_in("digidata.bin", std::ofstream::in | std::ios::binary);
  int i=0;
  while (digidata_in.read((char*)&detId[i], sizeof(detId_t)).gcount() == sizeof(detId_t)) {
    //digidata_in.read((char*)&fedId[i], sizeof(fedId_t));
    //digidata_in.read((char*)&fedCh[i], sizeof(fedCh_t));
    digidata_in.read((char*)&stripId[i], sizeof(uint16_t));
    digidata_in.read((char*)&adc[i], sizeof(uint16_t));
    digidata_in.read((char*)&noise[i], sizeof(float));
    digidata_in.read((char*)&gain[i], sizeof(float));
    digidata_in.read((char*)&bad[i], sizeof(bool));
    if (bad[i])
      std::cout<<"index "<<i<<" detid "<<detId[i]<<" stripId "<<stripId[i]<<
        " adc "<<adc[i]<<" noise "<<noise[i]<<" gain "<<gain[i]<<" bad "<<bad[i]<<std::endl;
   // std::cout<< "index " << i<<" noise "<< noise[i] <<std::endl;
    i++;
  }
  int nStrips=i;
  
  gettimeofday(&timecheck, NULL);
  start = (double)timecheck.tv_sec *1000 + (double)timecheck.tv_usec /1000;

  int nSeedStripsNC=0;
  int* seedStripMask;
  int* seedStripsNCMask;
  float* gain_d;
  uint16_t* adc_d;
  uint16_t* stripId_d;
  float* noise_d;
  cudaMallocManaged((void**)&seedStripMask, nStrips*sizeof(int));
  cudaMallocManaged((void**)&seedStripsNCMask, nStrips*sizeof(int));
  cudaMalloc((void**)&gain_d, max_strips*sizeof(float));
  cudaMalloc((void**)&adc_d, max_strips*sizeof(uint16_t));
  cudaMalloc((void**)&stripId_d, max_strips*sizeof(uint16_t));
  cudaMalloc((void**)&noise_d, max_strips*sizeof(float));
  cudaMemcpy(gain_d, gain,max_strips*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(adc_d, adc,max_strips*sizeof(uint16_t),cudaMemcpyHostToDevice);
  cudaMemcpy(stripId_d, stripId,max_strips*sizeof(uint16_t),cudaMemcpyHostToDevice);
  cudaMemcpy(noise_d, noise,max_strips*sizeof(float),cudaMemcpyHostToDevice);


  getNCSeedStrips<<<128,256>>>(nStrips,adc_d,noise_d,stripId_d,seedStripMask,seedStripsNCMask);
  cudaDeviceSynchronize();
  getNCSeedStrips1<<<128,256>>>(nStrips,stripId_d,seedStripMask,seedStripsNCMask);
  cudaDeviceSynchronize();
  nSeedStripsNC=0;
  for(int l=0; l<nStrips;l++){
      if ( l == 32768 || l == 65536 || l==458752){ seedStripsNCMask[l] = true;}
    nSeedStripsNC += seedStripsNCMask[l];
  }
//printf("nSeedStripsNC = %d\n",nSeedStripsNC);
//for (int i=0; i<nStrips;i++){
//printf("NCMask i n: %d %d\n",i, seedStripsNCMask[i]);
//}

  int *seedStripsNCIndex ;//= (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  int *clusterLastIndexLeft ;//= (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  int *clusterLastIndexRight ;//= (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  float *clusterNoiseSquared;// = (float *)_mm_malloc(nSeedStripsNC*sizeof(float), IDEAL_ALIGNMENT);
  uint8_t *clusterADCs ;//= (uint8_t *)_mm_malloc(nSeedStripsNC*256*sizeof(uint8_t), IDEAL_ALIGNMENT);
  bool *trueCluster;//= (bool *)_mm_malloc(nSeedStripsNC*sizeof(bool), IDEAL_ALIGNMENT);

  cudaMallocManaged((void**)&seedStripsNCIndex,nSeedStripsNC*sizeof(int));
  cudaMallocManaged((void**)&clusterLastIndexLeft,nSeedStripsNC*sizeof(int));
  cudaMallocManaged((void**)&clusterLastIndexRight,nSeedStripsNC*sizeof(int));
  cudaMallocManaged((void**)&clusterNoiseSquared,nSeedStripsNC*sizeof(float));
  cudaMallocManaged((void**)&clusterADCs,nSeedStripsNC*256*sizeof(uint8_t));
  cudaMallocManaged((void**)&trueCluster,nSeedStripsNC*sizeof(bool));
  
  int j=0;
  for (int i=0; i<nStrips; i++) {
    if (seedStripsNCMask[i] == true) {
      seedStripsNCIndex[j] = i;
      j++;
    }
  }

  if (j!=nSeedStripsNC) {
    std::cout<<"j "<<j<<"nSeedStripsNC "<<nSeedStripsNC<<std::endl;
    exit (1);
  }
//printf("SeedStripsNC: %d\n",nSeedStripsNC);
//for(int l=0; l<nSeedStripsNC;l++){
//printf("stripsNCMask[%d]: %d\n",l,seedStripsNCIndex[l]);
//} 

findBoundries<<<1,1>>>(nStrips, nSeedStripsNC,seedStripsNCIndex,clusterNoiseSquared,clusterLastIndexLeft,clusterLastIndexRight,stripId_d,noise_d,adc_d);
cudaDeviceSynchronize();

//for(int l=0; l<nSeedStripsNC;l++){
//printf("clusterLIL[%d]: %d\n",l,clusterLastIndexLeft[l]);
//printf("clusterLIR[%d]: %d\n",l,clusterLastIndexRight[l]);
//}  

clusterChecker<<<128,128>>>(nSeedStripsNC,clusterLastIndexLeft,clusterLastIndexRight,adc_d, clusterNoiseSquared,gain_d, clusterADCs, trueCluster);
cudaDeviceSynchronize();
  




  gettimeofday(&timecheck, NULL);
  end = (double)timecheck.tv_sec *1000 + (double)timecheck.tv_usec/1000;
 //print out the result
  for (int i=0; i<nSeedStripsNC; i++) {    
    if (trueCluster[i]){
      int index = clusterLastIndexLeft[i];
      std::cout<<" det id "<<detId[index]<<" strip "<<stripId[clusterLastIndexLeft[i]]<< ": ";
      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      int size=right-left+1;
      for (int j=0; j<size; j++){
	std::cout<<(int)clusterADCs[j*nSeedStripsNC+i]<<" ";
      }
      std::cout<<std::endl;
    }
  }


  printf("time: %e (ms)\n",(end-start));
	
  free(detId);
  free(fedId);
  free(stripId);
  free(adc);
  free(noise);
  cudaFree(adc_d);
  cudaFree(noise_d);
  cudaFree(gain_d);
  cudaFree(stripId_d);
  free(gain);
  free(bad);
  cudaFree(seedStripMask);
  cudaFree(seedStripsNCMask);
  cudaFree(seedStripsNCIndex);
  cudaFree(clusterNoiseSquared);
  cudaFree(clusterLastIndexLeft);
  cudaFree(clusterLastIndexRight);
  cudaFree(clusterADCs);
  cudaFree(trueCluster);

  return 0;

}
