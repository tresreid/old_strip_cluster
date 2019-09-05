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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void getNCSeedStrips(const int nStrips,const float* noise_d,const uint16_t* adc,const int SeedThreshold,int* nSeedStripsNC_old)
{

  // find the seed strips
   unsigned int in = threadIdx.x + blockIdx.x*blockDim.x+1;
while(in<nStrips){
    float noise_i = noise_d[in];
    uint8_t adc_i = static_cast<uint8_t>(adc[in]);
    float noise_iR = noise_d[in-1];
    uint8_t adc_iR = static_cast<uint8_t>(adc[in-1]);
     nSeedStripsNC_old[in] = ((adc_iR >= static_cast<uint8_t>( noise_iR * SeedThreshold)) && (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)));
     printf("xxx: %d %d\n",in, ((adc_iR >= static_cast<uint8_t>( noise_iR * SeedThreshold)) && (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold))));
    //seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? true:false;
 // printf("test 5: %d\n",in);
    //if (seedStripMask[i] == true) nSeedStrips++;

in = in + blockDim.x*gridDim.x;
  }
//__syncthreads();
//if(in==0){// nSeedStripsNC_old = nSeedStripsNCx;
//printf("done");
//}
//  int nSeedStripsNC=0;
//  for (int i=0; i<nStrips; i++) {
//    if (seedStripMask[i] == true) {
//      if (stripId[i]-stripId[i-1]!=1) {
//	seedStripNCMask[i] = true;
//	nSeedStripsNC++;
  //    }
  //  }
 // }
//in += blockDim.x*gridDim.x;
}



int main()
{

  double start, end;
  struct timeval timecheck;
  gettimeofday(&timecheck, NULL);
  start = (double)timecheck.tv_sec *1000 + (double)timecheck.tv_usec /1000;

  int max_strips = 1400000;
  detId_t *detId = (detId_t *)_mm_malloc(max_strips*sizeof(detId_t), IDEAL_ALIGNMENT);
  fedId_t *fedId = (fedId_t *)_mm_malloc(max_strips*sizeof(fedId_t), IDEAL_ALIGNMENT);
  fedCh_t *fedCh = (fedCh_t *)_mm_malloc(max_strips*sizeof(fedCh_t), IDEAL_ALIGNMENT);
  uint16_t *stripId = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  uint16_t *adc = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  float *noise = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  float *gain = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  bool *bad = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripMask = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripNCMask = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);

  //gpuErrchk((cudaMallocManaged((void**)adc,max_strips*sizeof(uint16_t))));
  //gpuErrchk((cudaMallocManaged((void**)noise,max_strips*sizeof(float))));

  // read in the data
  std::ifstream digidata_in("digidata.bin", std::ofstream::in | std::ios::binary);
  int i=0;
  while (digidata_in.read((char*)&detId[i], sizeof(detId_t)).gcount() == sizeof(detId_t)) {
    digidata_in.read((char*)&fedId[i], sizeof(fedId_t));
    digidata_in.read((char*)&fedCh[i], sizeof(fedCh_t));
    digidata_in.read((char*)&stripId[i], sizeof(uint16_t));
    digidata_in.read((char*)&adc[i], sizeof(uint16_t));
    digidata_in.read((char*)&noise[i], sizeof(float));
    digidata_in.read((char*)&gain[i], sizeof(float));
    digidata_in.read((char*)&bad[i], sizeof(bool));
    if (bad[i])
      std::cout<<"detid "<<detId[i]<<" fedId "<<fedId[i]<<" fedCh "<<(int)fedCh[i]<<" stripId "<<stripId[i]<<
      " adc "<<adc[i]<<" noise "<<noise[i]<<" gain "<<gain[i]<<" bad "<<bad[i]<<std::endl;

    i++;
  }
  int nStrips=i;
  

  float ChannelThreshold = 2.0, SeedThreshold = 3.0, ClusterThresholdSquared = 25.0;
  uint8_t MaxSequentialHoles = 0, MaxSequentialBad = 1, MaxAdjacentBad = 0;
  bool RemoveApvShots = true;
  float minGoodCharge = 1620.0;
  int nSeedStripsNC=0;
  int* nSeedStripsNC_d;
  int* nSeedStripsNC_out;
  //cudaMallocManaged((void*)nSeedStripsNC,sizeof(int));

//
//  // find the seed strips
//  int nSeedStrips=0;
//  for (int i=0; i<nStrips; i++) {
//    float noise_i = noise[i];
//    uint8_t adc_i = static_cast<uint8_t>(adc[i]);
//    seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? true:false;
//    if (seedStripMask[i] == true) nSeedStrips++;
//  }
//
//  int nSeedStripsNC=0;
//  for (int i=0; i<nStrips; i++) {
//    if (seedStripMask[i] == true) {
//      if (stripId[i]-stripId[i-1]!=1) {
//	seedStripNCMask[i] = true;
//	nSeedStripsNC++;
//      }
//    }
//  }
  float* noise_d;
  uint16_t* adc_d;
  cudaMalloc((void**)&noise_d, sizeof(noise));
  cudaMalloc((void**)&adc_d, sizeof(adc));
  //cudaMalloc((void**)&nSeedStripsNC_d, max_strips*sizeof(int));
  cudaMallocManaged((void**)&nSeedStripsNC_d, nStrips*sizeof(int));
  //cudaalloc((void**)&nSeedStripsNC_out, max_strips*sizeof(int));
  //cudaMemcpy(&nSeedStripsNC_d, &nSeedStripsNC,sizeof(nSeedStripsNC),cudaMemcpyHostToDevice);
  cudaMemcpy(noise_d, noise,sizeof(noise),cudaMemcpyHostToDevice);
  cudaMemcpy(adc_d, adc,sizeof(adc),cudaMemcpyHostToDevice);
printf("test 0: %d\n", nStrips);
//for( int j =0; j<nStrips; j++){
//printf("noise[%d]: %f\n",j,noise[j]);} 
getNCSeedStrips<<<32,128>>>(nStrips,noise_d,adc_d,SeedThreshold,nSeedStripsNC_d);
cudaDeviceSynchronize();
//std::cout<<"nStrips "<<nStrips<<"nSeedStrips "<<nSeedStrips<<"nSeedStripsNC "<<nSeedStripsNC<<std::endl;

printf("test 1:%d\n",nSeedStripsNC_d[439133]);
//cudaMemcpy(nSeedStripsNC_out, nSeedStripsNC_d,nStrips*sizeof(int),cudaMemcpyDeviceToHost);

printf("test 2");
for (int j=0; j< nStrips; j++){ nSeedStripsNC += nSeedStripsNC_d[j];}
//printf("test 3");
printf("test x: %d\n",nSeedStripsNC);
//  int *seedStripsNCIndex = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
//  int *clusterLastIndexLeft = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
//  int *clusterLastIndexRight = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
//  float *clusterNoiseSquared = (float *)_mm_malloc(nSeedStripsNC*sizeof(float), IDEAL_ALIGNMENT);
//  uint8_t *clusterADCs = (uint8_t *)_mm_malloc(nSeedStripsNC*256*sizeof(uint8_t), IDEAL_ALIGNMENT);
//  bool *trueCluster= (bool *)_mm_malloc(nSeedStripsNC*sizeof(bool), IDEAL_ALIGNMENT);
//
//  int j=0;
//  for (int i=0; i<nStrips; i++) {
//    if (seedStripNCMask[i] == true) {
//      seedStripsNCIndex[j] = i;
//      j++;
//    }
//  }
//
//  if (j!=nSeedStripsNC) {
//    std::cout<<"j "<<j<<"nSeedStripsNC "<<nSeedStripsNC<<std::endl;
//    exit (1);
//  }
//
//  for (int i=0; i<nSeedStripsNC; i++) {
//    trueCluster[i] = false;
//    clusterNoiseSquared[i] = 0;
//  }
//
//  // find the left and right bounday of the candidate cluster
//  // (currently, we assume no bad strip. fix later)
//  for (int i=0; i<nSeedStripsNC; i++) {
//    int index=seedStripsNCIndex[i];
//    clusterLastIndexLeft[i] = index;
//    clusterLastIndexRight[i] = index;
//    uint8_t adc_i = adc[index];
//    float noise_i = noise[index];
//    clusterNoiseSquared[i] += noise_i*noise_i;
//    // find left boundary
//    int testIndex=index-1;
//    while(index>0&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)<=MaxSequentialHoles)){
//      float testNoise = noise[testIndex];
//      uint8_t testADC = adc[testIndex];
//      if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
//	--clusterLastIndexLeft[i];
//	clusterNoiseSquared[i] += testNoise*testNoise;
//      }
//      --testIndex;
//    }
//
//    // find right boundary
//    testIndex=index+1;
//    while(testIndex<nStrips&&((stripId[testIndex]-stripId[clusterLastIndexRight[i]]-1)<=MaxSequentialHoles)) {
//      float testNoise = noise[testIndex];
//      uint8_t testADC = adc[testIndex];
//      if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
//        ++clusterLastIndexRight[i];
//	clusterNoiseSquared[i] += testNoise*testNoise;
//      }
//      ++testIndex;
//    }
//  }
//
//  // check if the candidate cluster is a true cluster
//  // if so, do some adjustment for the adc values
//  for (int i=0; i<nSeedStripsNC; i++){
//    int left=clusterLastIndexLeft[i];
//    int right=clusterLastIndexRight[i];
//    int size=right-left+1;
//    int adcsum = 0;
//    for (int j=0; j<size; j++) {
//      adcsum += (int)adc[left+j];
//    }
//    bool noiseSquaredPass = clusterNoiseSquared[i]*ClusterThresholdSquared <= ((float)(adcsum)*float(adcsum));
//    bool chargePerCMPass = (float)(adcsum)/0.047f > minGoodCharge;
//    if (noiseSquaredPass&&chargePerCMPass) {
//      for (int j=0; j<size; j++){
//	uint8_t adc_j = adc[left+j];
//	float gain_j = gain[left+j];
//	auto charge = int( float(adc_j)/gain_j + 0.5f );
//	if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
//	clusterADCs[j*nSeedStripsNC+i] = adc_j;
//      }
//      trueCluster[i] = true;
//    }
//  }
//
//  // print out the result
//  for (int i=0; i<nSeedStripsNC; i++) {
//    if (trueCluster[i]){
//      int index = clusterLastIndexLeft[i];
//      std::cout<<"cluster "<<i<<" det Id "<<detId[index]<<" fed Id "<<fedId[index]<<" strip "<<stripId[clusterLastIndexLeft[i]]<<" ADC ";
//      int left=clusterLastIndexLeft[i];
//      int right=clusterLastIndexRight[i];
//      int size=right-left+1;
//      for (int j=0; j<size; j++){
//	std::cout<<(int)clusterADCs[j*nSeedStripsNC+i]<<" ";
//      }
//      std::cout<<std::endl;
//    }
//  }
//

  gettimeofday(&timecheck, NULL);
  end = (double)timecheck.tv_sec *1000 + (double)timecheck.tv_usec/1000;
  printf("time: %e (ms)\n",(end-start));
	
  free(detId);
  free(fedId);
  free(stripId);
  free(adc);
  free(noise);
  cudaFree(adc_d);
  cudaFree(noise_d);
  free(gain);
  free(bad);
//  free(seedStripMask);
//  free(seedStripNCMask);
//  free(seedStripsNCIndex);
//  free(clusterNoiseSquared);
//  free(clusterLastIndexLeft);
//  free(clusterLastIndexRight);
//  free(clusterADCs);
// free(trueCluster);

  return 0;

}
