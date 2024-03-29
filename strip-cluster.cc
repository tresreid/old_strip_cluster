#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <mm_malloc.h>
#include <omp.h>
#include <sys/time.h>

#define IDEAL_ALIGNMENT 64
using detId_t = uint32_t;
//using fedId_t = uint16_t;
//using fedCh_t = uint8_t;

int main()
{
double startx, endx;
struct timeval timecheck;

  int max_strips = 1400000;
  detId_t *detId = (detId_t *)_mm_malloc(max_strips*sizeof(detId_t), IDEAL_ALIGNMENT);
  //  fedId_t *fedId = (fedId_t *)_mm_malloc(max_strips*sizeof(fedId_t), IDEAL_ALIGNMENT);
  //fedCh_t *fedCh = (fedCh_t *)_mm_malloc(max_strips*sizeof(fedCh_t), IDEAL_ALIGNMENT);
  uint16_t *stripId = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  uint16_t *adc = (uint16_t *)_mm_malloc(max_strips*sizeof(uint16_t), IDEAL_ALIGNMENT);
  float *noise = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  float *gain = (float *)_mm_malloc(max_strips*sizeof(float), IDEAL_ALIGNMENT);
  bool *bad = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripMask = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);
  bool *seedStripNCMask = (bool *)_mm_malloc(max_strips*sizeof(bool), IDEAL_ALIGNMENT);

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
//std::cout<< "index " << i<<" noise "<< noise[i] <<std::endl; 
    i++;
  }
  int nStrips=i;
  gettimeofday(&timecheck, NULL);
  startx = (double)timecheck.tv_sec *1000 + (double)timecheck.tv_usec /1000;

  double start = omp_get_wtime();
  float ChannelThreshold = 2.0, SeedThreshold = 3.0, ClusterThresholdSquared = 25.0;
  uint8_t MaxSequentialHoles = 0, MaxSequentialBad = 1, MaxAdjacentBad = 0;
  bool RemoveApvShots = true;
  float minGoodCharge = 1620.0;

  for (int i=0; i<nStrips; i++) {
    seedStripMask[i] = false;
    seedStripNCMask[i] = false;
  }
  // find the seed strips
  int nSeedStrips=0;
#pragma omp parallel for reduction(+:nSeedStrips)
  for (int i=0; i<nStrips; i++) {
    float noise_i = noise[i];
    uint8_t adc_i = static_cast<uint8_t>(adc[i]);
    seedStripMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? true:false;
//printf("test i n a b: %d %f %d %d\n",i,noise_i,adc_i,seedStripMask[i]);
    nSeedStrips += static_cast<int>(seedStripMask[i]);
  }

  int nSeedStripsNC=0;
  seedStripNCMask[0] = seedStripMask[0];
  if (seedStripNCMask[0]) nSeedStripsNC++;
#pragma omp parallel for reduction(+:nSeedStripsNC)
  for (int i=1; i<nStrips; i++) {
	seedStripNCMask[i] == false;
    if (seedStripMask[i] == true) {
      if (stripId[i]-stripId[i-1]!=1||((stripId[i]-stripId[i-1]==1)&&!seedStripMask[i-1])) {
	seedStripNCMask[i] = true;
	nSeedStripsNC += static_cast<int>(seedStripNCMask[i]);
      }
    }
}

//printf("nSeedStripsNC = %d\n",nSeedStripsNC);
//for (int i=0; i<nStrips;i++){
//printf("NCMask i n: %d %d\n",i, seedStripNCMask[i]); 
//}

  //  std::cout<<"nStrips "<<nStrips<<"nSeedStrips "<<nSeedStrips<<"nSeedStripsNC "<<nSeedStripsNC<<std::endl;

  int *seedStripsNCIndex = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  int *clusterLastIndexLeft = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  int *clusterLastIndexRight = (int *)_mm_malloc(nSeedStripsNC*sizeof(int), IDEAL_ALIGNMENT);
  float *clusterNoiseSquared = (float *)_mm_malloc(nSeedStripsNC*sizeof(float), IDEAL_ALIGNMENT);
  uint8_t *clusterADCs = (uint8_t *)_mm_malloc(nSeedStripsNC*256*sizeof(uint8_t), IDEAL_ALIGNMENT);
  bool *trueCluster= (bool *)_mm_malloc(nSeedStripsNC*sizeof(bool), IDEAL_ALIGNMENT);

  int j=0;
  for (int i=0; i<nStrips; i++) {
    if (seedStripNCMask[i] == true) {
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
  // find the left and right bounday of the candidate cluster
  // (currently, we assume no bad strip. fix later)
#pragma omp parallel for
  for (int i=0; i<nSeedStripsNC; i++) {
    clusterNoiseSquared[i] = 0.0;
    int index=seedStripsNCIndex[i];
    clusterLastIndexLeft[i] = index;
    clusterLastIndexRight[i] = index;
    uint8_t adc_i = adc[index];
    float noise_i = noise[index];
//    printf("index: %d, noise %d\n",index, noise_i);
    clusterNoiseSquared[i] += noise_i*noise_i;
//printf("noise i n: %d %f\n",i, noise_i*noise_i);
    // find left boundary
    int testIndex=index-1;
    while(testIndex>0&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)>=0)&&((stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)<=MaxSequentialHoles)){
      float testNoise = noise[testIndex];
      uint8_t testADC = adc[testIndex];
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
	--clusterLastIndexLeft[i];
	clusterNoiseSquared[i] += testNoise*testNoise;
       //printf("pass: index: %d, testADC %d, testNoise %f, result %d\n",index,testADC, testNoise, clusterLastIndexLeft[i]);
      }
      --testIndex;
      //printf("index %d, indexL %d, testIndexL %d, testNoise: %f, testADC: %d\n",index,testIndex,clusterLastIndexLeft[i],testNoise, testADC);
    }
//printf("xxindex %d, testIndexL %d, 1: %d, 2:%d, stripID_indexL: %d, stripID_testIndex: %d\n",index, testIndex,(stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)>=0,(stripId[clusterLastIndexLeft[i]]-stripId[testIndex]-1)<=MaxSequentialHoles, stripId[clusterLastIndexLeft[i]],stripId[testIndex]);

    // find right boundary
    testIndex=index+1;
    while(testIndex<nStrips&&((stripId[testIndex]-stripId[clusterLastIndexRight[i]]-1)>=0)&&((stripId[testIndex]-stripId[clusterLastIndexRight[i]]-1)<=MaxSequentialHoles)) {
      float testNoise = noise[testIndex];
      uint8_t testADC = adc[testIndex];
      if (testADC > static_cast<uint8_t>(testNoise * ChannelThreshold)) {
        ++clusterLastIndexRight[i];
	clusterNoiseSquared[i] += testNoise*testNoise;
      }
      ++testIndex;
    }
  }
//for(int l=0; l<nSeedStripsNC;l++){
//printf("clusterLIL[%d]: %d\n",l,clusterLastIndexLeft[l]);
//printf("clusterLIR[%d]: %d\n",l,clusterLastIndexRight[l]);
//}
  // check if the candidate cluster is a true cluster
  // if so, do some adjustment for the adc values
#pragma omp parallel for
  for (int i=0; i<nSeedStripsNC; i++){
    trueCluster[i] = false;
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
  }

  double end = omp_get_wtime();
  gettimeofday(&timecheck, NULL);
  endx = (double)timecheck.tv_sec *1000 + (double)timecheck.tv_usec/1000;
  // print out the result
  for (int i=0; i<nSeedStripsNC; i++) {
    if (trueCluster[i]){
      int index = clusterLastIndexLeft[i];
      //std::cout<<"cluster "<<i<<" det Id "<<detId[index]<<" strip "<<stripId[clusterLastIndexLeft[i]]<<" seed strip "<<stripId[seedStripsNCIndex[i]]<<" ADC ";
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
  

  std::cout<<"clustering time "<<end-start<<std::endl;
printf("time: %e (ms)\n",(endx-startx));

  free(detId);
  //free(fedId);
  //free(fedCh);
  free(stripId);
  free(adc);
  free(noise);
  free(gain);
  free(bad);
  free(seedStripMask);
  free(seedStripNCMask);
  free(seedStripsNCIndex);
  free(clusterNoiseSquared);
  free(clusterLastIndexLeft);
  free(clusterLastIndexRight);
  free(clusterADCs);
  free(trueCluster);

  return 0;

}
