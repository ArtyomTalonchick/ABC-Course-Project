#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>
#define Thread 1024
__global__ void KernelGaussSeidel(float* deviceA, float* deviceF, float* deviceX0, float*
	deviceX1, int N) {
	float sum = 0.0f;
	for (int i = 0; i < N; i++) deviceX0[i] = deviceX1[i];
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = 0; j < t; j++) sum += deviceA[j + t * N] * deviceX1[j];
	for (int j = t + 1; j < N; j++) sum += deviceA[j + t * N] * deviceX0[j];
	deviceX1[t] = (deviceF[t] - sum) / deviceA[t + t * N];
}
__global__ void EpsGaussSeidel(float *deviceX0, float *deviceX1, float *delta, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	delta[i] += fabs(deviceX0[i] - deviceX1[i]);
	deviceX0[i] = deviceX1[i];
}
int main() {
	srand(time(NULL));
	float *hostA, *hostX, *hostX0, *hostX1, *hostF, *hostDelta;
	float sum, eps;
	float EPS = 1.e-5;
	int N = 10000;
	float size = N * N;
	int count;
	int Block = (int)ceil((float)N / Thread);
	dim3 Blocks(Block);
	dim3 Threads(Thread);
	int Num_diag = 0.5f*(int)N*0.3f;
	float mem_sizeA = sizeof(float)*size;
	unsigned int mem_sizeX = sizeof(float)*(N);
	hostA = (float*)malloc(mem_sizeA);
	hostF = (float*)malloc(mem_sizeX);
	hostX = (float*)malloc(mem_sizeX);
	hostX0 = (float*)malloc(mem_sizeX);
	hostX1 = (float*)malloc(mem_sizeX);
	hostDelta = (float*)malloc(mem_sizeX);
	for (int i = 0; i < size; i++) {
		hostA[i] = 0.0f;
	}
	for (int i = 0; i < N; i++) {
		hostA[i + i * N] = rand() % 50 + 1.0f*N;
	}
	for (int k = 1; k < Num_diag + 1; k++) {
			for (int i = 0; i < N - k; i++) {
				hostA[i + k + i * N] = rand() % 5;
				hostA[i + (i + k)*N] = rand() % 5;
			}
	}
	for (int i = 0; i < N; i++) {
		hostX[i] = rand() % 50;
		hostX0[i] = 1.0f;
		hostDelta[i] = 0.0f;
	}
	for (int i = 0; i < N; i++) {
		sum = 0.0f;
		for (int j = 0; j < N; j++) sum += hostA[j + i * N] * hostX[j];
		hostF[i] = sum;
	}
	float *deviceA, *deviceX0, *deviceX1, *deviceF, *delta;
	for (int i = 0; i < N; i++) hostX1[i] = 1.0f;
	cudaMalloc((void**)&deviceA, mem_sizeA);
	cudaMalloc((void**)&deviceF, mem_sizeX);
	cudaMalloc((void**)&deviceX0, mem_sizeX);
	cudaMalloc((void**)&deviceX1, mem_sizeX);
	cudaMalloc((void**)&delta, mem_sizeX);
	cudaMemcpy(deviceA, hostA, mem_sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceF, hostF, mem_sizeX, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceX0, hostX0, mem_sizeX, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceX1, hostX1, mem_sizeX, cudaMemcpyHostToDevice);
	count = 0; eps = 1.0f;
	while (eps > EPS)
	{
		count++;
		cudaMemcpy(delta, hostDelta, mem_sizeX, cudaMemcpyHostToDevice);
		KernelGaussSeidel << < Blocks, Threads >> > (deviceA, deviceF, deviceX0,
			deviceX1, N);
		EpsGaussSeidel << < Blocks, Threads >> > (deviceX0, deviceX1, delta, N);
		cudaMemcpy(hostDelta, delta, mem_sizeX, cudaMemcpyDeviceToHost);
		eps = 0.0f;
		for (int j = 0; j < N; j++) {
			eps += hostDelta[j]; hostDelta[j] = 0;
		}
		eps = eps / N;
	}
	cudaMemcpy(hostX1, deviceX1, mem_sizeX, cudaMemcpyDeviceToHost);
	cudaFree(deviceA);
	cudaFree(deviceF);
	cudaFree(deviceX0);
	cudaFree(deviceX1);
	free(hostA);
	free(hostF);
	free(hostX0);
	free(hostX1);
	free(hostX);
	free(hostDelta);
	return 0;
}
