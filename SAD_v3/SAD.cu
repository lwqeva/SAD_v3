#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <memory>
#include <cassert>

#include "SAD.h"

#define CUDA_CHECK(x,y)  if((x) != cudaSuccess){ puts(y); assert(0); }

__global__ void propagateKernel(float *p1, float *p2, float *dK, float *pd, int N, int b)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int m = 3*j;
	int k = 6*j;

	float *d1 = p1, *d2 = p2, *t;

	float dK_row[3];
	dK_row[0] = dK[m];
	dK_row[1] = dK[m+1];
	dK_row[2] = dK[m+2];

	for(int n = 0; n < N; ++n)
	{
		d2[j] = pd[k] * dK_row[0] + pd[k+1] * dK_row[1] + pd[k+2] * dK_row[2]
			+ pd[k+3] * d1[j-1] + pd[k+4] * d1[j] + pd[k+5] * d1[j+1];
		k += b;

		t = d2, d2 = d1, d1 = t;
		__syncthreads();
	}
	p1[j] = d1[j] * pd[b*N + j];
}

// MemoryManager allocates memory for Jacobian computation
void MemoryManagerForward::Allocate(int M, int N)
	{
		this->M = M;
		this->N = N;

		nvar_S = M-1;
		nvar_K = nvar_S + 3*(M-1);
		nvar_T0  = nvar_K + (M-1) + 4;
		nvar_T = nvar_T0 + (M-1)*M*M;
		nvar_SqE = nvar_T + 1;

		nnzpd_S = 0;
		nnzpd_K = 5 + (M-3)*7 + 5;
		nnzpd_T0 = nnzpd_K;
		nnzpd_T = nnzpd_T0 + (M-1)*M*M*6;
		nnzpd_SqE = nnzpd_T + (M-1);

		assert(ADS::nvar == nvar_SqE);
		assert(ADS::nnz_pd == nnzpd_SqE);

		cudaError_t err = cudaSuccess;

		err = cudaMalloc(&D1, sizeof(*D1)*(M+1));
		CUDA_CHECK(err, "D1 allocation failed.");
		cudaMemset(D1, 0, sizeof(*D1)*(M+1));
		d1 = D1 + 1;
		h_d1 = new float[M-1];

		err = cudaMalloc(&D2, sizeof(*D2)*(M+1));
		CUDA_CHECK(err, "D2 allocation failed.");
		cudaMemset(D2, 0, sizeof(*D2)*(M+1));
		d2 = D2 + 1;

		err = cudaMalloc(&dK, sizeof(*dK)*(nvar_K-nvar_S));
		CUDA_CHECK(err, "dK allocation failed.");
		h_dK = new float[(nvar_K-nvar_S)];
		memset(h_dK, 0, sizeof(*h_dK)*(nvar_K-nvar_S));

		err = cudaMalloc(&pd, sizeof(*pd)*ADS::nnz_pd);
		CUDA_CHECK(err,"pd allocation failed.");
		err = cudaMemcpy(pd, ADS::pd, sizeof(*pd)*ADS::nnz_pd, cudaMemcpyHostToDevice);
		CUDA_CHECK(err,"pd memcpy failed.");
		
	}

void MemoryManagerForward::Clear()
	{
		cudaFree(D1);
		cudaFree(D2);
		cudaFree(dK);
		cudaFree(pd);

		delete [] h_d1;
		delete [] h_dK;	
	}


void ADS::cudaGetJacobianForward(float *J, int m, MemoryManagerForward &mmf)
{
	int M = mmf.M;
	int N = mmf.N;
	int NOI = M - 1;

	cudaError_t err = cudaSuccess;
	float *dK = mmf.dK,
		*h_dK = mmf.h_dK;

	float *d1 = mmf.d1;
	float *d2 = mmf.d2;
	float *h_d1 = mmf.h_d1;

	int *rid = ADS::cooRow;
	int *cid = ADS::cooCol;
	float *pd = mmf.pd, *h_pd = ADS::pd;

	int n = M - 1;
	for(int xid = 0; xid < n; ++xid)
	{
		for(int ipd = 0; ipd < mmf.nnzpd_K; ++ipd)
		{
			if( cid[ipd] == xid )
				h_dK[ rid[ipd] - mmf.nvar_S ] += h_pd[ipd];
		}
		err = cudaMemcpy(dK, h_dK, sizeof(*h_dK)*(mmf.nvar_K-mmf.nvar_S), cudaMemcpyHostToDevice);
		CUDA_CHECK(err,"dK memcpy failed.");

		propagateKernel<<<1,mmf.nvar_S>>>(d1, d2, dK, pd+mmf.nnzpd_K, N, (M-1)*6);
		err = cudaGetLastError();;
		CUDA_CHECK(err,"Kernel error");

		err = cudaMemcpy(h_d1, d1, sizeof(*h_d1)*NOI, cudaMemcpyDeviceToHost);
		CUDA_CHECK(err,"h_d1 memcpy failed.");

		float adj = 0;
		for(int j = 0; j < NOI; ++j)
		{
			adj += h_d1[j];
		}

		memcpy(J+m*xid, &adj, sizeof(*J)*m);

		cudaMemset(d1, 0, sizeof(*d1)*NOI);
		memset(h_dK, 0, sizeof(*dK)*(mmf.nvar_K-mmf.nvar_S));
	}
}