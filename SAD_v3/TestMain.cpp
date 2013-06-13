#include "HeatConduction.h"
#include "SAD.h"

#include <cmath>
#include <cstdio>
#include <ctime>

#define M_PI       3.14159265358979323846

void TestCase4(int size)
{
	clock_t tic = clock();
	Rod::Initialize(size);	// 430 is maximum
	int NOI = Rod::M-1;

	float *s = new float[NOI];
	for(int i = 1; i < Rod::M; ++i)
	{
		s[i-1] = (float)i/(float)Rod::M;//(float)( (1 + tanh( (2*i/Rod::M - 1)*M_PI ))/2 );
	}

	Rod targetRod = Rod::CreateRod(s);
	Solve_HeatEquation(targetRod.s, targetRod.tN);
	clock_t toc_fd = clock() - tic;

	ADS::Initialize();
	ADV *S = new ADV[Rod::M-1];
	for(int i = 0; i < NOI; ++i)
	{
		S[i] = (float)0.5*s[i];
	}
	
	tic = clock();
	ADV *tN = new ADV[Rod::M+1];
	printf("nvar = %d\tnnz_pd = %d\n", ADS::nvar, ADS::nnz_pd);
	Solve_HeatEquation(S, tN);
	printf("nvar = %d\tnnz_pd = %d\n", ADS::nvar, ADS::nnz_pd);
	ADV sqe = SquaredError(NOI, targetRod.tN+1, tN+1);
	printf("nvar = %d\tnnz_pd = %d\n", ADS::nvar, ADS::nnz_pd);
	printf("SqE = %6.4f\n",sqe.v);
	clock_t toc_sqe = clock() - tic;
	
	float *J = new float[NOI];

	tic = clock();
	ADS::GetJacobianForward(J, 1, NOI);
	clock_t toc_grad = clock() - tic;

	MemoryManagerForward mmf;
	mmf.Allocate(Rod::M,Rod::N);
	float *J2 = new float[NOI];
	tic = clock();
	ADS::cudaGetJacobianForward(J2, 1, mmf);
	clock_t toc_gpu_grad = clock() - tic;
	mmf.Clear();
	
	ShowJacobian(J, 1, NOI, true);
	puts("\n\n");
	ShowJacobian(J2, 1, NOI, true);

	float elap = float(toc_fd) / CLOCKS_PER_SEC;
	printf("\n%8.6f sec\t Forward Solve\n", elap);
	elap = float(toc_sqe) / CLOCKS_PER_SEC; 
	printf("\n%8.6f sec\t Squared Error\n", elap);
	elap = float(toc_grad) / CLOCKS_PER_SEC; 
	printf("\n%8.6f sec\t Gradient by Forward mode\n", elap);
	elap = float(toc_gpu_grad) / CLOCKS_PER_SEC; 
	printf("\n%8.6f sec\t Gradient by cuda Forward mode\n", elap);

	delete [] s;
	delete [] S;
	delete [] tN;
	targetRod.Destroy();
	ADS::Clear();
	Rod::Clear();
}

int main()
{
	clock_t begin = clock();
	TestCase4(32*1+1);  // 430 is maximum
	clock_t end = clock();

	float elap = float( end - begin ) / CLOCKS_PER_SEC;
	printf("\n\nTotal Elapse = %f sec\n", elap);
	return 0;
}