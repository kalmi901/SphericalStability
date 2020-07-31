#ifndef PERTHREAD_SYSTEMDEFINITION_H
#define PERTHREAD_SYSTEMDEFINITION_H
#include <cuda_runtime.h>
#include "MPGOS/MPGOS_Overloaded_MathFunction.cuh"

#define PI 3.14159265358979323846

// SYSTEM
__device__ void PerThread_OdeFunction(int tid, int NT, double* F, double* X, double T, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	double rx1 = 1.0 / X[0];
	double p = pow(rx1, cPAR[10]);

	double s1;
	double c1;
	sincospi(2.0*T, &s1, &c1);

	double s2 = sin(2.0*cPAR[11] * PI*T + cPAR[12]);
	double c2 = cos(2.0*cPAR[11] * PI*T + cPAR[12]);

	double N;
	double D;
	double rD;


	N = (cPAR[0] + cPAR[1] * X[1])*p - cPAR[2] * (1.0 + cPAR[9] * X[1]) - cPAR[3] * rx1 - cPAR[4] * X[1] * rx1 - 1.5*(1.0 - cPAR[9] * X[1] * (1.0 / 3.0))*X[1] * X[1] - (cPAR[5] * s1 + cPAR[6] * s2) * (1.0 + cPAR[9] * X[1]) - X[0] * (cPAR[7] * c1 + cPAR[8] * c2);
	D = X[0] - cPAR[9] * X[0] * X[1] + cPAR[4] * cPAR[9];
	rD = 1.0 / D;

	F[0] = X[1];
	F[1] = N * rD;

	// BLA
	double BL;
	double A;
	double B;
	double R;
	

	for (int i = 0; i < NM; i++)
	{
		R = X[0] * cPAR[14];
		BL = 2 * MPGOS::FMIN(MPGOS::FMAX(cPAR[21], cPAR[22]), R / (2 * sPAR[0 + i * 5]));
		D = 1 / (1 + BL / R);

		A = -sPAR[1 + i * 5] * F[1] * rx1 +
			(sPAR[4 + i * 5] * 0.5 * cPAR[3] + 0.5*cPAR[4] * X[1] * (sPAR[4 + i * 5] - sPAR[3 + i * 5] * D)) * rx1*rx1*rx1;
		B = 3 * X[1] * rx1 + 0.4 * cPAR[4] * rx1*rx1 *(sPAR[2 + i * 5] * D - sPAR[4 + i * 5]);

		F[i + 2] = X[i + NM + 2];
		F[i + NM + 2] = -B * X[i + NM +2] - A * X[i + 2];
	}
	
	/*if (tid == 0)
	{
		printf("%f, %f, %f \n", T, F[0], F[1]);
	}*/
}

// EVENTS
__device__ void PerThread_EventFunction(int tid, int NT, double* EF, double* X, double T, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	EF[0] = X[1];
}

__device__ void PerThread_ActionAfterEventDetection(int tid, int NT, int IDX, int CNT, double &T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{

}

// ACCESSORIES
__device__ void PerThread_ActionAfterSuccessfulTimeStep(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{

}

__device__ void PerThread_Initialization(int tid, int NT, double T, double &dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{

}

__device__ void PerThread_Finalization(int tid, int NT, double T, double dT, double* TD, double* X, double* cPAR, double* sPAR, int* sPARi, double* ACC, int* ACCi)
{
	TD[0] = T;
	printf("Thread %d is ready \n", tid);
}

#endif