#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

constexpr int			   NM = 5;				// Number of Investigated Modes
constexpr double Perturbation = 1e-3;

#include "SphericalStability_SystemDefinition.cuh"
#include "MPGOS/SingleSystem_PerThread.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Physical control parameters
const int NumberOfFrequency1 = 101;
const int NumberOfFrequency2 = 101;
const int NumberOfAmplitude1 = 2;
const int NumberOfAmplitude2 = 2;

// Solver Configuration
#define SOLVER RKCK45 // RK4, RKCK45
const int NT	= NumberOfFrequency1 * NumberOfFrequency2; // NumberOfThreads
const int SD	= 2+2*NM;// SystemDimension
const int NCP	= 23;    // NumberOfControlParameters
const int NSP	= 5*NM;	 // NumberOfSharedParameters
const int NISP	= 0;     // NumberOfIntegerSharedParameters
const int NE	= 1;     // NumberOfEvents
const int NA	= 2;     // NumberOfAccessories
const int NIA	= 0;     // NumberOfIntegerAccessories
const int NDO	= 0;     // NumberOfPointsOfDenseOutput

void Linspace(vector<double>&, double, double, int);
void Logspace(vector<double>&, double, double, int);
void FillSolverObject(ProblemSolver<NT, SD, NCP, NSP, NISP, NE, NA, NIA, NDO, SOLVER, double>&, const vector<double>&, const vector<double>&, const vector<double>&, const vector<double>&, int, int);
void PerturbateSolverObject(ProblemSolver<NT, SD, NCP, NSP, NISP, NE, NA, NIA, NDO, SOLVER, double>&, int);

int main()
{

	int BlockSize = 64;

	vector<double> Frequency1(NumberOfFrequency1, 0);
	vector<double> Frequency2(NumberOfFrequency2, 0);
	vector<double> Amplitude1(NumberOfAmplitude1, 0);
	vector<double> Amplitude2(NumberOfAmplitude2, 0);

	Logspace(Frequency1, 20.0, 2000.0, NumberOfFrequency1);
	Logspace(Frequency2, 20.0, 2000.0, NumberOfFrequency2);
	Linspace(Amplitude1, 0.0, 2.0, NumberOfAmplitude1);
	Linspace(Amplitude2, 0.0, 2.0, NumberOfAmplitude2);

	// Setup CUDA a device
	ListCUDADevices();

	int MajorRevision = 3;
	int MinorRevision = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);

	PrintPropertiesOfSpecificDevice(SelectedDevice);

	// Solver Object configuration
	int NumberOfProblems = NumberOfFrequency1 * NumberOfFrequency2 * NumberOfAmplitude1 * NumberOfAmplitude2;
	int NumberOfThreads = NT;

	ProblemSolver<NT, SD, NCP, NSP, NISP, NE, NA, NIA, NDO, SOLVER, double> CheckSphericalStability(SelectedDevice);

	CheckSphericalStability.SolverOption(ThreadsPerBlock, BlockSize);
	CheckSphericalStability.SolverOption(RelativeTolerance, 0, 1e-12);
	CheckSphericalStability.SolverOption(AbsoluteTolerance, 1, 1e-12);
	for (int i = 2; i < SD; i++)	// Set tolerance of Surface Dynamics
	{
		CheckSphericalStability.SolverOption(RelativeTolerance, 0, 1e-12);
		CheckSphericalStability.SolverOption(AbsoluteTolerance, 1, 1e-12);
	}
	CheckSphericalStability.SolverOption(EventDirection, 0, -1);
	CheckSphericalStability.SolverOption(EventStopCounter, 0, 1);
	CheckSphericalStability.SolverOption(DenseOutputTimeStep, -1e-2);

	// SIMULATIONS ------------------------------------------------------------------------------------

	int NumberOfSimulationLaunches = NumberOfProblems / NumberOfThreads;
	int ProblemStartIndex;

	vector< vector<double>> CollectedData;
	CollectedData.resize(NumberOfThreads, vector<double>(NM + 10, 0));
	// 6 physical paramters +
	// 1 initial time of the stability analysis
	// 1 end time of the stability analysis
	// 2 state varialbes (x1, x2) before the stability analysis
	// NM growth rates for each investiageted modes

	double ActualPA1;
	double ActualPA2;
	clock_t SimulationStart = clock();
	for (int LaunchCounter = 1; LaunchCounter < NumberOfSimulationLaunches; LaunchCounter++)
	{
		// Fill Solver Object
		ProblemStartIndex = LaunchCounter * NumberOfThreads;
		FillSolverObject(CheckSphericalStability, Frequency1, Frequency2, Amplitude1, Amplitude2, ProblemStartIndex, NumberOfThreads);
		CheckSphericalStability.SynchroniseFromHostToDevice(All);

		// Generate a unique filename for the current launch
		stringstream StreamFilename;
		StreamFilename.precision(2);
		StreamFilename.setf(ios::fixed);

		ActualPA1 = CheckSphericalStability.GetHost(0, ControlParameters, 15);
		ActualPA2 = CheckSphericalStability.GetHost(0, ControlParameters, 17);
		StreamFilename << "SphericalStability_PA1_" << ActualPA1 << "_PA2_" << ActualPA2 << ".txt";

		string Filename = StreamFilename.str();
		remove(Filename.c_str());

		// Collect physical parameters
		for (int tid = 0; tid < NumberOfThreads; tid++)
		{
			CollectedData[tid][0] = CheckSphericalStability.GetHost(tid, ControlParameters, 15);
			CollectedData[tid][1] = CheckSphericalStability.GetHost(tid, ControlParameters, 16);
			CollectedData[tid][2] = CheckSphericalStability.GetHost(tid, ControlParameters, 17);
			CollectedData[tid][3] = CheckSphericalStability.GetHost(tid, ControlParameters, 18);
			CollectedData[tid][4] = CheckSphericalStability.GetHost(tid, ControlParameters, 19);
			CollectedData[tid][5] = CheckSphericalStability.GetHost(tid, ControlParameters, 20);
		}
		cout << "LaunchCounter: " << LaunchCounter << "/" << NumberOfSimulationLaunches - 1 << " ... ";
		cout << "Transient Iterations ... ";
		for (int i = 0; i < 1024; i++)
		{
			//cout << "Transient Iteration: " << i << endl;
			CheckSphericalStability.Solve();
			CheckSphericalStability.InsertSynchronisationPoint();
			CheckSphericalStability.SynchroniseSolver();
		}
	
		// Collect date aifter the treandient iteration
		CheckSphericalStability.SynchroniseFromDeviceToHost(TimeDomain);
		CheckSphericalStability.SynchroniseFromDeviceToHost(Accessories);
		CheckSphericalStability.InsertSynchronisationPoint();
		CheckSphericalStability.SynchroniseSolver();
		for (int tid = 0; tid < NumberOfThreads; tid++)
		{
			CollectedData[tid][6] = CheckSphericalStability.GetHost(tid, TimeDomain, 0);
			CollectedData[tid][8] = CheckSphericalStability.GetHost(tid, Accessories, 0);
			CollectedData[tid][9] = CheckSphericalStability.GetHost(tid, Accessories, 1);
		}
			
		// Stability analysis and data collection
		PerturbateSolverObject(CheckSphericalStability, NumberOfThreads);
		cout << "Stability Iterations ... ";
		for (int i = 0; i < 32; i++)
		{
			//cout << "Stability Iteration: " << i << endl;
			CheckSphericalStability.Solve();
			CheckSphericalStability.InsertSynchronisationPoint();
			CheckSphericalStability.SynchroniseSolver();
		}

		CheckSphericalStability.SynchroniseFromDeviceToHost(All);
		CheckSphericalStability.InsertSynchronisationPoint();
		CheckSphericalStability.SynchroniseSolver();
		cout << "Done" << endl;
		for (int tid = 0; tid < NumberOfThreads; tid++)
		{
			CollectedData[tid][7] = CheckSphericalStability.GetHost(tid, TimeDomain, 0);

			// Calculate Growth rates
			for (int i = 0; i < NM; i++)
			{
				CollectedData[tid][10 + i] = log(abs(CheckSphericalStability.GetHost(tid, ActualState, i + 2)) / Perturbation) / (CollectedData[tid][7] - CollectedData[tid][6]);
			}
		}
		
		// Save collected data to file
		ofstream DataFile;
		DataFile.open(Filename.c_str(), std::fstream::app);
		int Width = 18;
		DataFile.precision(10);
		DataFile.flags(ios::scientific);

		for (int tid = 0; tid < NumberOfThreads; tid++)
		{
			for (int col = 0; col < 10 + NM; col++)
			{
				if (col < (10 + NM - 1))
				{
					DataFile.width(Width); DataFile << CollectedData[tid][col] << ',';
				}
				else
				{
					DataFile.width(Width); DataFile << CollectedData[tid][col];
				}
			}
			DataFile << '\n';
		}
		DataFile.close();
	}
	clock_t SimulationEnd = clock();
		cout << "Total simulation time: " << 1000.0*(SimulationEnd - SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	return 0;
}


// ------------------------------------------------------------------------------------------------

void Linspace(vector<double>& x, double B, double E, int N)
{
	double Increment;

	x[0] = B;

	if (N > 1)
	{
		x[N - 1] = E;
		Increment = (E - B) / (N - 1);

		for (int i = 1; i < N - 1; i++)
		{
			x[i] = B + i * Increment;
		}
	}
}

void Logspace(vector<double>& x, double B, double E, int N)
{
	x[0] = B;

	if (N > 1)
	{
		x[N - 1] = E;
		double ExpB = log10(B);
		double ExpE = log10(E);
		double ExpIncr = (ExpE - ExpB) / (N - 1);
		for (int i = 1; i < N - 1; i++)
		{
			x[i] = pow(10, ExpB + i * ExpIncr);
		}
	}
}

// ------------------------------------------------------------------------------------------------
void FillSolverObject(ProblemSolver<NT, SD, NCP, NSP, NISP, NE, NA, NIA, NDO, SOLVER, double>& Solver, const vector<double>& F1_Values, const vector<double>& F2_Values, const vector<double>& PA1_Values, const vector<double>& PA2_Values, int ProblemStartIndex, int NumberOfThreads)
{
	// Declaration of physical control parameters
	double P1; // pressure amplitude1 [bar]
	double P2; // frequency1          [kHz]
	double P3; // pressure amplitude2 [bar]
	double P4; // frequency2          [kHz]

	// Declaration of constant parameters
	double P5 = 0.0*PI;	// phase shift          [-]
	double P6 = 10.0;	// equilibrium radius   [mum]
	double P7 = 1.0;	// ambient pressure     [bar]
	double P9 = 1.4;	// polytrophic exponent [-]

	// Material properties
	double Pv = 3.166775638952003e+03;
	double Rho = 9.970639504998557e+02;
	double ST = 0.071977583160056;
	double Vis = 8.902125058209557e-04;
	double CL = 1.497251785455527e+03;

	// Auxiliary variables
	double Pinf;
	double PA1;
	double PA2;
	double RE;
	double f1;
	double f2;

	// Set Shared Parameters
	double n = 2.0;
	for (int i = 0; i < NM; i++)
	{
		Solver.SetHost(SharedParameters, 0 + i * 5, n);
		Solver.SetHost(SharedParameters, 1 + i * 5, n - 1);
		Solver.SetHost(SharedParameters, 2 + i * 5, n * (n + 2)*(n + 2));
		Solver.SetHost(SharedParameters, 3 + i * 5, n * (n - 1)*(n + 2));
		Solver.SetHost(SharedParameters, 4 + i * 5, (n - 1)*(n + 1)*(n + 2));
		n +=1.0;
	}

	int ProblemNumber = 0;
	int GlobalCounter = 0;
	for (auto const& CP4 : PA2_Values) // pressure amplitude2 [bar]
	{
		for (auto const& CP3 : PA1_Values) // pressure amplitude1 [bar]
		{
			for (auto const& CP2 : F2_Values) // frequency2 [kHz]
			{
				for (auto const& CP1 : F1_Values) // frequency1 [kHz]
				{
					if (GlobalCounter < ProblemStartIndex)
					{
						GlobalCounter++;
						continue;
					}

					// Update physical parameters
					P1 = CP3;
					P2 = CP1;
					P3 = CP4;
					P4 = CP2;

					Solver.SetHost(ProblemNumber, TimeDomain, 0, 0);
					Solver.SetHost(ProblemNumber, TimeDomain, 1, 1e10);

					// Initial conditions are the equilibrium condition y1=1; y2=0;
					Solver.SetHost(ProblemNumber, ActualState, 0, 1.0);
					Solver.SetHost(ProblemNumber, ActualState, 1, 0.0);

					// BLA - Initial Perturbation
					for (int i = 2; i < 2 + 2 * NM; i++)
					{
						Solver.SetHost(ProblemNumber, ActualState, i, 0.0);
					}

					// Scaling of physical parameters to SI
					Pinf = P7 * 1e5;
					PA1	 = P1 * 1e5;
					PA2	 = P3 * 1e5;
					RE	 = P6 / 1e6;

					// Scale to angular frequency
					f1 = 2.0*PI*(P2 * 1000);
					f2 = 2.0*PI*(P4 * 1000);

					// System coefficients and other, auxiliary parameters
					Solver.SetHost(ProblemNumber, ControlParameters, 0, (2.0*ST / RE + Pinf - Pv) * pow(2.0*PI / RE / f1, 2.0) / Rho);
					Solver.SetHost(ProblemNumber, ControlParameters, 1, (1.0 - 3.0*P9) * (2 * ST / RE + Pinf - Pv) * (2.0*PI / RE / f1) / CL / Rho);
					Solver.SetHost(ProblemNumber, ControlParameters, 2, (Pinf - Pv) * pow(2.0*PI / RE / f1, 2.0) / Rho);
					Solver.SetHost(ProblemNumber, ControlParameters, 3, (2.0*ST / RE / Rho) * pow(2.0*PI / RE / f1, 2.0));
					Solver.SetHost(ProblemNumber, ControlParameters, 4, (4.0*Vis / Rho / pow(RE, 2.0)) * (2.0*PI / f1));
					Solver.SetHost(ProblemNumber, ControlParameters, 5, PA1 * pow(2.0*PI / RE / f1, 2.0) / Rho);
					Solver.SetHost(ProblemNumber, ControlParameters, 6, PA2 * pow(2.0*PI / RE / f1, 2.0) / Rho);
					Solver.SetHost(ProblemNumber, ControlParameters, 7, (RE*f1*PA1 / Rho / CL) * pow(2.0*PI / RE / f1, 2.0));
					Solver.SetHost(ProblemNumber, ControlParameters, 8, (RE*f2*PA2 / Rho / CL) * pow(2.0*PI / RE / f1, 2.0));
					Solver.SetHost(ProblemNumber, ControlParameters, 9, RE*f1 / (2.0*PI) / CL);
					Solver.SetHost(ProblemNumber, ControlParameters, 10, 3.0*P9);
					Solver.SetHost(ProblemNumber, ControlParameters, 11, P4 / P2);
					Solver.SetHost(ProblemNumber, ControlParameters, 12, P5);

					Solver.SetHost(ProblemNumber, ControlParameters, 13, 2.0*PI / f1);	// tref
					Solver.SetHost(ProblemNumber, ControlParameters, 14, RE);			// Rref

					Solver.SetHost(ProblemNumber, ControlParameters, 15, P1);
					Solver.SetHost(ProblemNumber, ControlParameters, 16, P2);
					Solver.SetHost(ProblemNumber, ControlParameters, 17, P3);
					Solver.SetHost(ProblemNumber, ControlParameters, 18, P4);
					Solver.SetHost(ProblemNumber, ControlParameters, 19, P5);
					Solver.SetHost(ProblemNumber, ControlParameters, 20, P6);

					Solver.SetHost(ProblemNumber, ControlParameters, 21, PA1 != 0 ? sqrt(Vis / Rho / f1) : 0.0);
					Solver.SetHost(ProblemNumber, ControlParameters, 22, PA2 != 0 ? sqrt(Vis / Rho / f2) : 0.0);

					ProblemNumber++;

					if (ProblemNumber == NumberOfThreads)
						goto ExitSolverFilling;
				}
			}
		}
	}
	ExitSolverFilling:;
}

void PerturbateSolverObject(ProblemSolver<NT, SD, NCP, NSP, NISP, NE, NA, NIA, NDO, SOLVER, double>& Solver, int NumberOfThreads)
{
	Solver.SynchroniseFromDeviceToHost(ActualState);
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();
	int ProblemNumber = 0;
	while (ProblemNumber < NumberOfThreads)
	{
		for (int i = 2; i < 2 + 2 * NM; i++)
		{
			if (i < 2 + NM) { Solver.SetHost(ProblemNumber, ActualState, i, Perturbation); }
			else			{ Solver.SetHost(ProblemNumber, ActualState, i, 0.0);  }
		}
		ProblemNumber++;
	}
	Solver.SynchroniseFromHostToDevice(ActualState);
	Solver.InsertSynchronisationPoint();
	Solver.SynchroniseSolver();
}