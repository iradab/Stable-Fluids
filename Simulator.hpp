#include <Eigen/SparseCore>
#include <Eigen/Sparse>

class Simulator {
public:
	Simulator();

	~Simulator();
	void set_boundary(int b, float x[64][64]);
	void SwapV(float U1[2][64][64], float U0[2][64][64]);
	void SwapS(float S_new[64][64], float S_prev[64][64]);
	void Dissipate(float S0[64][64], float S1[64][64]);
	void Sstep(float S[64][64], float Sprev[64][64], float U[2][64][64]);
	void Vstep(float U[2][64][64], float U0[2][64][64]);
	void addForce(float Grid0[64][64], float Force[64][64]);
	void Transport(int b, float Grid1[64][64], float Grid0[64][64], float U[2][64][64]);
	void Diffuse(int B, Eigen::SparseMatrix<double> sparseA, float Grid1[64][64], float Grid0[64][64]);
	float LinInterp(float X0[2], float Grid0[64][64]);
	void TraceParticle(float X[2], float X0[2], float U[2][64][64]);
	void Project(float U[2][64][64], float pdiv[2][64][64]);
	const int nDim = 2;
	const int n = 64;
	float Uprev[2][64][64] = { 0 };
	float U[2][64][64] = { 0 };
	float S[64][64] = { 0 };
	float Sprev[64][64] = { 0 };
	float dt{ 0.1 };
	float visc{ 0.1 };
	float kS{ 0.1 }; // diffusion constant
	float aS{ 0.1 }; // dissipation rate
	float Origin[2] = { 0, 0 };
	float Length[2] = { 64, 64 };
	float D[2] = { Length[0] / 64, Length[1] / 64 };
	float F[2][64][64] = { 0 };
	float Ssource[64][64] = { 0 };
	Eigen::SparseMatrix<double> sparseA;
	Eigen::SparseMatrix<double> sparseA_ks;
	Eigen::SparseMatrix<double> sparseA_pr;


};
