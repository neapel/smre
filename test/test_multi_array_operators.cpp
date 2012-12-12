#include "multi_array_operators.h"
#include "multi_array_print.h"
#include "multi_array_fft.h"
#include <iostream>
#include <typeinfo>
#include <random>
using namespace boost;
using namespace std;
using namespace mimas;



int main(int, char**) {
	multi_array<double, 2> A(extents[4][3]);
	iota(A, 1);
	double v = 8;
	assert(A[2][1] == v);

	A *= 4.0;
	v *= 4.0; assert(A[2][1] == v);

	A /= 3.0;
	v /= 3.0; assert(A[2][1] == v);

	A += 4.0;
	v += 4.0; assert(A[2][1] == v);

	A -= 1.0;
	v -= 1.0; assert(A[2][1] == v);

	// Test convolution and its adjoint

	const auto width = 10, height = 10;
	const auto c_width = width, c_height = height/2+1;
	const auto size = extents[width][height], c_size = extents[c_width][c_height];
	const float scale = 1.0f/(width*height);

	multi_array<float,2> U(size), V(size), kernel_padded(size), lhs(size), rhs(size);
	multi_array<complex<float>, 2> U_hat(c_size), V_hat(c_size),kernel_hat_padded(c_size),adjoint_kernel_hat_padded(c_size), lhs_hat(c_size), rhs_hat(c_size);
	multi_array<float,2> kernel(extents[5][5]);
	fill(kernel, 0.2);

	default_random_engine generator;
	normal_distribution<double> distribution(2,1.0);


	// kernel ffts
	kernel_pad(kernel,kernel_padded);
	fftw::forward(kernel_padded, kernel_hat_padded)();

	fill(kernel_padded,0);
	kernel_pad(kernel,kernel_padded,1);
	fftw::forward(kernel_padded, adjoint_kernel_hat_padded)();

	for(int dx = 0; dx < width; dx++)
		for(int dy = 0; dy < height; dy++){
			U[dx][dy] = distribution(generator);
			V[dx][dy] = distribution(generator);}

	// signal ffts
	fftw::forward(U, U_hat)();
	fftw::forward(V, V_hat)();

	// convolution in Fourier domain
	for(int dx = 0; dx < c_width; dx++)
		for(int dy = 0; dy < c_height; dy++){
			lhs_hat[dx][dy] = U_hat[dx][dy] * kernel_hat_padded[dx][dy] * scale;
			rhs_hat[dx][dy] = V_hat[dx][dy] * adjoint_kernel_hat_padded[dx][dy] * scale;
		}

	// backwards ffts
	fftw::backward(lhs_hat, lhs)();
	fftw::backward(rhs_hat, rhs)();



	float a = 0, b = 0;
	for(int dx = 0; dx < width; dx++)
		for(int dy = 0; dy < height; dy++){
			a += lhs[dx][dy] * V[dx][dy];
			b += rhs[dx][dy] * U[dx][dy];
		}

	cout << "<v,Ku> = " << a << " and <K*v,u> = " << b << endl;

	return 0;
}
