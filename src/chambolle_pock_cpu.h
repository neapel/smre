#ifndef __CHAMBOLLE_POCK_CPU_H__
#define __CHAMBOLLE_POCK_CPU_H__


#include "chambolle_pock.h"
#include "multi_array_fft.h"
#include "resolvent.h"

#if HAVE_OPENMP
#include <omp.h>
#endif




template<class T>
struct chambolle_pock<CPU_IMPL, T> : public impl<T> {
	typedef std::complex<T> T2;

	typedef boost::multi_array<T, 2> A;
	typedef boost::multi_array<T2, 2> A2;

	using impl<T>::p;

	struct constraint {
		// size of the box kernel.
		size_t k_size;
		// FFT of the kernel and of the adjungated kernel.
		A2 f_k, f_adj_k;
		// y for this constraint
		A y;
		// specific q for this constraint.
		T q, shift_q;

		constraint(size_t k_size, size2_t size, size2_t fft_size)
		: k_size(k_size), f_k(fft_size), f_adj_k(fft_size), y(size), q(-1), shift_q(0) {}
	};

	inline T soft_shrink(T x, T q) {
		if(x < -q) return x + q;
		if(x > q) return x - q;
		return 0;
	}

	fftw::plan<T, T2, 2> fft;
	fftw::plan<T2, T, 2> ifft;
	const size2_t fft_size;
	const T scale;
	T total_norm;
	A2 temp;
	std::vector<constraint> constraints;
	resolvent_impl<CPU_IMPL, T> *resolvent;


	chambolle_pock(const params<T> &p)
	: impl<T>(p),
	  fft(p.size), ifft(p.size),
	  fft_size{{p.size[0], p.size[1]/2+1}},
	  scale(1.0 / (p.size[0] * p.size[1])),
	  temp(fft_size),
	  resolvent(p.resolvent->cpu_runner(p.size)) {
		update_kernels();
	}

	void update_kernels() {
		using namespace boost;
		using namespace mimas;
		constraints.clear();
		total_norm = 0;
		for(size_t k_size : p.kernel_sizes) {
			const auto v = 1 / (M_SQRT2 * k_size);
			A k(p.size), adj_k(p.size);
			fill(k, 0);
			fill(adj_k, 0);
			// k[-0,-1,...,-(k_size-1)] = v; rest 0.
			// adj_k[i] = k[-i] => adj_k[0,1,...,(k_size-1)] = v;
			for(size_t i0 = 0 ; i0 < k_size ; i0++)
				for(size_t i1 = 0 ; i1 < k_size ; i1++) {
					k[(p.size[0] - i0) % p.size[0]][(p.size[1] - i1) % p.size[1]] = v;
					adj_k[i0][i1] = v;
				}
			// store FFT of that
			constraints.emplace_back(k_size, p.size, fft_size);
			auto &c = constraints.back();
			fft(k, c.f_k);
			fft(adj_k, c.f_adj_k);
			// Calculate max norm of transformed kernel
			total_norm += max(norm<T>(c.f_k));
		}

		if(p.penalized_scan)
			for(auto &c : constraints)
				c.shift_q = sqrt(log(1.0 * p.size[0] * p.size[1] / pow(c.k_size, 2)));
		calc_q();
	}

	void convolve(const A2 &kernel, const A &in, A &out) {
		fft(in, temp);
		convolve(kernel, temp, out);
	}

	void convolve(const A2 &kernel, const A2 &in, A &out) {
		temp = mimas::multi_func<T2>(kernel, in,
			[&](T2 a, T2 b){return a * b * scale;});
		ifft(temp, out);
	}

	void calc_q() {
		using namespace std;
		const T q = impl<T>::cached_q([&](std::vector<std::vector<T>> &k_qs){
			A data(p.size), convolved(p.size);
			A2 f_data(fft_size);
			random_device dev;
			mt19937 gen(dev());
			normal_distribution<T> dist(/*mean*/0, /*stddev*/1);
			for(size_t i = 0 ; i < p.monte_carlo_steps ; i++) {
				for(auto row : data) for(auto &x : row) x = dist(gen);
				fft(data, f_data);
				for(size_t j = 0 ; j < constraints.size() ; j++) {
					convolve(constraints[j].f_k, f_data, convolved);
					auto norm_inf = max(abs(convolved));
					auto k_q = norm_inf - constraints[j].shift_q;
					k_qs[j].push_back(k_q);
				}
			}
		});
		for(auto &c : constraints)
			c.q = q + c.shift_q;
	}


	#define debug(buffer, n) \
		if(p.debug) { \
			std::cerr << #buffer << " (" << #n << "=" << (n) << ")" << std::endl; \
			impl<T>::debug_log.emplace_back(buffer, #buffer); \
		}

	virtual A run(const A &Y_) {
		using namespace mimas;

		auto Y = Y_;
#ifdef DEBUG_WATERMARK
		for(size_t i0 = 0 ; i0 < 20 ; i0++)
			for(size_t i1 = 0 ; i1 < 20 ; i1++)
				Y[i0 + 20][i1 + p.size[1] - 40] = 0;		
#endif

		const int original_threads = omp_get_num_threads();
		if(p.debug) omp_set_num_threads(1);

		A x(Y), bar_x(Y), old_x(p.size), w(p.size), convolved(p.size);
		A2 fft_bar_x(fft_size);

		debug(x,0)
		for(auto &c : constraints) {
			c.y = Y;
			debug(c.y,0)
		}

		T tau = p.tau;
		T sigma = p.sigma;

		// Adjust sigma with norm.
		sigma /= tau * total_norm;

		// Repeat until good enough.
		for(size_t n = 0 ; n < p.max_steps ; n++) {
			// reset accumulator
			fill(w, 0);
			// transform bar_x for convolutions
			fft(bar_x, fft_bar_x);
			for(auto &c : constraints) {
				// convolve bar_x with kernel
				convolve(c.f_k, fft_bar_x, convolved);
				debug(convolved,n)
				// calculate new y_i
				convolved *= sigma;
				debug(convolved,n)
				c.y += convolved;
				debug(c.y,n)
				c.y = mimas::multi_func<T>(c.y,
					[&](T v){return soft_shrink(v, c.q * sigma);});
				debug(c.y,n)
				// convolve y_i with conjugate transpose of kernel
				convolve(c.f_adj_k, c.y, convolved);
				debug(convolved,n)
				// accumulate
				w += convolved;
				debug(w,n)
			}
			old_x = x;
			w *= tau;
			bar_x = x;
			bar_x -= w;
			bar_x -= Y;
			debug(bar_x,n)
			resolvent->evaluate(tau, bar_x, x);
			x += Y;
			debug(x,n)
			const T theta = 1 / sqrt(1 + 2 * tau * resolvent->gamma);
			tau *= theta;
			sigma /= theta;
			bar_x = x;
			bar_x -= old_x;
			bar_x *= theta;
			bar_x += x;
			debug(bar_x,n)
		}
		auto out = Y - x;
		debug(out,0)
		if(p.debug) omp_set_num_threads(original_threads);
		return out;
	}

	#undef debug
};
#endif




