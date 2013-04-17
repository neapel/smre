#ifndef __CHAMBOLLE_POCK_CL_H__
#define __CHAMBOLLE_POCK_CL_H__

#include "chambolle_pock.h"
#include "multi_array_fft.h"

#include <vexcl/vexcl.hpp>
#include <vexcl/fft.hpp>
#include <vexcl/random.hpp>

template<class T>
struct chambolle_pock<GPU_IMPL, T> : public impl<T> {
	typedef typename vex::cl_vector_of<T, 2>::type T2;

	typedef vex::vector<T> A;
	typedef vex::vector<T2> A2;

	using impl<T>::p;

	struct constraint {
		size_t k_size;
		A2 f_k, f_adj_k;
		A y;
		T q, shift_q;

		constraint(size_t k_size, size2_t size)
		: k_size(k_size),
		  f_k(size[0]*size[1]), f_adj_k(size[0]*size[1]), y(size[0]*size[1]),
		  q(-1), shift_q(0) {}
	};	

	VEX_FUNCTION(norm, T(T2),
		"return dot(prm1, prm1);");
	VEX_FUNCTION(complex_mul, T2(T2, T2),
		"return (float2)("
		"prm1.x * prm2.x - prm1.y * prm2.y,"
		"prm1.x * prm2.y + prm1.y * prm2.x);");
	VEX_FUNCTION(soft_shrink, T(T, T),
		"if(prm1 < -prm2) return prm1 + prm2;"
		"if(prm1 > prm2) return prm1 - prm2;"
		"return 0;");

	vex::FFT<T, T2> fft;
	vex::FFT<T2, T> ifft;
	const vex::Reductor<T, vex::MAX> max;
	const size_t size_1d;
	T total_norm;
	A2 temp;
	std::vector<constraint> constraints;
	resolvent_impl<GPU_IMPL, T> *resolvent;


	chambolle_pock(const params<T> &p)
	: impl<T>(p),
	  fft({p.size[0], p.size[1]}),
	  ifft({p.size[0], p.size[1]}, vex::inverse),
	  size_1d(p.size[0] * p.size[1]),
	  temp(size_1d),
	  resolvent(p.resolvent->gpu_runner(p.size)) {
		update_kernels();
	}

	void update_kernels() {
		constraints.clear();
		total_norm = 0;
		for(auto k_size : p.kernel_sizes) {
			// TODO: use GPU for that.
			const auto v = 1 / (M_SQRT2 * k_size);
			boost::multi_array<T, 2> k(p.size), adj_k(p.size);
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
			constraints.emplace_back(k_size, p.size);
			auto &c = constraints.back();
			A k_(size_1d, k.data()), adj_k_(size_1d, adj_k.data());
			c.f_k = fft(k_);
			c.f_adj_k = fft(adj_k_);
			// Calculate max norm of transformed kernel
			total_norm += max(norm(c.f_k));
		}
	
		if(p.penalized_scan)
			for(auto &c : constraints)
				c.shift_q = sqrt(log(1.0 * size_1d / pow(c.k_size, 2)));
		calc_q();
	}

	void convolve(const A2 &kernel, const A &in, A &out) {
		temp = fft(in);
		convolve(kernel, temp, out);
	}

	void convolve(const A2 &kernel, const A2 &in, A &out) {
		temp = complex_mul(kernel, in);
		out = ifft(temp);
	}

	void calc_q() {
		// If needed, calculate `q/sigma` value.
		float q = impl<T>::cached_q([&](std::vector<std::vector<T>> &k_qs){
			A data(size_1d), convolved(size_1d);
			A2 f_data(size_1d);
			vex::RandomNormal<T> random;
			std::random_device dev;
			std::mt19937 seed(dev());
			for(size_t i = 0 ; i < p.monte_carlo_steps ; i++) {
				data = random(vex::element_index(), seed());
				f_data = fft(data);
				for(size_t j = 0 ; j < constraints.size() ; j++) {
					convolve(constraints[j].f_k, f_data, convolved);
					auto norm_inf = max(fabs(convolved));
					auto k_q = norm_inf - constraints[j].shift_q;
					k_qs[j].push_back(k_q);
				}
			}
		});
		for(auto &c : constraints)
			c.q = q + c.shift_q;
	}


	virtual boost::multi_array<T, 2> run(const boost::multi_array<T, 2> &Y__) {
		auto Y_ = Y__;
#if DEBUG_WATERMARK
		for(size_t i0 = 0 ; i0 < 20 ; i0++)
			for(size_t i1 = 0 ; i1 < 20 ; i1++)
				Y_[i0 + 20][i1 + 20] = 0;
#endif
		A Y(size_1d, Y_.data()), out(size_1d);
		run(Y, out);
		boost::multi_array<T, 2> out_(p.size);
		copy(out, out_.data());
		return out_;
	}

	#define debug(buffer, n) \
		if(p.debug) {\
			std::cerr << #buffer << " (" << #n << "=" << (n) << ")" << std::endl; \
			boost::multi_array<T,2> __data(p.size); \
			copy(buffer, __data.data()); \
			impl<T>::debug_log.emplace_back(__data, #buffer); \
		}

	virtual void run(const A &Y, A &out) {
		A x(Y), bar_x(Y), old_x(size_1d), w(size_1d), convolved(size_1d);
		A2 fft_bar_x(size_1d);

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
			w = 0.0f;
			// transform bar_x for convolutions
			fft_bar_x = fft(bar_x);
			for(auto &c : constraints) {
				// convolve bar_x with kernel
				convolve(c.f_k, fft_bar_x, convolved);
				debug(convolved,n)
				// calculate new y_i
				convolved *= sigma;
				debug(convolved,n)
				c.y += convolved;
				debug(c.y,n)
				c.y = soft_shrink(c.y, c.q * sigma);
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
		out = Y - x;
		debug(out,0)
	}

	#undef debug
};
#endif
