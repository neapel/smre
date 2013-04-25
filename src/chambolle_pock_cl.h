#ifndef __CHAMBOLLE_POCK_CL_H__
#define __CHAMBOLLE_POCK_CL_H__

#include "convolution.h"

#include "chambolle_pock.h"
#include <vexcl/random.hpp>

template<class T>
struct chambolle_pock<GPU_IMPL, T> : public impl<T> {
	typedef vex::vector<T> A;

	using impl<T>::p;

	struct constraint {
		size_t k_size;
		std::shared_ptr<prepared_kernel> k, adj_k;
		A y;
		T q, shift_q;

		constraint(size_t k_size, size_t size_1d,
			std::shared_ptr<prepared_kernel> k, std::shared_ptr<prepared_kernel> adj_k)
		: k_size(k_size),
		  k(k), adj_k(adj_k), y(size_1d),
		  q(-1), shift_q(0) {}
	};	

	VEX_FUNCTION(soft_shrink, T(T, T),
		"if(prm1 < -prm2) return prm1 + prm2;"
		"if(prm1 > prm2) return prm1 - prm2;"
		"return 0;");

	const size_t size_1d;
	T total_norm;
	std::vector<constraint> constraints;
	std::unique_ptr<resolvent_impl<GPU_IMPL, T>> resolvent;
	std::unique_ptr<convolver<A>> convolution;

	chambolle_pock(const params<T> &p)
	: impl<T>(p),
	  size_1d(p.size[0] * p.size[1]),
	  resolvent(p.resolvent->gpu_runner(p.size)) {
		if(p.use_fft) convolution.reset(new gpu_fft_convolver<T>(p.size));
		else convolution.reset(new gpu_sat_convolver<T>(p.size));
		update_kernels();
	}

	void update_kernels() {
		constraints.clear();
		total_norm = 0;
		for(auto k_size : p.kernel_sizes) {
			auto prep_k = convolution->prepare_kernel(k_size, false);
			auto adj_prep_k = convolution->prepare_kernel(k_size, true);
			constraints.emplace_back(k_size, size_1d, prep_k, adj_prep_k);
			total_norm += k_size / M_SQRT2;
		}
		if(p.penalized_scan)
			for(auto &c : constraints)
				c.shift_q = sqrt(log(1.0 * size_1d / pow(c.k_size, 2)));
		calc_q();
	}

	void calc_q() {
		// If needed, calculate `q/sigma` value.
		T q = impl<T>::cached_q([&](std::vector<std::vector<T>> &k_qs){
			const vex::Reductor<T, vex::MAX> max;
			A data(size_1d), convolved(size_1d);
			vex::RandomNormal<T> random;
			std::random_device dev;
			std::mt19937 seed(dev());
			for(size_t i = 0 ; i < p.monte_carlo_steps ; i++) {
				data = random(vex::element_index(), seed());
				auto f_data = convolution->prepare_image(data);
				for(size_t j = 0 ; j < constraints.size() ; j++) {
					convolution->conv(f_data, constraints[j].k, convolved);
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

	virtual void run(A &Y, A &out) {
		A x(Y), bar_x(Y), old_x(size_1d), w(size_1d), convolved(size_1d);

		debug(x,0)
		for(auto &c : constraints) {
			c.y = 0;
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
			auto f_bar_x = convolution->prepare_image(bar_x);
			for(auto &c : constraints) {
				// convolve bar_x with kernel
				convolution->conv(f_bar_x, c.k, convolved);
				debug(convolved,n)
				// calculate new y_i
				c.y = soft_shrink(c.y + convolved * sigma, c.q * sigma);
				debug(c.y,n)
				// convolve y_i with conjugate transpose of kernel
				auto f_y = convolution->prepare_image(c.y);
				convolution->conv(f_y, c.adj_k, convolved);
				debug(convolved,n)
				// accumulate
				w += convolved;
				debug(w,n)
			}

			old_x = x;
			bar_x = x - Y - w*tau;
			debug(bar_x,n)
			resolvent->evaluate(tau, bar_x, x);
			x += Y;
			debug(x,n)
			const T theta = 1 / sqrt(1 + 2 * tau * resolvent->gamma);
			tau *= theta;
			sigma /= theta;
			bar_x = x + (x - old_x) * theta;
			debug(bar_x,n)
		}
		out = Y - x;
		debug(out,0)
	}

	#undef debug
};
#endif
