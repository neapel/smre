#ifndef __CHAMBOLLE_POCK_CL_H__
#define __CHAMBOLLE_POCK_CL_H__

#include <boost/format.hpp>

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
	bool initialized = false;

	chambolle_pock(const params<T> &p)
	: impl<T>(p),
	  size_1d(p.size[0] * p.size[1]),
	  resolvent(p.resolvent->gpu_runner(p.size)) {
		if(p.use_fft) convolution.reset(new gpu_fft_convolver<T>(p.size));
		else convolution.reset(new gpu_sat_convolver<T>(p.size));
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
		impl<T>::q = impl<T>::cached_q([&](std::vector<std::vector<T>> &k_qs){
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
				this->progress(double(i) / p.monte_carlo_steps, "Monte Carlo simulation for q");
			}
		});
		for(auto &c : constraints)
			c.q = impl<T>::q + c.shift_q;
	}


	bool current(const A &a, size_t s) {
		if(impl<T>::current_cb) {
			boost::multi_array<T, 2> a_(p.size);
			copy(a, a_.data());
			return impl<T>::current_cb(a_, s);
		}
		return true;
	}

	void debug(const A &a, std::string d) {
		if(impl<T>::debug_cb) {
			boost::multi_array<T, 2> a_(p.size);
			copy(a, a_.data());
			impl<T>::debug_cb(a_, d);
		}
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

	virtual void run(A &Y, A &out) {
		A x(Y), bar_x(Y), old_x(size_1d), w(size_1d), convolved(size_1d);

		if(!initialized) {
			update_kernels();
			initialized = true;
		}

		debug(x, "x_in");
		for(auto &c : constraints) {
			c.y = 0;
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
			for(size_t i = 0 ; i < constraints.size() ; i++) {
				auto c = constraints[i];
				// convolve bar_x with kernel
				convolution->conv(f_bar_x, c.k, convolved);
				debug(convolved, str(boost::format("convolved_%d") % i));
				// calculate new y_i
				c.y = soft_shrink(c.y + convolved * sigma, c.q * sigma);
				debug(c.y, str(boost::format("y_%d") % i));
				// convolve y_i with conjugate transpose of kernel
				auto f_y = convolution->prepare_image(c.y);
				convolution->conv(f_y, c.adj_k, convolved);
				debug(convolved, str(boost::format("adj_convolved_%d") % i));
				// accumulate
				w += convolved;
				debug(w, str(boost::format("w_%d") % i));
				this->progress(double(n * constraints.size() + i) / (p.max_steps * constraints.size()),
					str(boost::format("Chambolle-Pock step %d") % n));
			}

			old_x = x;
			bar_x = x - Y - w*tau;
			debug(bar_x, "resolv_in");
			resolvent->evaluate(tau, bar_x, x);
			x += Y;
			debug(x, "resolv_out");
			const T theta = 1 / sqrt(1 + 2 * tau * resolvent->gamma);
			tau *= theta;
			sigma /= theta;
			bar_x = x + (x - old_x) * theta;
			debug(bar_x, "bar_x");

			out = Y - x;
			if(!current(out, n)) break;
		}
	}

	#undef debug
};
#endif
