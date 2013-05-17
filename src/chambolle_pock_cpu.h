#ifndef __CHAMBOLLE_POCK_CPU_H__
#define __CHAMBOLLE_POCK_CPU_H__


#include "chambolle_pock.h"
#include "resolvent.h"
#include "convolution.h"


#if HAVE_OPENMP
#include <omp.h>
#endif



template<class T>
struct chambolle_pock<CPU_IMPL, T> : public impl<T> {
	typedef boost::multi_array<T, 2> A;

	using impl<T>::p;

	struct constraint {
		// size of the box kernel.
		size_t k_size;
		std::shared_ptr<prepared_kernel> k, adj_k;
		// y for this constraint
		A y;
		// specific q for this constraint.
		T q, shift_q;

		constraint(size_t k_size, size2_t size,
			std::shared_ptr<prepared_kernel> k, std::shared_ptr<prepared_kernel> adj_k)
		: k_size(k_size), k(k), adj_k(adj_k), y(size), q(-1), shift_q(0) {}
	};

	inline T soft_shrink(T x, T q) {
		if(x < -q) return x + q;
		if(x > q) return x - q;
		return 0;
	}

	T total_norm;
	std::vector<constraint> constraints;
	std::unique_ptr<resolvent_impl<CPU_IMPL, T>> resolvent;
	std::unique_ptr<convolver<A>> convolution;


	chambolle_pock(const params<T> &p)
	: impl<T>(p),
	  resolvent(p.resolvent->cpu_runner(p.size)) {
		if(p.use_fft) convolution.reset(new cpu_fft_convolver<T>(p.size));
		else convolution.reset(new cpu_sat_convolver<T>(p.size));
		update_kernels();
	}

	void update_kernels() {
		constraints.clear();
		total_norm = 0;
		for(auto k_size : p.kernel_sizes) {
			auto prep_k = convolution->prepare_kernel(k_size, false);
			auto adj_prep_k = convolution->prepare_kernel(k_size, true);
			constraints.emplace_back(k_size, p.size, prep_k, adj_prep_k);
			total_norm += k_size / M_SQRT2;
		}
		if(p.penalized_scan)
			for(auto &c : constraints)
				c.shift_q = sqrt(log(1.0 * p.size[0] * p.size[1] / pow(c.k_size, 2)));
		calc_q();
	}

	void calc_q() {
		using namespace mimas;
		using namespace std;
		const T q = impl<T>::cached_q([&](std::vector<std::vector<T>> &k_qs){
			A data(p.size), convolved(p.size);
			random_device dev;
			mt19937 gen(dev());
			normal_distribution<T> dist(/*mean*/0, /*stddev*/1);
			for(size_t i = 0 ; i < p.monte_carlo_steps ; i++) {
				for(auto row : data) for(auto &x : row) x = dist(gen);
				auto f_data = convolution->prepare_image(data);
				for(size_t j = 0 ; j < constraints.size() ; j++) {
					convolution->conv(f_data, constraints[j].k, convolved);
					auto norm_inf = max(abs(convolved));
					auto k_q = norm_inf - constraints[j].shift_q;
					k_qs[j].push_back(k_q);
				}
			}
		});
		for(auto &c : constraints)
			c.q = q + c.shift_q;
	}

	virtual A run(const A &Y_) {
		using namespace mimas;

		auto Y = Y_;
#ifdef DEBUG_WATERMARK
		for(size_t i0 = 0 ; i0 < 20 ; i0++)
			for(size_t i1 = 0 ; i1 < 20 ; i1++)
				Y[i0 + 20][i1 + p.size[1] - 40] = 0;		
#endif

		A x(Y), bar_x(Y), old_x(p.size), w(p.size), out(p.size);

		this->debug(x, "x_in");
		for(auto &c : constraints) {
			fill(c.y, 0);
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
			auto f_bar_x = convolution->prepare_image(bar_x);
			#pragma omp parallel for
			for(size_t i = 0 ; i < constraints.size() ; i++) {
				auto &c = constraints[i];
				// convolve bar_x with kernel
				A convolved(p.size);
				convolution->conv(f_bar_x, c.k, convolved);
				this->debug(convolved, "convolved_i");
				// calculate new y_i
				convolved *= sigma;
				c.y += convolved;
				c.y = mimas::multi_func<T>(c.y,
					[&](T v){return soft_shrink(v, c.q * sigma);});
				this->debug(c.y, "y_i");
				// convolve y_i with conjugate transpose of kernel
				auto f_y = convolution->prepare_image(c.y);
				convolution->conv(f_y, c.adj_k, convolved);
				this->debug(convolved, "adj_convolved_i");
				// accumulate
				#pragma omp critical
				w += convolved;
				this->debug(w, "accum");
			}
			this->progress(double(n) / p.max_steps);

			old_x = x;
			w *= tau;
			bar_x = x; bar_x -= Y; bar_x -= tau;
			this->debug(bar_x, "bar_x");
			resolvent->evaluate(tau, bar_x, x);
			x += Y;
			this->debug(x, "resolv_x");
			const T theta = 1 / sqrt(1 + 2 * tau * resolvent->gamma);
			tau *= theta;
			sigma /= theta;
			bar_x = x;
			bar_x -= old_x;
			bar_x *= theta;
			bar_x += x;
			this->debug(bar_x, "bar_x");

			out = Y;
			out -= x;
			if(!this->current(out, n)) break;
		}
		return out;
	}

	#undef debug
};
#endif




