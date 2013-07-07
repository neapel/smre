#ifndef __CHAMBOLLE_POCK_CL_H__
#define __CHAMBOLLE_POCK_CL_H__

#include <boost/format.hpp>

#include "convolution.h"
#include "image_variance.h"
#include "chambolle_pock.h"
#include <vexcl/random.hpp>

template<class T>
struct chambolle_pock_gpu : public impl<T> {
	typedef vex::vector<T> A;

	using impl<T>::p;
	using impl<T>::input_stddev;
	using impl<T>::q;

	struct constraint {
		// size of the box kernel.
		size_t k_size;
		std::shared_ptr<prepared_kernel> k, adj_k;
		// y for this constraint
		A y;
		// specific q for this constraint.
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

	const vex::Reductor<T, vex::MAX> max;
	const vex::Reductor<T, vex::SUM> sum;

	const size_t size_1d;
	T total_norm;
	std::vector<constraint> constraints;
	std::shared_ptr<resolvent_gpu<T>> resolv;
	std::shared_ptr<gpu_convolver<T>> convolution;
	bool initialized = false;

	chambolle_pock_gpu(const params<T> &p)
	: impl<T>(p),
	  size_1d(p.size[0] * p.size[1]),
	  resolv(p.resolvent->gpu_runner(p.size)) {
		if(p.use_fft) convolution = std::make_shared<gpu_fft_convolver<T>>(p.size);
		else convolution = std::make_shared<gpu_sat_convolver<T>>(p.size);
	}

	void update_kernels() {
		constraints.clear();
		total_norm = 0;
		for(auto k_size : p.kernel_sizes) {
			auto prep_k = convolution->prepare_kernel(k_size, false);
			auto adj_prep_k = convolution->prepare_kernel(k_size, true);
			constraints.emplace_back(k_size, size_1d, prep_k, adj_prep_k);
			total_norm += k_size * k_size / 2;
		}
		if(p.penalized_scan)
			for(auto &c : constraints)
				c.shift_q = sqrt(log(1.0 * size_1d / pow(c.k_size, 2)));
		calc_q();
	}

	T norm_inf(const A &a) {
		return max(fabs(a));
	}

	T norm_1(const A &a) {
		return sum(fabs(a));
	}

	void calc_q() {
		// If needed, calculate `q/sigma` value.
		q = this->cached_q([&](std::vector<std::vector<T>> &k_qs){
			A data(size_1d), convolved(size_1d);
			vex::RandomNormal<T> random;
			std::random_device dev;
			std::mt19937 seed(dev());
			for(size_t i = 0 ; i < p.monte_carlo_steps ; i++) {
				data = random(vex::element_index(), seed());
				auto f_data = convolution->prepare_image(data);
				for(size_t j = 0 ; j < constraints.size() ; j++) {
					convolution->conv(f_data, constraints[j].k, convolved);
					auto k_q = norm_inf(convolved) - constraints[j].shift_q;
					k_qs[j].push_back(k_q);
				}
				if(i % 10 == 0) this->progress(double(i) / p.monte_carlo_steps, "Monte Carlo simulation for q");
			}
		});
		for(auto &c : constraints)
			c.q = q + c.shift_q;
	}


	bool current(const A &a, size_t s) {
		if(this->current_cb) {
			boost::multi_array<T, 2> a_(p.size);
			copy(a, a_.data());
			return this->current_cb(a_, s);
		}
		return true;
	}

	void debug(const A &a, std::string d) {
		if(this->debug_cb) {
			boost::multi_array<T, 2> a_(p.size);
			copy(a, a_.data());
			this->debug_cb(a_, d);
		}
	}

	void profile_push(std::string name) {
		if(this->profiler)
			this->profiler->tic_cl(name);
	}

	void profile_pop() {
		if(this->profiler)
			this->profiler->toc("");
	}

	virtual boost::multi_array<T, 2> run(const boost::multi_array<T, 2> &Y__) {
		auto Y_ = Y__;
#if DEBUG_WATERMARK
		for(size_t i0 = 0 ; i0 < 20 ; i0++)
			for(size_t i1 = 0 ; i1 < 20 ; i1++)
				Y_[i0 + 20][i1 + 20] = 0;
#endif
		if(p.input_stddev >= 0)
			input_stddev = p.input_stddev;
		else
			input_stddev = median_absolute_deviation(Y_);

		profile_push("gpu run");
		A Y(size_1d, Y_.data()), out(size_1d);
		run(Y, out);
		boost::multi_array<T, 2> out_(p.size);
		copy(out, out_.data());
		profile_pop();
		return out_;
	}

	virtual void run(A &Y, A &out) {
		profile_push("run");
		profile_push("allocate");
			A x(Y), bar_x(Y), old_x(size_1d), w(size_1d), convolved(size_1d);
		profile_pop();

		if(!initialized) {
			profile_push("update kernels");
			update_kernels();
			profile_pop();
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
		profile_push("iteration");
		for(size_t n = 0 ; n < p.max_steps ; n++) {
			profile_push("step");
			// reset accumulator
			profile_push("(a) reset w");
				w = 0.0f;
			profile_pop();
			// transform bar_x for convolutions
			profile_push("(b) prepare bar_x");
				const auto f_bar_x = convolution->prepare_image(bar_x);
			profile_pop();
			profile_push("constraints");
			for(size_t i = 0 ; i < constraints.size() ; i++) {
				profile_push("kernel");
				auto &c = constraints[i];
				// convolve bar_x with kernel
				profile_push("(c) k * bar_x");
					convolution->conv(f_bar_x, c.k, convolved);
				profile_pop();
				debug(convolved, str(boost::format("convolved_%d") % i));
				// calculate new y_i
				profile_push("(d) soft_shrink");
					c.y = soft_shrink(c.y + convolved * sigma, c.q * sigma * input_stddev);
				profile_pop();
				debug(c.y, str(boost::format("y_%d") % i));
				// convolve y_i with conjugate transpose of kernel
				profile_push("(e) prepare y");
					const auto f_y = convolution->prepare_image(c.y);
				profile_pop();
				profile_push("(f) adj_k * y");
					convolution->conv(f_y, c.adj_k, convolved);
				profile_pop();
				debug(convolved, str(boost::format("adj_convolved_%d") % i));
				// accumulate
				profile_push("(g) accumulate w");
					w += convolved;
				profile_pop();
				profile_pop(/*kernel*/);
				debug(w, str(boost::format("w_%d") % i));
			}
			profile_pop();
			if(n % 10 == 0) this->progress(double(n) / p.max_steps, str(boost::format("Chambolle-Pock step %d") % n));

			profile_push("(h) resolvent");
				old_x = x;
				bar_x = x - Y - w*tau;
				debug(bar_x, "resolv_in");
				resolv->evaluate(tau, bar_x, x);
				x += Y;
			profile_pop();
			debug(x, "resolv_out");
			const T theta = 1 / sqrt(1 + 2 * tau * resolv->gamma);
			tau *= theta;
			sigma /= theta;
			profile_push("(i) bar_x");
				bar_x = x + (x - old_x) * theta;
				debug(bar_x, "bar_x");

				out = Y - x;
			profile_pop();
			profile_pop(/*step*/);

			if(!current(out, n)) break;

			if(n > 1 && p.tolerance > 0) {
				const T ch = norm_1(x) / norm_1(x - old_x);
				if(ch >= p.tolerance) break;
			}
		}
		profile_pop();
		profile_pop(/*run*/);
	}

	#undef debug
};
#endif
