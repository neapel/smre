#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include <vexcl/vexcl.hpp>
#include <vexcl/sat.hpp>

#include "multi_array_fft.h"
#include "multi_array.h"

struct prepared_image {
	virtual ~prepared_image() {}
};
struct prepared_kernel {
	virtual ~prepared_kernel() {}
};

template<class A>
struct convolver {
	virtual std::shared_ptr<prepared_image> prepare_image(const A &) = 0;
	virtual std::shared_ptr<prepared_kernel> prepare_kernel(size_t, bool) = 0;
	virtual void conv(std::shared_ptr<prepared_image>, std::shared_ptr<prepared_kernel>, A &) = 0;
	virtual ~convolver() {}
};

template<class T>
struct cpu_convolver : convolver<boost::multi_array<T, 2>> {
	typedef boost::multi_array<T, 2> A;

	virtual std::shared_ptr<prepared_image> _prepare_image(const A &k) {
		return this->prepare_image(k);
	};
	virtual void _conv(std::shared_ptr<prepared_image> i, std::shared_ptr<prepared_kernel> k, A &o) {
		this->conv(i,k,o);
	};
};

template<class T>
struct gpu_convolver : convolver<vex::vector<T>> {
	// implement CPU interface, too; copies data automatically, easier testing.

	typedef boost::multi_array<T, 2> cpu_A;
	typedef vex::vector<T> gpu_A;

	virtual std::shared_ptr<prepared_image> _prepare_image(const cpu_A &h_in) {
		gpu_A d_in(h_in.num_elements(), h_in.data());
		return this->prepare_image(d_in);
	}

	virtual void _conv(std::shared_ptr<prepared_image> i, std::shared_ptr<prepared_kernel> k, cpu_A &h_out) {
		gpu_A d_out(h_out.num_elements());
		this->conv(i, k, d_out);
		vex::copy(d_out, h_out.data());
	}
};



template<class T>
struct gpu_fft_convolver : gpu_convolver<T> {
	typedef typename vex::cl_vector_of<T, 2>::type T2;
	typedef vex::vector<T> A;
	typedef vex::vector<T2> A2;

	VEX_FUNCTION(complex_mul, T2(T2, T2),
		"return (float2)("
			"prm1.x * prm2.x - prm1.y * prm2.y,"
			"prm1.x * prm2.y + prm1.y * prm2.x);");

	struct prep : prepared_image, prepared_kernel {
		A2 f;
		prep(size_t n) : f(n) {};
	};

	vex::FFT<T, T2> fft;
	vex::FFT<T2, T> ifft;
	const size_t n;
	const size2_t s;
	A2 temp;

	gpu_fft_convolver(size2_t s)
	: fft({s[0], s[1]}), ifft({s[0], s[1]}, vex::fft::inverse), n(s[0] * s[1]), s(s),
	  temp(n) {}

	virtual std::shared_ptr<prepared_image> prepare_image(const A &in) {
		auto i = std::make_shared<prep>(n);
		i->f = fft(in);
		return i;
	}

	virtual std::shared_ptr<prepared_kernel> prepare_kernel(size_t h, bool adj) {
		const T v = 1 / (M_SQRT2 * h);
		boost::multi_array<T, 2> k(s);
		mimas::fill(k, 0);
		for(size_t i0 = 0 ; i0 < h ; i0++)
			for(size_t i1 = 0 ; i1 < h ; i1++) {
				if(adj) k[i0][i1] = v;
				else k[(s[0] - i0) % s[0]][(s[1] - i1) % s[1]] = v;			
			}
		A d_k(n, k.data());
		auto p = std::make_shared<prep>(n);
		p->f = fft(d_k);
		return p;
	}

	virtual void conv(std::shared_ptr<prepared_image> i, std::shared_ptr<prepared_kernel> k, A &out) {
		auto fi = std::dynamic_pointer_cast<prep>(i)->f;
		auto fk = std::dynamic_pointer_cast<prep>(k)->f;
		temp = complex_mul(fi, fk);
		out = ifft(temp);
	}
};



template<class T>
struct cpu_fft_convolver : cpu_convolver<T> {
	typedef std::complex<T> T2;
	typedef boost::multi_array<T, 2> A;
	typedef boost::multi_array<T2, 2> A2;

	struct prep : prepared_image, prepared_kernel {
		A2 f;
		prep(size2_t s) : f(s) {}
	};

	fftw::plan<T, T2, 2> fft;
	fftw::plan<T2, T, 2> ifft;
	const size2_t s, f_s;

	cpu_fft_convolver(size2_t s)
	: fft(s), ifft(s), s(s), f_s{{s[0], s[1]/2+1}} {}

	virtual std::shared_ptr<prepared_image> prepare_image(const A &in) {
		auto i = std::make_shared<prep>(f_s);
		fft(in, i->f);
		return i;
	}

	virtual std::shared_ptr<prepared_kernel> prepare_kernel(size_t h, bool adj) {
		const T v = 1 / (s[0] * s[1] * M_SQRT2 * h);
		boost::multi_array<T, 2> k(s);
		mimas::fill(k, 0);
		for(size_t i0 = 0 ; i0 < h ; i0++)
			for(size_t i1 = 0 ; i1 < h ; i1++) {
				if(adj) k[i0][i1] = v;
				else k[(s[0] - i0) % s[0]][(s[1] - i1) % s[1]] = v;			
			}
		auto p = std::make_shared<prep>(f_s);
		fft(k, p->f);
		return p;
	}

	virtual void conv(std::shared_ptr<prepared_image> i, std::shared_ptr<prepared_kernel> k, A &out) {
		auto fi = std::dynamic_pointer_cast<prep>(i)->f;
		auto fk = std::dynamic_pointer_cast<prep>(k)->f;
		A2 temp(f_s);
		for(size_t i0 = 0 ; i0 < f_s[0] ; i0++)
			for(size_t i1 = 0 ; i1 < f_s[1] ; i1++)
				temp[i0][i1] = fi[i0][i1] * fk[i0][i1];
		ifft(temp, out);
	}
};



template<class T>
struct gpu_sat_convolver : gpu_convolver<T> {
	typedef vex::vector<T> A;

	struct prep_i : prepared_image {
		A f;
		prep_i(size_t s) : f(s) {}
	};

	struct prep_k : prepared_kernel {
		size_t h;
		bool adj;
		prep_k(size_t h, bool adj) : h(h), adj(adj) {}
	};

	const size2_t s;
	const std::vector<cl::CommandQueue> &queues;
	vex::SAT<T> calc_sat;
	cl::Kernel box_sum;

	gpu_sat_convolver(size2_t s)
	: s(s), queues(vex::current_context().queue()), calc_sat(queues, s[0], s[1]) {
		std::ostringstream o;
		o << vex::standard_kernel_header(vex::qdev(queues[0]))
		  << "typedef " << vex::type_name<T>() << " T;\n"
		  << box_sum_code;
		auto program = vex::build_sources(vex::qctx(queues[0]), o.str());
		box_sum = cl::Kernel(program, "box_sum");
	}

	virtual std::shared_ptr<prepared_image> prepare_image(const A &in) {
		auto i = std::make_shared<prep_i>(s[0] * s[1]);
		i->f = calc_sat(in);
		return i;
	}

	virtual std::shared_ptr<prepared_kernel> prepare_kernel(size_t h, bool adj) {
		return std::make_shared<prep_k>(h, adj);
	}

	virtual void conv(std::shared_ptr<prepared_image> i, std::shared_ptr<prepared_kernel> k_, A &out) {
		const auto sat = std::dynamic_pointer_cast<prep_i>(i)->f;
		const auto k = std::dynamic_pointer_cast<prep_k>(k_);
		const T v = 1 / (M_SQRT2 * k->h);
		box_sum.setArg(0, sat(0));
		box_sum.setArg(1, out(0));
		box_sum.setArg(2, v);
		box_sum.setArg(3, cl_uint2{{(cl_uint)s[0], (cl_uint)s[1]}});
		if(k->adj) {
			box_sum.setArg(4, cl_int2{{-(cl_int)k->h, -(cl_int)k->h}});
			box_sum.setArg(5, cl_int2{{0, 0}});
		} else {
			box_sum.setArg(4, cl_int2{{-1, -1}});
			box_sum.setArg(5, cl_int2{{(cl_int)k->h - 1, (cl_int)k->h - 1}});
		}

		auto dev = vex::qdev(queues[0]);
		size_t w_size = vex::kernel_workgroup_size(box_sum, dev);
		size_t n_wg = vex::num_workgroups(dev);
		size_t g_size = n_wg * w_size;

		queues[0].enqueueNDRangeKernel(box_sum, cl::NullRange,
			g_size, w_size);
	}

	const std::string box_sum_code = R"(
		kernel void box_sum(global const T *sat, global T *out, T scale, uint2 s, int2 di, int2 dj) {
			const size_t off0 = get_global_id(0);
			for(size_t k = off0 ; k < s.s0 * s.s1 ; k += get_global_size(0)) {
				const size_t
					k1 = k % s.s1,
					k0 = k / s.s1,
					i0 = (k0 + s.s0 + di.s0) % s.s0,
					i1 = (k1 + s.s1 + di.s1) % s.s1,
					j0 = (k0 + s.s0 + dj.s0) % s.s0,
					j1 = (k1 + s.s1 + dj.s1) % s.s1;
				// corners
				T sum = sat[s.s1 * i0 + i1] - sat[s.s1 * i0 + j1] - sat[s.s1 * j0 + i1] + sat[s.s1 * j0 + j1];
				// bottom edge
				if(i0 > j0) sum += sat[s.s1 * (s.s0 - 1) + j1] - sat[s.s1 * (s.s0 - 1) + i1];
				if(i1 > j1) {
					// right edge
					sum += sat[s.s1 * j0 + (s.s1 - 1)] - sat[s.s1 * i0 + (s.s1 - 1)];
					// bottom right corner
					if(i0 > j0) sum += sat[s.s1 * (s.s0 - 1) + (s.s1 - 1)];
				}
				out[s.s1 * k0 + k1] = scale * sum;
			}
		}
	)";
};



template<class T>
struct cpu_sat_convolver : cpu_convolver<T> {
	typedef boost::multi_array<T, 2> A;

	struct prep_i : prepared_image {
		A f;
		prep_i(size2_t s) : f(s) {}
	};

	struct prep_k : prepared_kernel {
		size_t h;
		bool adj;
		prep_k(size_t h, bool adj) : h(h), adj(adj) {}
	};

	const size2_t s;

	cpu_sat_convolver(size2_t s)
	: s(s) {}

	virtual std::shared_ptr<prepared_image> prepare_image(const A &in) {
		auto i = std::make_shared<prep_i>(s);
		for(size_t i0 = 0 ; i0 < s[0] ; i0++)
			for(size_t i1 = 0 ; i1 < s[1] ; i1++)
				i->f[i0][i1] = in[i0][i1]
					+ (i0 > 0 ? i->f[i0 - 1][i1] : 0)
					+ (i1 > 0 ? i->f[i0][i1 - 1] : 0)
					- (i0 > 0 && i1 > 0 ? i->f[i0 - 1][i1 - 1] : 0);
		return i;
	}

	virtual std::shared_ptr<prepared_kernel> prepare_kernel(size_t h, bool adj) {
		return std::make_shared<prep_k>(h, adj);
	}

	virtual void conv(std::shared_ptr<prepared_image> i, std::shared_ptr<prepared_kernel> k_, A &out) {
		const auto sat = std::dynamic_pointer_cast<prep_i>(i)->f;
		const auto k = std::dynamic_pointer_cast<prep_k>(k_);
		const T v = 1 / (M_SQRT2 * k->h);
		if(k->adj) {
			for(size_t i0 = 0 ; i0 < s[0] ; i0++)
				for(size_t i1 = 0 ; i1 < s[1] ; i1++)
					out[i0][i1] = v * box_sum(sat,
						(i0 + s[0] - k->h) % s[0], (i1 + s[1] - k->h) % s[1], i0, i1);
		} else {
			for(size_t i0 = 0 ; i0 < s[0] ; i0++)
				for(size_t i1 = 0 ; i1 < s[1] ; i1++)
					out[i0][i1] = v * box_sum(sat,
						(i0 + s[0] - 1) % s[0], (i1 + s[1] - 1) % s[1],
						(i0 + k->h - 1) % s[0], (i1 + k->h - 1) % s[1]);
		}
	}

	// sum of i0..j0 i1..j1 inclusive, circular.
	private:
	inline T box_sum(const A &sat, size_t i0, size_t i1, size_t j0, size_t j1) const {
		// corners
		T sum = sat[i0][i1] - sat[i0][j1] - sat[j0][i1] + sat[j0][j1];
		// projections to bottom
		if(i0 > j0)	sum += sat[s[0]-1][j1] - sat[s[0]-1][i1];
		if(i1 > j1) {
			// projections to right
			sum += sat[j0][s[1]-1] - sat[i0][s[1]-1];
			// lower right corner
			if(i0 > j0) sum += sat[s[0]-1][s[1]-1];
		}
		return sum;
	};
};




#endif
