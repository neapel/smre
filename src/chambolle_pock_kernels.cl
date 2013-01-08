// real image -> complex image
__kernel void real2complex(__global float *input, __global float2 *output) {
	const size_t i = get_global_id(0);
	output[i].x = input[i];
	output[i].y = 0;
}

// real kernel -> padded complex kernel
// place input in (0,0) corner of output.
__kernel void pad(__global float *input, __global float2 *output, uint ow) {
	const size_t ky = get_global_id(0), kx = get_global_id(1);
	const size_t kh = get_global_size(0), kw = get_global_size(1);
	output[ow * ky + kx].x = input[kw * ky + kx];
}

// real kernel -> (conjugate) transposed padded complex kernel
// place transposed input in (h+2, w+2) corner of output.
__kernel void conjugate_transpose_pad(__global float *input, __global float2 *output, uint ow, uint oh) {
	const size_t ky = get_global_id(0), kx = get_global_id(1);
	const size_t kh = get_global_size(0), kw = get_global_size(1);
	const size_t dx = ow - kw + 1, dy = oh - kh + 1;
	output[ow * ((kx + dx) % ow) + ((ky + dy) % oh)].x = input[kw * kx + ky];
}



#include "random123/threefry.h"
typedef threefry2x32_key_t rand_k;
typedef threefry2x32_ctr_t rand_c;
#define rand(c,k) threefry2x32(c,k)

// returns a random float2 [0..1]
float2 rand_f(rand_c *c, rand_k k) {
	c->v[0]++;
	union {rand_c rc; uint2 ri;} res;
	res.rc = rand(*c, k);
	float2 out;
	out.x = u01_closed_closed_32_24(res.ri.x);
	out.y = u01_closed_closed_32_24(res.ri.y);
	return out;
}

// Marsaglia polar method:
// from uniform random numbers [0..1]
// generate normal distributed mean=0, sigma=1
float2 normal_distributed(rand_c *c, rand_k k) {
	float r2;
	float2 v;
	do {
		v = 2 * rand_f(c, k) - (float2)(1, 1);
		r2 = dot(v, v);
	} while(r2 > 1 || r2 == 0);
	v *= sqrt(-2 * log(r2) / r2);
	return v;
}

__kernel void random_fill(__global float2 *out, uint len, uint seed, float sigma) {
	const size_t i = get_global_id(0);
	const rand_k k = {{i, seed}};
	rand_c c = {{0, 0x12345678}};
	float2 r = normal_distributed(&c, k) * sigma;
	out[i].x = r.x;
	out[i].y = 0;
}

__kernel void reduce_max_abs(__global float2 *in, uint len, __global float *out) {
	if(get_global_id(0) == 0) {
		float m = -1;
		for(size_t i = 0 ; i < len ; i++)
			m = fmax(m, fabs(in[i].x));
		out[0] = m;
	}
}

// find max(norm(x)) for x in data, write to output[0].
__kernel void reduce_max_norm(__global float2 *data, __global float *output) {
	if(get_global_id(0) == 0) {
		float m = 0;
		for(size_t i = 0 ; i < get_global_size(0) ; i++) {
			float d = dot(data[i], data[i]);
			m = max(m, d);
		}
		output[0] = m;
	}
}

// a * b
float2 _complex_mul(float2 a, float2 b) {
	float2 out;
	out.x = a.x * b.x - a.y * b.y;
	out.y = a.x * b.y + a.y * b.x;
	return out;
}

// Complex: out = a * b
__kernel void complex_mul(__global float2 *out, __global float2 *a, __global float2 *b) {
	const size_t i = get_global_id(0);
	out[i] = _complex_mul(a[i], b[i]);
}

// Real: out = a + b.
__kernel void add2v(__global float *out, __global float *a, __global float2 *b) {
	const size_t i = get_global_id(0);
	out[i] = a[i] + b[i].x;
}

// Real: out = a * as + b * bs + c * cs
__kernel void mul_add3vs(__global float *out, __global float *a, float as, __global float *b, float bs, __global float *c, float cs) {
	const size_t i = get_global_id(0);
	out[i] = a[i] * as + b[i] * bs + c[i] * cs;
}

// Real: out = a * as + b * bs
__kernel void mul_add2vs(__global float2 *out, __global float *a, float as, __global float *b, float bs) {
	const size_t i = get_global_id(0);
	out[i].x = a[i] * as + b[i] * bs;
	out[i].y = 0;
}

// Real: out = a - b
__kernel void sub(__global float *out, __global float *a, __global float *b) {
	const size_t i = get_global_id(0);
	out[i] = a[i] - b[i];
}

float soft_clamp(float x, float lo, float hi) {
	if(x < lo) return x - lo;
	if(x > hi) return x - hi;
	return 0;
}

// Real: out = [a + b * s]
__kernel void mul_add_clamp(__global float2 *out, __global float2 *a, __global float2 *b, float s, float lo, float hi) {
	const size_t i = get_global_id(0);
	out[i].x = soft_clamp(a[i].x + b[i].x * s, lo, hi);
	out[i].y = 0;
}
