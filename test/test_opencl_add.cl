kernel void multiply_add(
	global float *input_a,
	const float mul_a,
	global float *input_b,
	const float mul_b,
	global float *output
) {
	uint tid = get_global_id(0);
	output[tid] = input_a[tid] * mul_a + input_b[tid] * mul_b;
}
