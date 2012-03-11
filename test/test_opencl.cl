#pragma OPENCL EXTENSION cl_amd_printf : enable

kernel void multiply_add(
	global float *input_a,
	const float mul_a,
	global float *input_b,
	const float mul_b,
	global float *output
) {
	uint tid = get_global_id(0);
	if(false) {
		for(uint i = 0 ; i < get_work_dim() ; i++) {
			printf("dim %d: global id=%d size=%d offset=%d local id=%d size=%d group=%d/%d\n",
				i,
				get_global_id(i), get_global_size(0), get_global_offset(0),
				get_local_id(0), get_local_size(0),
				get_group_id(0), get_num_groups(0));
		}
	}
	output[tid] = input_a[tid] * mul_a + input_b[tid] * mul_b;
}
