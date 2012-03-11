#pragma OPENCL EXTENSION cl_amd_printf : enable

kernel void add(
	global float *input_a,
	global float *input_b,
	global float *output
) {
	if(false) {
		for(uint i = 0 ; i < get_work_dim() ; i++) {
			printf("dim %d: global id=%d size=%d offset=%d local id=%d size=%d group=%d/%d\n",
				i,
				get_global_id(i), get_global_size(0), get_global_offset(0),
				get_local_id(0), get_local_size(0),
				get_group_id(0), get_num_groups(0));
		}
	}
	const uint x = get_global_id(0);
	const uint y = get_global_id(1);
	const uint i = x + y * get_global_size(0);
	output[i] = input_a[i] + input_b[i];
}

