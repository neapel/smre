#include "opencl_helpers.h"

using namespace cl;
using namespace std;



string read_file(string name) {
	using namespace std;
	fstream f(name.c_str(), fstream::in | fstream::binary);
	return string(
		istreambuf_iterator<char>(f),
		istreambuf_iterator<char>()
	);
}



vector<device> cl::platform::all_devices(cl_device_type type) const {
	cl_uint size;
	cl_int status = clGetDeviceIDs(native_platform, type, 0, NULL, &size);
	if(status != CL_SUCCESS || size == 0)
		throw error(status, "Couldn't enumerate devices");
	cl_device_id devices[size];
	status = clGetDeviceIDs(native_platform, type, size, devices, &size);
	if(status != CL_SUCCESS || size == 0)
		throw error(status, "Couldn't enumerate devices");
	vector<device> out;
	for(cl_uint i = 0 ; i < size ; i++)
		out.push_back( device(devices[i]) );
	return out;
}



ostream &cl::operator<<(ostream &o, const device &d) {
	o << "  Device (" << d.info<string>(CL_DEVICE_NAME) << ")\n";
	o << "    Type: ";
	switch(d.info<cl_device_type>(CL_DEVICE_TYPE)) {
		case CL_DEVICE_TYPE_CPU: o << "CPU"; break;
		case CL_DEVICE_TYPE_GPU: o << "GPU"; break;
		case CL_DEVICE_TYPE_ACCELERATOR: o << "Accelerator"; break;
		case CL_DEVICE_TYPE_DEFAULT: o << "Default"; break;
		default: o << "Other"; break;
	}
	o << '\n';
	o
		<< "    Available: " << d.info<cl_bool>(CL_DEVICE_AVAILABLE) << '\n'
		<< "    Compiler available: " << d.info<cl_bool>(CL_DEVICE_COMPILER_AVAILABLE) << '\n'
//		<< "    Linker available: " << d.info<cl_bool>(CL_DEVICE_LINKER_AVAILABLE) << '\n'
		<< "    Vendor: " << d.info<string>(CL_DEVICE_VENDOR) << '\n'
		<< "    Version: " << d.info<string>(CL_DEVICE_VERSION) << '\n'
		<< "    Profile: " << d.info<string>(CL_DEVICE_PROFILE) << '\n'
		<< "    Language Version: " << d.info<string>(CL_DEVICE_OPENCL_C_VERSION) << '\n'
		<< "    Driver Version: " << d.info<string>(CL_DRIVER_VERSION) << '\n'
		<< "    Memory size: " << d.info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE) / (1024 * 1024) << 'M' << '\n'
		<< "    Compute units: " << d.info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS) << '\n'
		<< "    Bits: " << d.info<cl_uint>(CL_DEVICE_ADDRESS_BITS) << '\n'
		<< "    Cache size: " << d.info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) / 1024 << 'k' << '\n'
		<< "    Cacheline size: " << d.info<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << '\n'
//		<< "    Builtin Kernels: " << d.info<string>(CL_DEVICE_BUILT_IN_KERNELS) << '\n'
		<< "    Extensions: " << d.info<string>(CL_DEVICE_EXTENSIONS) << '\n'
	;
	return o;
}



string cl::platform::info(cl_platform_info name) const {
	size_t size;
	cl_int status = clGetPlatformInfo(native_platform, name, 0, NULL, &size);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't get platform info");
	char buf[size];
	status = clGetPlatformInfo(native_platform, name, size, buf, &size);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't get platform info");
	return string(buf, size);
}


ostream &cl::operator<<(ostream &o, const platform &p) {
	o
		<< "Platform (" << p.info(CL_PLATFORM_NAME) << ")\n"
		<< "  Vendor: " << p.info(CL_PLATFORM_VENDOR) << '\n'
		<< "  Version: " << p.info(CL_PLATFORM_VERSION) << '\n'
		<< "  Profile: " << p.info(CL_PLATFORM_PROFILE) << '\n'
		<< "  Extensions: " << p.info(CL_PLATFORM_EXTENSIONS) << '\n'
	;
	for(auto d : p.all_devices())
		o << d << '\n';
	return o;
}



vector<platform> cl::all_platforms() {
	cl_uint size;
	cl_int status = clGetPlatformIDs(0, NULL, &size);
	if(status != CL_SUCCESS || size == 0)
		throw error(status, "Couldn't enumerate platforms");
	cl_platform_id platforms[size];
	status = clGetPlatformIDs(size, platforms, &size);
	if(status != CL_SUCCESS || size == 0)
		throw error(status, "Couldn't enumerate platforms");
	vector<platform> out;
	for(cl_uint i = 0 ; i < size ; i++)
		out.push_back( platform(platforms[i]) );
	return out;
}




void cl::context::construct(platform p) {
	cl_int status;
	// create the context
	cl_context_properties context_props[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(p.native_platform),
		0
	};
	ctx = clCreateContextFromType(
		context_props,
		CL_DEVICE_TYPE_CPU,
		[](const char *error, const void *, size_t, void *) {
			cerr << "OpenCL Error: " << error << endl;
		},
		this,
		&status
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create CL context");

	// get the devices
	size_t num_devices;
	status = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &num_devices);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't query CL context");
	cl_device_id devices[num_devices];
	status = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, num_devices, devices, &num_devices);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't query CL context");
	device = devices[0];
	clog << "Using" << ::device(device) << endl;

	// create queue
	queue = clCreateCommandQueue(ctx, device, 0, &status);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create CL queue");
}



program cl::context::load_program(string filename) {
	return program(this, filename);
}



void cl::context::wait(vector<event> events) {
	if(events.size() == 0) return;
	cl_event list[events.size()];
	for(size_t i = 0 ; i < events.size() ; i++)
		list[i] = events[i].native_event;
	cl_uint status = clWaitForEvents(events.size(), list);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't wait for events");
}



event cl::kernel::run(vector<size_t> size, vector<size_t> offset , vector<size_t> group_size, vector<event> wait) {
	size_t *global_work_size = new size_t[size.size()];
	copy(size.begin(), size.end(), global_work_size);
	size_t *global_work_offset = nullptr;
	if(offset.size() == size.size()) {
		global_work_offset = new size_t[size.size()];
		copy(offset.begin(), offset.end(), global_work_offset);
	}
	size_t *local_work_size = nullptr;
	if(group_size.size() == size.size()) {
		local_work_size = new size_t[size.size()];
		copy(group_size.begin(), group_size.end(), local_work_size);
	}
	cl_event wait_list[wait.size()];
	for(size_t i = 0 ; i < wait.size() ; i++)
		wait_list[i] = wait[i].native_event;
	cl_event e;
	cl_uint status = clEnqueueNDRangeKernel(
		parent_context->queue,
		k,
		size.size(),
		global_work_offset,
		global_work_size,
		local_work_size,
		wait.size(),
		wait.size() == 0 ? nullptr : wait_list,
		&e
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't run kernel");
	return event(e);
}



cl::program::program(context *parent, string filename) : parent(parent) {
	// load code
	string data = read_file(filename);
	const char *text = data.c_str();
	size_t length = data.size();
	cl_int status;
	native_program = clCreateProgramWithSource(
		parent->ctx, 1,
		&text,
		&length,
		&status
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't load CL program from source");

	// compile code
	cl_device_id devs[] = {parent->device};
	status = clBuildProgram(
		native_program,
		1,
		devs,
		nullptr,
		//"-cl-no-signed-zeros -cl-finite-math-only -Werror"
		//"-cl-std=CL1.1 -cl-kernel-arg-info",
		nullptr,
		nullptr
	);
	if(status != CL_SUCCESS && status != CL_BUILD_PROGRAM_FAILURE)
		throw error(status, "Couldn't build CL program from source");
	cl_int old_status = status;
	
	char build_log[2048];
	size_t log_length;
	status = clGetProgramBuildInfo(
		native_program,
		parent->device,
		CL_PROGRAM_BUILD_LOG,
		sizeof(build_log),
		build_log,
		&log_length
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't get build log");
	if(log_length > 0)
		cerr << "Build log for " << filename << ":" << endl << build_log << endl;
	if(old_status != CL_SUCCESS)
		throw error(old_status, "Couldn't build CL program from source");
}



kernel cl::program::operator[](string name) {
	cl_int status;
	cl_kernel k = clCreateKernel(native_program, name.c_str(), &status);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create kernel from program");
	return kernel(parent, k);
}



string cl::status_to_string(cl_int status) {
	#define e(m) case m: return #m;
	switch(status) {
		e(CL_SUCCESS                                  )
		e(CL_DEVICE_NOT_FOUND                         )
		e(CL_DEVICE_NOT_AVAILABLE                     )
		e(CL_COMPILER_NOT_AVAILABLE                   )
		e(CL_MEM_OBJECT_ALLOCATION_FAILURE            )
		e(CL_OUT_OF_RESOURCES                         )
		e(CL_OUT_OF_HOST_MEMORY                       )
		e(CL_PROFILING_INFO_NOT_AVAILABLE             )
		e(CL_MEM_COPY_OVERLAP                         )
		e(CL_IMAGE_FORMAT_MISMATCH                    )
		e(CL_IMAGE_FORMAT_NOT_SUPPORTED               )
		e(CL_BUILD_PROGRAM_FAILURE                    )
		e(CL_MAP_FAILURE                              )
		e(CL_MISALIGNED_SUB_BUFFER_OFFSET             )
		e(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
		e(CL_INVALID_VALUE                            )
		e(CL_INVALID_DEVICE_TYPE                      )
		e(CL_INVALID_PLATFORM                         )
		e(CL_INVALID_DEVICE                           )
		e(CL_INVALID_CONTEXT                          )
		e(CL_INVALID_QUEUE_PROPERTIES                 )
		e(CL_INVALID_COMMAND_QUEUE                    )
		e(CL_INVALID_HOST_PTR                         )
		e(CL_INVALID_MEM_OBJECT                       )
		e(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          )
		e(CL_INVALID_IMAGE_SIZE                       )
		e(CL_INVALID_SAMPLER                          )
		e(CL_INVALID_BINARY                           )
		e(CL_INVALID_BUILD_OPTIONS                    )
		e(CL_INVALID_PROGRAM                          )
		e(CL_INVALID_PROGRAM_EXECUTABLE               )
		e(CL_INVALID_KERNEL_NAME                      )
		e(CL_INVALID_KERNEL_DEFINITION                )
		e(CL_INVALID_KERNEL                           )
		e(CL_INVALID_ARG_INDEX                        )
		e(CL_INVALID_ARG_VALUE                        )
		e(CL_INVALID_ARG_SIZE                         )
		e(CL_INVALID_KERNEL_ARGS                      )
		e(CL_INVALID_WORK_DIMENSION                   )
		e(CL_INVALID_WORK_GROUP_SIZE                  )
		e(CL_INVALID_WORK_ITEM_SIZE                   )
		e(CL_INVALID_GLOBAL_OFFSET                    )
		e(CL_INVALID_EVENT_WAIT_LIST                  )
		e(CL_INVALID_EVENT                            )
		e(CL_INVALID_OPERATION                        )
		e(CL_INVALID_GL_OBJECT                        )
		e(CL_INVALID_BUFFER_SIZE                      )
		e(CL_INVALID_MIP_LEVEL                        )
		e(CL_INVALID_GLOBAL_WORK_SIZE                 )
		e(CL_INVALID_PROPERTY                         )
		default: return "";
	}
	#undef e
}
