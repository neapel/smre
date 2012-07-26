#include "opencl_helpers.h"

#ifdef HAVE_OPENCL
#include <sstream>

using namespace cl;
using namespace std;




cl_uint make_wait_list(vector<event> events, cl_event *&out) {
	if(events.size() == 0) {
		out = nullptr;
		return 0;
	} else {
		out = new cl_event[events.size()];
		size_t i = 0;
		for(event e : events)
			out[i++] = e.native;
		return events.size();
	}
}



vector<device> cl::platform::all_devices(cl_device_type type) const {
	cl_uint size;
	cl_int status = clGetDeviceIDs(native, type, 0, NULL, &size);
	if(status != CL_SUCCESS || size == 0)
		throw error(status, "Couldn't enumerate devices");
	cl_device_id devices[size];
	status = clGetDeviceIDs(native, type, size, devices, &size);
	if(status != CL_SUCCESS || size == 0)
		throw error(status, "Couldn't enumerate devices");
	vector<device> out;
	for(cl_uint i = 0 ; i < size ; i++)
		out.push_back( device(devices[i]) );
	return out;
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



event::event(const context *c, cl_event e) : parent_context(c), native(e) {
	if(false) {
		clSetEventCallback(e, CL_SUBMITTED,
			[](cl_event e, cl_int, void *) { clog << "Event " << e << " SUBMITTED" << endl; },
			nullptr);
		clSetEventCallback(e, CL_RUNNING,
			[](cl_event e, cl_int, void *) { clog << "Event " << e << " RUNNING" << endl; },
			nullptr);
		clSetEventCallback(e, CL_COMPLETE,
			[](cl_event e, cl_int, void *) { clog << "Event " << e << " COMPLETE" << endl; },
			nullptr);
	}
}


after event::then() {
	return move( after{*this} );
}


void construct(buffer &buf, const context *ctx) {
	if(buf.parent_context != nullptr && buf.parent_context != ctx)
		throw logic_error("Buffer already registered with another context");
	buf.parent_context = ctx;
	if(buf.native != 0)
		return;
	cl_mem_flags f = 0;
	if((buf.flags & kernel_read) && (buf.flags & kernel_write)) f |= CL_MEM_READ_WRITE;
	else if(buf.flags & kernel_read) f |= CL_MEM_READ_ONLY;
	else if(buf.flags & kernel_write) f |= CL_MEM_WRITE_ONLY;
	else throw invalid_argument("Buffer must be accessible from kernel");
	if((buf.flags & host_read) || (buf.flags & host_write)) {
		f |= CL_MEM_ALLOC_HOST_PTR;
#if 0 // OpenCL1.2
		if(!(buf.flags & host_read)) f |= CL_MEM_COPY_HOST_WRITE_ONLY;
		else if(!(buf.flag & host_write)) f |= CL_MEM_COPY_HOST_READ_ONLY;
#endif
	}
#if 0 // OpenCL1.2
	else f |= CL_MEM_COPY_HOST_NO_ACCESS;
#endif
	cl_int status;
	buf.native = clCreateBuffer(ctx->native, f, buf.size, NULL, &status);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create CL buffer");
}


event after::operator()(buffer_write &&bw) {
	if(bw.buf.size != 0 && bw.buf.size != bw.size)
		throw std::range_error("Data size different from buffer size.");
	bw.buf.size = bw.size;
	construct(bw.buf, parent_context);
	cl_event *wait_list;
	cl_uint wait_size = make_wait_list(events, wait_list);
	cl_event e;
	cl_uint status = clEnqueueWriteBuffer(
		parent_context->queue,
		bw.buf.native,
		CL_FALSE, // blocking
		0, // offset
		bw.size,
		bw.data,
		wait_size,
		wait_list,
		&e
	);
	delete [] wait_list;
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't queue writing to CL buffer");
	return event(parent_context, e);
}



event after::operator()(buffer_read &&bw) {
	construct(bw.buf, parent_context);
	if(bw.buf.size != bw.size)
		throw std::range_error("Sizes different.");
	cl_event *wait_list;
	cl_uint wait_size = make_wait_list(events, wait_list);
	cl_event e;
	cl_uint status = clEnqueueReadBuffer(
		parent_context->queue,
		bw.buf.native,
		CL_FALSE, // blocking
		0, // offset
		bw.size,
		bw.data,
		wait_size,
		wait_list,
		&e
	);
	delete [] wait_list;
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't queue reading from CL buffer");
	return event(parent_context, e);
}



void construct(context &c, platform p) {
	cl_int status;
	// create the context
	cl_context_properties context_props[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(p.native),
		0
	};
	c.native = clCreateContextFromType(
		context_props,
		CL_DEVICE_TYPE_CPU,
		[](const char *error, const void *, size_t, void *) {
			cerr << "OpenCL Error: " << error << endl;
		},
		nullptr,
		&status
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create CL context");

	// get the devices
	size_t num_devices;
	status = clGetContextInfo(c.native, CL_CONTEXT_DEVICES, 0, NULL, &num_devices);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't query CL context");
	cl_device_id devices[num_devices];
	status = clGetContextInfo(c.native, CL_CONTEXT_DEVICES, num_devices, devices, &num_devices);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't query CL context");
	c.device = devices[0];

	// create queue
	c.queue = clCreateCommandQueue(c.native, c.device, 0, &status);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create CL queue");
}


context::context() {
	construct(*this, all_platforms().front());
}


context::context(platform p) {
	construct(*this, p);
}


context::~context() {
#if HAVE_AMD_FFT
	//clAmdFftTeardown();
#endif
}




string read_file(string name) {
	using namespace std;
	fstream f(name.c_str(), fstream::in | fstream::binary);
	return string(
		istreambuf_iterator<char>(f),
		istreambuf_iterator<char>()
	);
}

program cl::context::load(string filename) {
	return program(this, read_file(filename));
}

program cl::context::compile(string code) {
	return program(this, code);
}






void after::resume() {
	if(events.size() == 0) return;
	cl_event *wait_list;
	cl_uint wait_size = make_wait_list(events, wait_list);
	cl_uint status = clWaitForEvents(wait_size, wait_list);
	delete [] wait_list;
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't wait for events");
}




event after::operator()(const kernel &r) {
	if(r.global_size.size() == 0)
		throw invalid_argument("No work size set for kernel");
	const size_t dims = r.global_size.size();
	size_t *global_work_size = new size_t[dims];
	copy(r.global_size.begin(), r.global_size.end(), global_work_size);
	size_t *global_work_offset = nullptr;
	if(r.global_offset.size() == dims) {
		global_work_offset = new size_t[dims];
		copy(r.global_offset.begin(), r.global_offset.end(), global_work_offset);
	}
	cl_event *wait_list;
	cl_uint wait_size = make_wait_list(events, wait_list);
	cl_event e;
	cl_uint status = clEnqueueNDRangeKernel(
		parent_context->queue,
		r.native,
		dims,
		global_work_offset,
		global_work_size,
		nullptr,
		wait_size,
		wait_list,
		&e
	);
	delete [] wait_list;
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't run kernel");
	return event(parent_context, e);
}



cl::program::program(context *parent, string data) : parent(parent) {
	// load code
	const char *text = data.c_str();
	size_t length = data.size();
	cl_int status;
	native = clCreateProgramWithSource(
		parent->native, 1,
		&text,
		&length,
		&status
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't load CL program from source");

	// compile code
	cl_device_id devs[] = {parent->device};
	status = clBuildProgram(
		native,
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
		native,
		parent->device,
		CL_PROGRAM_BUILD_LOG,
		sizeof(build_log),
		build_log,
		&log_length
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't get build log");
	if(log_length > 3)
		cerr << "Build log:\n" << build_log << endl;
	if(old_status != CL_SUCCESS)
		throw error(old_status, "Couldn't build CL program from source");
}



kernel cl::program::operator[](string name) {
	cl_int status;
	cl_kernel k = clCreateKernel(native, name.c_str(), &status);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't create kernel from program");
	return kernel(parent, k);
}



kernel::kernel(context *parent_context, cl_kernel k) : native(k), parent_context(parent_context) {
	cl_uint value;
	cl_int status = clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(value), &value, nullptr);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't get argument count");
	argument_count = value;
}



void kernel::arg(cl_uint i, buffer &buf) {
	construct(buf, parent_context);
	cl_uint status = clSetKernelArg(
		native,
		i,
		sizeof(cl_mem),
		&buf.native
	);
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't set kernel argument to buffer");
}


#if HAVE_AMD_FFT
fft_run fft::forward(buffer &in, buffer &out) {
	return std::move( fft_run(*this, CLFFT_FORWARD, in, out) );
}

fft_run fft::backward(buffer &in, buffer &out) {
	return std::move( fft_run(*this, CLFFT_BACKWARD, in, out) );
}


void construct(fft &that, const context *ctx) {
	cl_uint status = clAmdFftCreateDefaultPlan(
		&that.native,
		ctx->native,
		that.dim,
		that.lengths
	);
	status |= clAmdFftSetPlanPrecision(that.native, CLFFT_SINGLE);
	status |= clAmdFftSetPlanScale(that.native, CLFFT_FORWARD, 1);
	status |= clAmdFftSetPlanScale(that.native, CLFFT_BACKWARD, 1);
	// stride x:1, y:len_x, z:len_x*len_y
	status |= clAmdFftSetLayout(that.native, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	status |= clAmdFftSetResultLocation(that.native, CLFFT_OUTOFPLACE);

	if(status != CL_SUCCESS) throw error(status, "Couldn't plan FFT.");
}


event after::operator()(const fft_run &&r) {
	construct(r.that, parent_context);
	construct(r.in, parent_context);
	construct(r.out, parent_context);

	cl_mem in[] = {r.in.native}, out[] = {r.out.native};
	cl_command_queue queue[] = {parent_context->queue};

	cl_event *wait_list;
	cl_uint wait_size = make_wait_list(events, wait_list);
	cl_event e;
	cl_uint status = clAmdFftEnqueueTransform(
		r.that.native, // handle
		r.dir, // direction
		1, // num queues and events
		queue, // queues
		wait_size, wait_list, &e,
		in, out,
		nullptr // temp
	);
	delete [] wait_list;
	if(status != CL_SUCCESS)
		throw error(status, "Couldn't run FFT.");
	return event(parent_context, e);
}

#endif



// Strings from constants
#define e(m) case m: return #m;

string cl::status_to_string(cl_int status) {
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
#if HAVE_AMD_FFT
		e(CLFFT_NOTIMPLEMENTED)
		e(CLFFT_FILE_NOT_FOUND)
		e(CLFFT_FILE_CREATE_FAILURE)
		e(CLFFT_VERSION_MISMATCH)
		e(CLFFT_INVALID_PLAN)
		e(CLFFT_DEVICE_NO_DOUBLE)
#endif
		default: return "?";
	}
}

string command_to_string(cl_command_type t) {
	switch(t) {
		e(CL_COMMAND_NDRANGE_KERNEL      )
		e(CL_COMMAND_TASK                )
		e(CL_COMMAND_NATIVE_KERNEL       )
		e(CL_COMMAND_READ_BUFFER         )
		e(CL_COMMAND_WRITE_BUFFER        )
		e(CL_COMMAND_COPY_BUFFER         )
		e(CL_COMMAND_READ_IMAGE          )
		e(CL_COMMAND_WRITE_IMAGE         )
		e(CL_COMMAND_COPY_IMAGE          )
		e(CL_COMMAND_COPY_IMAGE_TO_BUFFER)
		e(CL_COMMAND_COPY_BUFFER_TO_IMAGE)
		e(CL_COMMAND_MAP_BUFFER          )
		e(CL_COMMAND_MAP_IMAGE           )
		e(CL_COMMAND_UNMAP_MEM_OBJECT    )
		e(CL_COMMAND_MARKER              )
		e(CL_COMMAND_ACQUIRE_GL_OBJECTS  )
		e(CL_COMMAND_RELEASE_GL_OBJECTS  )
		e(CL_COMMAND_READ_BUFFER_RECT    )
		e(CL_COMMAND_WRITE_BUFFER_RECT   )
		e(CL_COMMAND_COPY_BUFFER_RECT    )
		e(CL_COMMAND_USER                )
		default: return "?";
	}
}

string execution_status_to_string(cl_int t) {
	switch(t) {
		e(CL_COMPLETE )
		e(CL_RUNNING  )
		e(CL_SUBMITTED)
		e(CL_QUEUED   )
		default: return "?";
	}
}

#undef e
#define f(m) if(t & m) { t &= ~m; if(s.tellp() > 0) s << '|'; s << #m; }
#define other_flags() \
	if(t != 0) { \
		if(s.tellp() > 0) \
			s << '|'; \
		s << t; \
	}

string device_type_to_string(cl_device_type t) {
	ostringstream s;
	f(CL_DEVICE_TYPE_DEFAULT    )
	f(CL_DEVICE_TYPE_CPU        )
	f(CL_DEVICE_TYPE_GPU        )
	f(CL_DEVICE_TYPE_ACCELERATOR)
	f(CL_DEVICE_TYPE_ALL        )
	other_flags()
	return s.str();
}

string buffer_flags_to_string(cl_mem_flags t) {
	ostringstream s;
	f(CL_MEM_READ_WRITE    )
	f(CL_MEM_WRITE_ONLY    )
	f(CL_MEM_READ_ONLY     )
	f(CL_MEM_USE_HOST_PTR  )
	f(CL_MEM_ALLOC_HOST_PTR)
	f(CL_MEM_COPY_HOST_PTR )
	other_flags()
	return s.str();
}

#undef f
#undef other_flags


// Debug output

#define MAKE_INFO(wrapper_type, info_type, info_function) \
template<typename T> \
T info(const wrapper_type &d, info_type name) { \
	size_t size; \
	cl_int status = info_function(d.native, name, 0, NULL, &size); \
	if(status != CL_SUCCESS) \
		throw error(status, "Couldn't query info"); \
	if(size > sizeof(T)) \
		throw std::domain_error("Result type too small"); \
	T buf; \
	status = info_function(d.native, name, size, &buf, &size); \
	if(status != CL_SUCCESS) \
		throw error(status, "Couldn't query device info"); \
	return buf; \
} \
\
string info_str(const wrapper_type &d, info_type name) { \
	size_t size; \
	cl_int status = info_function(d.native, name, 0, NULL, &size); \
	if(status != CL_SUCCESS) \
		throw error(status, "Couldn't query device info"); \
	char buf[size]; \
	status = info_function(d.native, name, size, buf, &size); \
	if(status != CL_SUCCESS) \
		throw error(status, "Couldn't query device info"); \
	return buf; \
} \
\
template<typename T> \
vector<T> info_vec(const wrapper_type &d, info_type name) { \
	size_t size; \
	cl_int status = info_function(d.native, name, 0, NULL, &size); \
	if(status != CL_SUCCESS) \
		throw error(status, "Couldn't query device info"); \
	T buf[size / sizeof(T)]; \
	status = info_function(d.native, name, size, buf, &size); \
	if(status != CL_SUCCESS) \
		throw error(status, "Couldn't query device info"); \
	vector<T> out; \
	for(size_t i = 0 ; i < size / sizeof(T) ; i++) \
		out.push_back(buf[i]); \
	return out; \
}



MAKE_INFO(device, cl_device_info, clGetDeviceInfo);

ostream &cl::operator<<(ostream &o, const device &d) {
	o << "  Device (" << info_str(d, CL_DEVICE_NAME) << ")\n"
		<< "    Type: " << device_type_to_string(info<cl_device_type>(d, CL_DEVICE_TYPE)) << '\n'
		<< "    Vendor: " << info_str(d, CL_DEVICE_VENDOR) << '\n'
		<< "    Version: " << info_str(d, CL_DEVICE_VERSION) << '\n'
		<< "    Profile: " << info_str(d, CL_DEVICE_PROFILE) << '\n'
		<< "    Language Version: " << info_str(d, CL_DEVICE_OPENCL_C_VERSION) << '\n'
		<< "    Driver Version: " << info_str(d, CL_DRIVER_VERSION) << '\n'
		<< "    Memory size: " << info<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE) / (1024 * 1024) << 'M' << '\n'
		<< "    Compute units: " << info<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS) << '\n'
		<< "    Cache size: " << info<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) / 1024 << 'k' << '\n'
		<< "    Cacheline size: " << info<cl_uint>(d, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << '\n'
	;
	return o;
}



MAKE_INFO(platform, cl_platform_info, clGetPlatformInfo);

ostream &cl::operator<<(ostream &o, const platform &p) {
	o
		<< "Platform (" << info_str(p, CL_PLATFORM_NAME) << ")\n"
		<< "  Vendor: " << info_str(p, CL_PLATFORM_VENDOR) << '\n'
		<< "  Version: " << info_str(p, CL_PLATFORM_VERSION) << '\n'
		<< "  Profile: " << info_str(p, CL_PLATFORM_PROFILE) << '\n'
		<< "  Extensions: " << info_str(p, CL_PLATFORM_EXTENSIONS) << '\n'
	;
	for(auto d : p.all_devices())
		o << d << '\n';
	return o;
}



MAKE_INFO(context, cl_context_info, clGetContextInfo);

ostream &cl::operator<<(ostream &o, const context &p) {
	o << "Context:"
		<< " id=" << p.native
		<< " refs=" << info<cl_uint>(p, CL_CONTEXT_REFERENCE_COUNT)
		<< '\n';
	for(auto d : info_vec<cl_device_id>(p, CL_CONTEXT_DEVICES))
		o << device(d);
	return o;
}



MAKE_INFO(program, cl_program_info, clGetProgramInfo)

ostream &cl::operator<<(ostream &o, const program &p) {
	return o << "Program:"
		<< " id=" << p.native
		<< " devices=" << info<cl_uint>(p, CL_PROGRAM_NUM_DEVICES)
		<< " refs=" << info<cl_uint>(p, CL_PROGRAM_REFERENCE_COUNT);
}



MAKE_INFO(kernel, cl_kernel_info, clGetKernelInfo)

ostream &cl::operator<<(ostream &o, const kernel &p) {
	return o << "Kernel(" << info_str(p, CL_KERNEL_FUNCTION_NAME) << "/" << info<cl_uint>(p, CL_KERNEL_NUM_ARGS) << ')'
		<< " id=" << p.native
		<< " refs=" << info<cl_uint>(p, CL_KERNEL_REFERENCE_COUNT);
}



MAKE_INFO(event, cl_event_info, clGetEventInfo)

ostream &cl::operator<<(ostream &o, const event &p) {
	return o << "Event:"
		<< " id=" << p.native
		<< " command=" << command_to_string(info<cl_command_type>(p, CL_EVENT_COMMAND_TYPE))
		<< " status=" << execution_status_to_string(info<cl_int>(p, CL_EVENT_COMMAND_EXECUTION_STATUS))
		<< " refs=" << info<cl_uint>(p, CL_EVENT_REFERENCE_COUNT);
}



MAKE_INFO(buffer, cl_mem_info, clGetMemObjectInfo)

ostream &cl::operator<<(ostream &o, const buffer &p) {
	if(p.native == 0)
		return o << "Buffer[" << p.name << "]: (staged)"
			<< " size=" << p.size;
	return o << "Buffer[" << p.name << "]:" 
		<< " id=" << p.native
		<< " flags=" << buffer_flags_to_string(info<cl_mem_flags>(p, CL_MEM_FLAGS))
		<< " size=" << info<size_t>(p, CL_MEM_SIZE)
		<< " offset=" << info<size_t>(p, CL_MEM_OFFSET)
		<< " maps=" << info<cl_uint>(p, CL_MEM_MAP_COUNT)
		<< " refs=" << info<cl_uint>(p, CL_MEM_REFERENCE_COUNT);
}




#endif
