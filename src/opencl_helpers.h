#ifndef __OPENCL_HELPERS_H__
#define __OPENCL_HELPERS_H__

#include <CL/cl.h>
#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include <vector>
#include <stdexcept>
#include <boost/type_traits/is_same.hpp>

/**
 * Simple C++ wrappers for OpenCL because <code>cl.hpp</code> is weird.
 */
namespace cl {


/**
 * Returns a string describing the status returned by an OpenCL call.
 */
std::string status_to_string(cl_int status);

/**
 * An OpenCL error, includes the status in its message.
 */
class error : public std::runtime_error {
	public:
	error(cl_int status, std::string message)
		: std::runtime_error(status_to_string(status) + ": " + message) {}
};


struct context;
template<typename T> struct buffer;
struct program;
struct event;


/**
 * An OpenCL device.
 */
class device {
	friend class platform;
	friend class context;

	cl_device_id native_device;
	device(cl_device_id o) : native_device(o) {}

public:
	/**
	 * Retrieve some information about the device.
	 */
	template<typename T>
	typename std::enable_if<!(boost::is_same<T, std::string>::value || boost::is_same<T, char *>::value), T>::type
	info(cl_device_info name) const {
		size_t size;
		cl_int status = clGetDeviceInfo(native_device, name, 0, NULL, &size);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't query device info");
		if(size > sizeof(T))
			throw std::domain_error("Result type too small");
		T buf;
		status = clGetDeviceInfo(native_device, name, size, &buf, &size);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't query device info");
		return buf;
	}

	template<typename T>
	typename std::enable_if<boost::is_same<T, std::string>::value, T>::type
	info(cl_device_info name) const {
		size_t size;
		cl_int status = clGetDeviceInfo(native_device, name, 0, NULL, &size);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't query device info");
		char buf[size];
		status = clGetDeviceInfo(native_device, name, size, buf, &size);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't query device info");
		return buf;
	}
};

/**
 * Print human-readable information about the device. 
 */
std::ostream &operator<<(std::ostream &, const device &);


/**
 * A platform to run OpenCL computations, consists of devices which can be used in groups.
 */
class platform {
	friend std::vector<platform> all_platforms();
	friend class context;

	cl_platform_id native_platform;
	platform(cl_platform_id o) : native_platform(o) {}

public:
	/**
	 * Retrieve some information about the platform.
	 */
	std::string info(cl_platform_info) const;

	/**
	 * Returns a list of all devices of the given type, for this platform.
	 */
	std::vector<device> all_devices(cl_device_type type = CL_DEVICE_TYPE_ALL) const;
};

/**
 * Print human-readable information about the platform.
 */
std::ostream &operator<<(std::ostream &, const platform &);

/**
 * Return a list of all available platforms.
 */
std::vector<platform> all_platforms();




/**
 * The OpenCL context. Used for all access to the core.
 */
class context {
	friend class program;
	friend class kernel;
	template<typename T> friend class buffer;

	cl_context ctx;
	cl_device_id device;
	cl_command_queue queue;

	void construct(platform p);
public:
	context() { construct(all_platforms().front()); }
	explicit context(platform p) { construct(p); }

	/**
	 * Creates a buffer of a certain size.
	 *
	 * @param count
	 * 	allocate <code>sizeof(T) * count</code> bytes.
	 * @param flags
	 * 	limit access to increase performance: CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY, CL_MEM_COPY_HOST_READ_ONLY or CL_MEM_COPY_HOST_NO_ACCESS
	 */
	template<typename T>
	buffer<T> create_buffer(size_t count, cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR) {
		cl_int status;
		cl_mem m = clCreateBuffer(ctx, flags, sizeof(T) * count, NULL, &status);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't create CL buffer");
		return buffer<T>(this, m, count);
	}

	/**
	 * Creates a buffer and fills it with the data given.
	 *
	 * @param data
	 * 	An array of the type and length of the buffer. The data is copied into the buffer.
	 * @param flags
	 * 	limit access to increase performance: CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY, CL_MEM_COPY_HOST_READ_ONLY or CL_MEM_COPY_HOST_NO_ACCESS
	 */
	template<typename T, size_t n>
	buffer<T> create_buffer(std::array<T, n> &data, cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR) {
		cl_int status;
		cl_mem m = clCreateBuffer(ctx, flags | CL_MEM_COPY_HOST_PTR, sizeof(T) * n, &data[0], &status);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't create CL buffer");
		return buffer<T>(this, m, n);
	}


	/**
	 * Load a program from a .cl code file.
	 */
	program load_program(std::string filename);


	/**
	 * Returns when all the events have occured.
	 */
	void wait(std::vector<event> events);
};


/**
 * A CL event, returned from asynchronous operations.
 * Wait for it using {@ref context::wait}.
 */
class event {
	friend class kernel;
	friend class context;
	template<typename T> friend class buffer;

	cl_event native_event;
	event(cl_event e = nullptr) : native_event(e) {}
};


/**
 * A buffer that might be accessed from the CPU and GPU
 */
template<typename T>
class buffer {
	friend class context;
	friend class kernel;
	
	cl_mem native_buffer;
	size_t count;
	context *parent_context;

	buffer(context * parent_context, cl_mem b = nullptr, size_t count = 0) : native_buffer(b), count(count), parent_context(parent_context) {}

public:
	template<size_t n>
	event write(const std::array<T, n> &data, size_t offset = 0, std::vector<event> wait = 0) {
		return write(&data[0], n, offset, wait);
	}

	event write(const T *data, size_t size, size_t offset = 0, std::vector<event> wait = 0) {
		if(offset + size > count)
			throw std::range_error("Data too large for buffer.");
		cl_event wait_list[wait.size()];
		for(size_t i = 0 ; i < wait.size() ; i++)
			wait_list[i] = wait[i].native_event;
		cl_event e;
		cl_uint status = clEnqueueWriteBuffer(
			parent_context->queue,
			native_buffer,
			CL_FALSE, // blocking
			sizeof(T) * offset,
			sizeof(T) * size,
			data,
			wait.size(),
			wait.size() == 0 ? NULL : wait_list,
			&e
		);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't queue writing to CL buffer");
		return event(e);
	}


	template<size_t n>
	event read(std::array<T, n> &data, size_t offset = 0, std::vector<event> wait = 0) {
		return read(&data[0], n, offset, wait);
	}

	event read(T *data, size_t size, size_t offset, std::vector<event> wait) {
		if(offset + size > count)
			throw std::range_error("Data too large for buffer.");
		cl_event wait_list[wait.size()];
		for(size_t i = 0 ; i < wait.size() ; i++)
			wait_list[i] = wait[i].native_event;
		cl_event e;
		cl_uint status = clEnqueueReadBuffer(
			parent_context->queue,
			native_buffer,
			CL_FALSE, // blocking
			sizeof(T) * offset,
			sizeof(T) * size,
			data,
			wait.size(),
			wait.size() == 0 ? NULL : wait_list,
			&e
		);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't queue reading from CL buffer");
		return event(e);
	}
};


/**
 * One kernel in a CL program.
 */
class kernel {
	friend class program;

	cl_kernel k;
	context *parent_context;
	kernel(context *parent_context, cl_kernel k) : k(k), parent_context(parent_context) {}

	template<size_t i>
	void argsN() {}

	template<size_t i, typename Head, typename... Tail>
	void argsN(Head head, Tail... tail) {
		arg(i, head);
		argsN<i + 1>(tail...);
	}

	public:
	/**
	 * Set one argument of the kernel to a new value.
	 */
	template<typename T>
	void arg(cl_uint i, const T &data) {
		cl_uint status = clSetKernelArg(
			k,
			i,
			sizeof(T),
			&data
		);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't set kernel argument to value");
	}

	template<typename T>
	void arg(cl_uint i, const buffer<T> &buf) {
		cl_uint status = clSetKernelArg(
			k,
			i,
			sizeof(cl_mem),
			&buf.native_buffer
		);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't set kernel argument to buffer");
	}

	/**
	 * Set all arguments of the kernel to new values.
	 */
	template<typename... T>
	void args(T... a) {
		argsN<0u>(a...);
	}


	/**
	 * Run the kernel with the requested configuration.
	 *
	 * @param size
	 * 	The number of work items in every dimension
	 * @param offset
	 * 	The offset for every dimension, or an empty list to use 0.
	 * @param group_size
	 * 	The local work group size for every dimension, or an emtpy list to allocate work automatically.
	 * @param wait
	 * 	A list of events to occur before this operation can be run.
	 */
	event run(std::vector<size_t> size, std::vector<size_t> offset = 0, std::vector<size_t> group_size = 0, std::vector<event> wait = 0);

	/**
	 * Run the kernel with the requested configuration.
	 *
	 * @param size
	 * 	The number of work items in every dimension
	 * @param wait
	 * 	A list of events to occur before this operation can be run.
	 */
	event run(std::vector<size_t> size, std::vector<event> wait) {
		return run(size, {}, {}, wait);
	}

};


/**
 * A program is a collection of kernels.
 */
class program {
	friend class context;

	cl_program native_program;
	context *parent;

	program(context *parent, std::string filename);

	public:
	/**
	 * Retrieve one kernel by its function name.
	 * Throws an {@ref error} if not found.
	 */
	kernel operator[](std::string name);
};




}; // cl::

#endif
