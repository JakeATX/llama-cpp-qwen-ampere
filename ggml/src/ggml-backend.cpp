// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-atx-moe-residency.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

static bool atx_moe_residency_env_enabled_early() {
    const char * global_experts = getenv("ATX_MOE_KEEP_EXPERTS");
    const char * layer_experts  = getenv("ATX_MOE_KEEP_LAYER_EXPERTS");
    return (global_experts != nullptr && global_experts[0] != '\0') ||
           (layer_experts  != nullptr && layer_experts[0]  != '\0');
}

static bool atx_tensor_name_is_packed_moe_early(const char * name) {
    if (name == nullptr) {
        return false;
    }
    const std::string s(name);
    return s.find(".ffn_gate_exps.weight")    != std::string::npos ||
           s.find(".ffn_up_exps.weight")      != std::string::npos ||
           s.find(".ffn_down_exps.weight")    != std::string::npos ||
           s.find(".ffn_gate_up_exps.weight") != std::string::npos;
}

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    const bool atx_force_moe_expert_offload =
                        tensor->op == GGML_OP_MUL_MAT_ID &&
                        i == 0 &&
                        atx_moe_residency_env_enabled_early() &&
                        atx_tensor_name_is_packed_moe_early(ggml_get_name(src));
                    if (ggml_backend_supports_op(sched->backends[b], tensor) &&
                            (ggml_backend_offload_op(sched->backends[b], tensor) || atx_force_moe_expert_offload)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_name(node->op), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

struct atx_moe_tensor_name {
    bool matched = false;
    int  layer   = -1;
    std::string kind;
};

struct atx_moe_tensor_mapping {
    std::string name;
    std::string backend;
    int layer = -1;
    int64_t n_expert = 0;
    size_t expert_size = 0;
    std::vector<int> resident_experts;
};

struct atx_moe_cache_entry {
    const ggml_tensor * input = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;
    std::unordered_map<int, ggml_tensor *> expert_tensors;
    ggml_tensor * expert_map_tensor = nullptr;
    std::unordered_set<int> hydrated;
    std::vector<int> resident_experts;
    size_t expert_size = 0;
    size_t slice_size = 0;
    size_t hot_stride_channel = 0;
    int64_t n_expert = 0;
    int64_t n_hot = 0;
    int layer = -1;
    int tensor_kind = 0;
    bool init_failed = false;
    bool warned_private = false;
};

struct atx_moe_layer_stats {
    uint64_t used_expert_slices_seen = 0;
    uint64_t resident_cache_hit_slices = 0;
    uint64_t resident_staging_copy_calls = 0;
    uint64_t resident_direct_hit_slices = 0;
    uint64_t resident_bytes_copied = 0;
    uint64_t resident_cache_hydrate_slices = 0;
    uint64_t resident_bytes_hydrated = 0;
    uint64_t cold_expert_miss_slices = 0;
    uint64_t cold_expert_miss_range_calls = 0;
    uint64_t host_expert_slice_copies = 0;
    uint64_t host_expert_range_copy_calls = 0;
    uint64_t host_expert_single_copy_calls = 0;
    uint64_t host_bytes_copied = 0;
    uint64_t host_copy_submit_ns = 0;
    uint64_t max_host_range_slices = 0;
    uint64_t coalesced_unused_gap_slices = 0;
    uint64_t decode_used_expert_slices_seen = 0;
    uint64_t decode_cold_expert_miss_slices = 0;
    uint64_t decode_resident_direct_hit_slices = 0;
    uint64_t prompt_used_expert_slices_seen = 0;
    uint64_t prompt_cold_expert_miss_slices = 0;
    uint64_t prompt_resident_direct_hit_slices = 0;
};

struct atx_moe_residency_state {
    bool initialized = false;
    bool enabled = false;
    bool stats_registered = false;
    std::string stats_path;
    std::set<int> global_experts;
    std::map<int, std::set<int>> layer_experts;
    std::mutex mutex;
    std::unordered_map<uint64_t, atx_moe_cache_entry> caches;
    std::vector<atx_moe_tensor_mapping> mappings;
    uint64_t tensors_seen = 0;
    uint64_t split_inputs_seen = 0;
    uint64_t moe_named_split_inputs_seen = 0;
    uint64_t moe_named_host_weight_inputs = 0;
    uint64_t moe_named_node_matches = 0;
    uint64_t moe_named_node_misses = 0;
    uint64_t resident_cache_hydrate_slices = 0;
    uint64_t resident_cache_hit_slices = 0;
    uint64_t resident_staging_copy_calls = 0;
    uint64_t resident_direct_hit_slices = 0;
    uint64_t direct_cache_registrations = 0;
    uint64_t direct_cache_query_hits = 0;
    uint64_t direct_cache_query_misses = 0;
    uint64_t direct_kernel_dispatch_mmvq = 0;
    uint64_t direct_kernel_dispatch_mmq = 0;
    uint64_t direct_kernel_dispatch_mmf = 0;
    uint64_t direct_kernel_fallbacks = 0;
    uint64_t direct_require_violations = 0;
    uint64_t hot_staging_violations = 0;
    uint64_t hot_staging_bytes = 0;
    uint64_t host_expert_slice_copies = 0;
    uint64_t host_expert_range_copy_calls = 0;
    uint64_t host_expert_single_copy_calls = 0;
    uint64_t host_copy_submit_ns = 0;
    uint64_t max_host_range_slices = 0;
    uint64_t coalesced_unused_gap_slices = 0;
    uint64_t cache_miss_slices = 0;
    uint64_t used_expert_slices_seen = 0;
    uint64_t cold_expert_miss_slices = 0;
    uint64_t cold_expert_miss_range_calls = 0;
    uint64_t resident_bytes_hydrated = 0;
    uint64_t resident_bytes_copied = 0;
    uint64_t host_bytes_copied = 0;
    uint64_t input_cpy_staging_bytes = 0;
    uint64_t input_backend_sync_calls = 0;
    uint64_t input_backend_sync_ns = 0;
    uint64_t ids_backend_sync_calls = 0;
    uint64_t ids_backend_sync_ns = 0;
    uint64_t resident_cache_hydrate_sync_calls = 0;
    uint64_t resident_cache_hydrate_sync_ns = 0;
    uint64_t decode_used_expert_slices_seen = 0;
    uint64_t decode_cold_expert_miss_slices = 0;
    uint64_t decode_resident_direct_hit_slices = 0;
    uint64_t prompt_used_expert_slices_seen = 0;
    uint64_t prompt_cold_expert_miss_slices = 0;
    uint64_t prompt_resident_direct_hit_slices = 0;
    std::string mode = "off";
    std::string prewarm = "lazy";
    std::string pin_cpu_experts = "auto";
    std::string cold_copy_mode = "cpu-id-readback";
    std::string policy_output_path;
    std::string trace_path;
    bool debug = false;
    int cold_coalesce_gap = 0;
    bool direct_cuda = false;
    bool direct_require = false;
    bool strict_hot_no_stage = false;
    std::map<int, atx_moe_layer_stats> layer_stats;
    std::map<int, std::map<int, uint64_t>> routed_expert_counts;
    std::unordered_map<const ggml_tensor *, ggml_atx_moe_direct_cache> direct_caches;
};

static atx_moe_residency_state & atx_moe_state() {
    static atx_moe_residency_state state;
    return state;
}

static void atx_moe_sync_backend(ggml_backend_t backend, uint64_t & calls, uint64_t & ns) {
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();
    calls++;
    ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

static std::string atx_json_escape(const std::string & s) {
    std::ostringstream out;
    for (const char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:   out << c;       break;
        }
    }
    return out.str();
}

static std::string atx_trim_copy(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}

static std::vector<int> atx_parse_int_ranges_backend(const char * env_value, int max_value, const char * label) {
    std::vector<int> out;
    if (env_value == nullptr || env_value[0] == '\0') {
        return out;
    }

    std::set<int> dedup;
    std::stringstream ss(env_value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = atx_trim_copy(item);
        if (item.empty()) {
            GGML_ABORT("ATX MoE residency: empty %s entry in '%s'\n", label, env_value);
        }

        const size_t dash = item.find('-');
        int first;
        int last;
        if (dash == std::string::npos) {
            first = last = std::stoi(item);
        } else {
            first = std::stoi(item.substr(0, dash));
            last  = std::stoi(item.substr(dash + 1));
        }
        if (first < 0 || last < first || (max_value >= 0 && last > max_value)) {
            GGML_ABORT("ATX MoE residency: invalid %s range '%s'\n", label, item.c_str());
        }
        for (int v = first; v <= last; ++v) {
            dedup.insert(v);
        }
    }

    out.assign(dedup.begin(), dedup.end());
    return out;
}

static void atx_parse_layer_experts_backend(const char * env_value, std::map<int, std::set<int>> & out) {
    if (env_value == nullptr || env_value[0] == '\0') {
        return;
    }

    std::stringstream ss(env_value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = atx_trim_copy(item);
        if (item.empty()) {
            GGML_ABORT("ATX MoE residency: empty layer-expert entry in '%s'\n", env_value);
        }

        const size_t colon = item.find(':');
        if (colon == std::string::npos || item.find(':', colon + 1) != std::string::npos) {
            GGML_ABORT("ATX MoE residency: invalid layer-expert entry '%s'\n", item.c_str());
        }

        const int layer = std::stoi(item.substr(0, colon));
        if (layer < 0) {
            GGML_ABORT("ATX MoE residency: layer id must be non-negative in '%s'\n", item.c_str());
        }

        const std::vector<int> experts = atx_parse_int_ranges_backend(item.substr(colon + 1).c_str(), 255, "expert");
        for (const int expert : experts) {
            out[layer].insert(expert);
        }
    }
}

static void atx_moe_write_stats() {
    atx_moe_residency_state & state = atx_moe_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.stats_path.empty()) {
        return;
    }

    std::ofstream out(state.stats_path);
    if (!out) {
        GGML_LOG_ERROR("ATX MoE residency: failed to write stats to %s\n", state.stats_path.c_str());
        return;
    }

    out << "{\n";
    out << "  \"enabled\": " << (state.enabled ? "true" : "false") << ",\n";
    out << "  \"mode\": \"" << atx_json_escape(state.mode) << "\",\n";
    out << "  \"direct_cuda\": " << (state.direct_cuda ? "true" : "false") << ",\n";
    out << "  \"direct_require\": " << (state.direct_require ? "true" : "false") << ",\n";
    out << "  \"strict_hot_no_stage\": " << (state.strict_hot_no_stage ? "true" : "false") << ",\n";
    out << "  \"prewarm_experts\": \"" << atx_json_escape(state.prewarm) << "\",\n";
    out << "  \"pin_cpu_experts\": \"" << atx_json_escape(state.pin_cpu_experts) << "\",\n";
    out << "  \"cold_copy_mode\": \"" << atx_json_escape(state.cold_copy_mode) << "\",\n";
    out << "  \"global_experts\": [";
    bool first = true;
    for (const int expert : state.global_experts) {
        out << (first ? "" : ",") << expert;
        first = false;
    }
    out << "],\n";
    out << "  \"layer_experts\": {";
    first = true;
    for (const auto & layer : state.layer_experts) {
        out << (first ? "" : ",") << "\n    \"" << layer.first << "\": [";
        bool first_expert = true;
        for (const int expert : layer.second) {
            out << (first_expert ? "" : ",") << expert;
            first_expert = false;
        }
        out << "]";
        first = false;
    }
    out << (state.layer_experts.empty() ? "" : "\n  ") << "},\n";
    out << "  \"counters\": {\n";
    out << "    \"tensors_seen\": " << state.tensors_seen << ",\n";
    out << "    \"split_inputs_seen\": " << state.split_inputs_seen << ",\n";
    out << "    \"moe_named_split_inputs_seen\": " << state.moe_named_split_inputs_seen << ",\n";
    out << "    \"moe_named_host_weight_inputs\": " << state.moe_named_host_weight_inputs << ",\n";
    out << "    \"moe_named_node_matches\": " << state.moe_named_node_matches << ",\n";
    out << "    \"moe_named_node_misses\": " << state.moe_named_node_misses << ",\n";
    out << "    \"resident_cache_hydrate_slices\": " << state.resident_cache_hydrate_slices << ",\n";
    out << "    \"resident_cache_hit_slices\": " << state.resident_cache_hit_slices << ",\n";
    out << "    \"resident_staging_copy_calls\": " << state.resident_staging_copy_calls << ",\n";
    out << "    \"resident_direct_hit_slices\": " << state.resident_direct_hit_slices << ",\n";
    out << "    \"direct_cache_registrations\": " << state.direct_cache_registrations << ",\n";
    out << "    \"direct_cache_query_hits\": " << state.direct_cache_query_hits << ",\n";
    out << "    \"direct_cache_query_misses\": " << state.direct_cache_query_misses << ",\n";
    out << "    \"direct_kernel_dispatch_mmvq\": " << state.direct_kernel_dispatch_mmvq << ",\n";
    out << "    \"direct_kernel_dispatch_mmq\": " << state.direct_kernel_dispatch_mmq << ",\n";
    out << "    \"direct_kernel_dispatch_mmf\": " << state.direct_kernel_dispatch_mmf << ",\n";
    out << "    \"direct_kernel_fallbacks\": " << state.direct_kernel_fallbacks << ",\n";
    out << "    \"direct_require_violations\": " << state.direct_require_violations << ",\n";
    out << "    \"hot_staging_violations\": " << state.hot_staging_violations << ",\n";
    out << "    \"hot_staging_bytes\": " << state.hot_staging_bytes << ",\n";
    out << "    \"host_expert_slice_copies\": " << state.host_expert_slice_copies << ",\n";
    out << "    \"host_expert_range_copy_calls\": " << state.host_expert_range_copy_calls << ",\n";
    out << "    \"host_expert_single_copy_calls\": " << state.host_expert_single_copy_calls << ",\n";
    out << "    \"host_copy_submit_ns\": " << state.host_copy_submit_ns << ",\n";
    out << "    \"max_host_range_slices\": " << state.max_host_range_slices << ",\n";
    out << "    \"coalesced_unused_gap_slices\": " << state.coalesced_unused_gap_slices << ",\n";
    out << "    \"cache_miss_slices\": " << state.cache_miss_slices << ",\n";
    out << "    \"used_expert_slices_seen\": " << state.used_expert_slices_seen << ",\n";
    out << "    \"cold_expert_miss_slices\": " << state.cold_expert_miss_slices << ",\n";
    out << "    \"cold_expert_miss_range_calls\": " << state.cold_expert_miss_range_calls << ",\n";
    out << "    \"resident_bytes_hydrated\": " << state.resident_bytes_hydrated << ",\n";
    out << "    \"resident_bytes_copied\": " << state.resident_bytes_copied << ",\n";
    out << "    \"host_bytes_copied\": " << state.host_bytes_copied << ",\n";
    out << "    \"input_cpy_staging_bytes\": " << state.input_cpy_staging_bytes << ",\n";
    out << "    \"input_backend_sync_calls\": " << state.input_backend_sync_calls << ",\n";
    out << "    \"input_backend_sync_ns\": " << state.input_backend_sync_ns << ",\n";
    out << "    \"ids_backend_sync_calls\": " << state.ids_backend_sync_calls << ",\n";
    out << "    \"ids_backend_sync_ns\": " << state.ids_backend_sync_ns << ",\n";
    out << "    \"resident_cache_hydrate_sync_calls\": " << state.resident_cache_hydrate_sync_calls << ",\n";
    out << "    \"resident_cache_hydrate_sync_ns\": " << state.resident_cache_hydrate_sync_ns << ",\n";
    out << "    \"decode_used_expert_slices_seen\": " << state.decode_used_expert_slices_seen << ",\n";
    out << "    \"decode_cold_expert_miss_slices\": " << state.decode_cold_expert_miss_slices << ",\n";
    out << "    \"decode_resident_direct_hit_slices\": " << state.decode_resident_direct_hit_slices << ",\n";
    out << "    \"prompt_used_expert_slices_seen\": " << state.prompt_used_expert_slices_seen << ",\n";
    out << "    \"prompt_cold_expert_miss_slices\": " << state.prompt_cold_expert_miss_slices << ",\n";
    out << "    \"prompt_resident_direct_hit_slices\": " << state.prompt_resident_direct_hit_slices << "\n";
    out << "  },\n";
    out << "  \"per_layer\": {";
    first = true;
    for (const auto & item : state.layer_stats) {
        const atx_moe_layer_stats & stats = item.second;
        out << (first ? "" : ",") << "\n    \"" << item.first << "\": {"
            << "\"used_expert_slices_seen\": " << stats.used_expert_slices_seen << ", "
            << "\"resident_cache_hit_slices\": " << stats.resident_cache_hit_slices << ", "
            << "\"resident_staging_copy_calls\": " << stats.resident_staging_copy_calls << ", "
            << "\"resident_direct_hit_slices\": " << stats.resident_direct_hit_slices << ", "
            << "\"resident_bytes_copied\": " << stats.resident_bytes_copied << ", "
            << "\"resident_cache_hydrate_slices\": " << stats.resident_cache_hydrate_slices << ", "
            << "\"resident_bytes_hydrated\": " << stats.resident_bytes_hydrated << ", "
            << "\"cold_expert_miss_slices\": " << stats.cold_expert_miss_slices << ", "
            << "\"cold_expert_miss_range_calls\": " << stats.cold_expert_miss_range_calls << ", "
            << "\"host_expert_slice_copies\": " << stats.host_expert_slice_copies << ", "
            << "\"host_expert_range_copy_calls\": " << stats.host_expert_range_copy_calls << ", "
            << "\"host_expert_single_copy_calls\": " << stats.host_expert_single_copy_calls << ", "
            << "\"host_bytes_copied\": " << stats.host_bytes_copied << ", "
            << "\"host_copy_submit_ns\": " << stats.host_copy_submit_ns << ", "
            << "\"max_host_range_slices\": " << stats.max_host_range_slices << ", "
            << "\"coalesced_unused_gap_slices\": " << stats.coalesced_unused_gap_slices << ", "
            << "\"decode_used_expert_slices_seen\": " << stats.decode_used_expert_slices_seen << ", "
            << "\"decode_cold_expert_miss_slices\": " << stats.decode_cold_expert_miss_slices << ", "
            << "\"decode_resident_direct_hit_slices\": " << stats.decode_resident_direct_hit_slices << ", "
            << "\"prompt_used_expert_slices_seen\": " << stats.prompt_used_expert_slices_seen << ", "
            << "\"prompt_cold_expert_miss_slices\": " << stats.prompt_cold_expert_miss_slices << ", "
            << "\"prompt_resident_direct_hit_slices\": " << stats.prompt_resident_direct_hit_slices
            << "}";
        first = false;
    }
    out << (state.layer_stats.empty() ? "" : "\n  ") << "},\n";
    out << "  \"bottleneck_layers\": [";
    std::vector<std::pair<int, atx_moe_layer_stats>> bottleneck_layers(state.layer_stats.begin(), state.layer_stats.end());
    std::sort(bottleneck_layers.begin(), bottleneck_layers.end(), [](const auto & a, const auto & b) {
        if (a.second.host_bytes_copied != b.second.host_bytes_copied) {
            return a.second.host_bytes_copied > b.second.host_bytes_copied;
        }
        return a.second.host_expert_single_copy_calls > b.second.host_expert_single_copy_calls;
    });
    first = true;
    for (const auto & item : bottleneck_layers) {
        const atx_moe_layer_stats & stats = item.second;
        if (stats.host_bytes_copied == 0 && stats.cold_expert_miss_slices == 0) {
            continue;
        }
        const double hot_coverage = stats.used_expert_slices_seen == 0 ? 0.0 :
            (double) stats.resident_direct_hit_slices / (double) stats.used_expert_slices_seen;
        const double single_range_ratio = stats.host_expert_range_copy_calls == 0 ? 0.0 :
            (double) stats.host_expert_single_copy_calls / (double) stats.host_expert_range_copy_calls;
        out << (first ? "" : ",") << "\n    {"
            << "\"layer\": " << item.first << ", "
            << "\"host_bytes_copied\": " << stats.host_bytes_copied << ", "
            << "\"cold_expert_miss_slices\": " << stats.cold_expert_miss_slices << ", "
            << "\"host_expert_range_copy_calls\": " << stats.host_expert_range_copy_calls << ", "
            << "\"host_expert_single_copy_calls\": " << stats.host_expert_single_copy_calls << ", "
            << "\"coalesced_unused_gap_slices\": " << stats.coalesced_unused_gap_slices << ", "
            << "\"single_range_ratio\": " << single_range_ratio << ", "
            << "\"host_copy_submit_ns\": " << stats.host_copy_submit_ns << ", "
            << "\"hot_coverage\": " << hot_coverage
            << "}";
        first = false;
    }
    out << (first ? "" : "\n  ") << "],\n";
    out << "  \"routed_expert_counts\": {";
    if (!state.routed_expert_counts.empty()) {
        out << "\n";
    }
    size_t layer_count_i = 0;
    for (const auto & layer_counts : state.routed_expert_counts) {
        out << "    \"" << layer_counts.first << "\": {";
        size_t expert_count_i = 0;
        for (const auto & expert_count : layer_counts.second) {
            out << (expert_count_i == 0 ? "" : ", ")
                << "\"" << expert_count.first << "\": " << expert_count.second;
            expert_count_i++;
        }
        out << "}";
        out << (++layer_count_i == state.routed_expert_counts.size() ? "\n" : ",\n");
    }
    out << (state.routed_expert_counts.empty() ? "" : "  ") << "},\n";
    out << "  \"tensor_mappings\": [\n";
    for (size_t i = 0; i < state.mappings.size(); ++i) {
        const atx_moe_tensor_mapping & mapping = state.mappings[i];
        out << "    {\"name\": \"" << atx_json_escape(mapping.name) << "\", "
            << "\"backend\": \"" << atx_json_escape(mapping.backend) << "\", "
            << "\"layer\": " << mapping.layer << ", "
            << "\"n_expert\": " << mapping.n_expert << ", "
            << "\"expert_size\": " << mapping.expert_size << ", "
            << "\"resident_experts\": [";
        for (size_t j = 0; j < mapping.resident_experts.size(); ++j) {
            out << (j == 0 ? "" : ",") << mapping.resident_experts[j];
        }
        out << "]}";
        out << (i + 1 == state.mappings.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void ggml_backend_atx_moe_residency_flush_stats(void) {
    atx_moe_write_stats();
}

bool ggml_backend_atx_moe_residency_direct_enabled(void) {
    atx_moe_residency_state & state = atx_moe_state();
    return state.direct_cuda;
}

bool ggml_backend_atx_moe_residency_direct_require(void) {
    atx_moe_residency_state & state = atx_moe_state();
    return state.direct_require;
}

bool ggml_backend_atx_moe_residency_strict_hot_no_stage(void) {
    atx_moe_residency_state & state = atx_moe_state();
    return state.strict_hot_no_stage;
}

void ggml_backend_atx_moe_residency_note_direct_dispatch(int kernel) {
    atx_moe_residency_state & state = atx_moe_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (kernel == GGML_ATX_MOE_DIRECT_KERNEL_MMVQ) {
        state.direct_kernel_dispatch_mmvq++;
    } else if (kernel == GGML_ATX_MOE_DIRECT_KERNEL_MMQ) {
        state.direct_kernel_dispatch_mmq++;
    } else {
        state.direct_kernel_dispatch_mmf++;
    }
}

void ggml_backend_atx_moe_residency_note_direct_fallback(const char * reason) {
    atx_moe_residency_state & state = atx_moe_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.direct_kernel_fallbacks++;
        if (state.direct_require) {
            state.direct_require_violations++;
        }
    }
    if (state.direct_require) {
        GGML_ABORT("ATX MoE direct required but CUDA direct fallback occurred: %s\n", reason ? reason : "unknown");
    }
}

bool ggml_backend_atx_moe_residency_get_direct_cache(
        const struct ggml_tensor * staged_src,
        struct ggml_atx_moe_direct_cache * out) {
    atx_moe_residency_state & state = atx_moe_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.direct_cuda) {
        if (out != nullptr) {
            *out = {};
        }
        return false;
    }
    auto found = state.direct_caches.find(staged_src);
    if (found == state.direct_caches.end()) {
        state.direct_cache_query_misses++;
        if (out != nullptr) {
            *out = {};
        }
        return false;
    }
    state.direct_cache_query_hits++;
    if (out != nullptr) {
        *out = found->second;
    }
    return true;
}

static void atx_moe_residency_init() {
    atx_moe_residency_state & state = atx_moe_state();
    if (state.initialized) {
        return;
    }

    state.initialized = true;
    const char * global_experts = getenv("ATX_MOE_KEEP_EXPERTS");
    const char * layer_experts  = getenv("ATX_MOE_KEEP_LAYER_EXPERTS");
    const char * stats_path     = getenv("ATX_MOE_RESIDENCY_STATS");
    const char * direct_cuda    = getenv("ATX_MOE_DIRECT_CUDA");
    const char * mode           = getenv("ATX_MOE_RESIDENCY_MODE");
    const char * direct_require = getenv("ATX_MOE_DIRECT_REQUIRE");
    const char * strict_hot     = getenv("ATX_MOE_DIRECT_STRICT_HOT_NO_STAGE");
    const char * prewarm        = getenv("ATX_MOE_PREWARM_EXPERTS");
    const char * pin_cpu        = getenv("ATX_MOE_PIN_CPU_EXPERTS");
    const char * cold_copy      = getenv("ATX_MOE_COLD_COPY_MODE");
    const char * policy_output  = getenv("ATX_MOE_POLICY_OUTPUT");
    const char * trace_path     = getenv("ATX_MOE_RESIDENCY_TRACE");
    const char * debug          = getenv("ATX_MOE_RESIDENCY_DEBUG");
    const char * coalesce_gap   = getenv("ATX_MOE_COLD_COALESCE_GAP");

    for (const int expert : atx_parse_int_ranges_backend(global_experts, 255, "expert")) {
        state.global_experts.insert(expert);
    }
    atx_parse_layer_experts_backend(layer_experts, state.layer_experts);

    if (mode != nullptr && mode[0] != '\0') {
        state.mode = mode;
    } else if (direct_cuda != nullptr && direct_cuda[0] != '\0' && direct_cuda[0] != '0') {
        state.mode = "direct";
    } else if (!state.global_experts.empty() || !state.layer_experts.empty()) {
        state.mode = "exact-v1";
    }
    if (state.mode != "off" && state.mode != "layer" && state.mode != "exact-v1" &&
            state.mode != "direct" && state.mode != "hybrid" && state.mode != "auto") {
        GGML_ABORT("ATX MoE residency: invalid mode '%s'\n", state.mode.c_str());
    }

    state.enabled = state.mode != "off" && (!state.global_experts.empty() || !state.layer_experts.empty());
    state.direct_cuda = state.enabled &&
        (state.mode == "direct" || state.mode == "hybrid" || state.mode == "auto" ||
         (direct_cuda != nullptr && direct_cuda[0] != '\0' && direct_cuda[0] != '0'));
    state.direct_require = direct_require != nullptr && direct_require[0] != '\0' && direct_require[0] != '0';
    state.strict_hot_no_stage = strict_hot != nullptr && strict_hot[0] != '\0' && strict_hot[0] != '0';
    if (prewarm != nullptr && prewarm[0] != '\0') {
        state.prewarm = prewarm;
    }
    if (pin_cpu != nullptr && pin_cpu[0] != '\0') {
        state.pin_cpu_experts = pin_cpu;
    }
    if (cold_copy != nullptr && cold_copy[0] != '\0') {
        state.cold_copy_mode = cold_copy;
    }
    if (policy_output != nullptr && policy_output[0] != '\0') {
        state.policy_output_path = policy_output;
    }
    if (trace_path != nullptr && trace_path[0] != '\0') {
        state.trace_path = trace_path;
    }
    state.debug = debug != nullptr && debug[0] != '\0' && debug[0] != '0';
    if (coalesce_gap != nullptr && coalesce_gap[0] != '\0') {
        state.cold_coalesce_gap = std::max(0, std::stoi(coalesce_gap));
    }

    if (stats_path != nullptr && stats_path[0] != '\0') {
        state.stats_path = stats_path;
    }
    if (!state.stats_path.empty() && !state.stats_registered) {
        std::atexit(atx_moe_write_stats);
        state.stats_registered = true;
    }

    if (state.enabled) {
        GGML_LOG_INFO("ATX MoE residency: mode=%s with %zu global experts and %zu layer-specific entries%s%s\n",
                state.mode.c_str(), state.global_experts.size(), state.layer_experts.size(),
                state.direct_cuda ? " (direct CUDA)" : "",
                state.strict_hot_no_stage ? " (strict hot-no-stage)" : "");
    }
}

static atx_moe_tensor_name atx_parse_moe_tensor_name(const char * name) {
    atx_moe_tensor_name parsed;
    if (name == nullptr) {
        return parsed;
    }

    const std::string s(name);
    const std::string prefix = "blk.";
    const size_t p = s.find(prefix);
    if (p == std::string::npos) {
        return parsed;
    }
    const size_t layer_start = p + prefix.size();
    const size_t layer_end = s.find('.', layer_start);
    if (layer_end == std::string::npos) {
        return parsed;
    }

    const std::string suffix = s.substr(layer_end + 1);
    const bool is_moe =
        suffix == "ffn_gate_exps.weight" ||
        suffix == "ffn_up_exps.weight" ||
        suffix == "ffn_down_exps.weight" ||
        suffix == "ffn_gate_up_exps.weight";
    if (!is_moe) {
        return parsed;
    }

    parsed.matched = true;
    parsed.layer = std::stoi(s.substr(layer_start, layer_end - layer_start));
    parsed.kind = suffix;
    return parsed;
}

static std::vector<int> atx_resident_experts_for_layer(const atx_moe_residency_state & state, int layer, int64_t n_expert) {
    std::set<int> selected = state.global_experts;
    auto it = state.layer_experts.find(layer);
    if (it != state.layer_experts.end()) {
        selected.insert(it->second.begin(), it->second.end());
    }

    std::vector<int> out;
    for (const int expert : selected) {
        if (expert >= 0 && expert < n_expert) {
            out.push_back(expert);
        }
    }
    return out;
}

static uint64_t atx_cache_key(const ggml_tensor * input, int backend_id) {
    return (uint64_t) (uintptr_t) input ^ ((uint64_t) (uint32_t) backend_id << 48);
}

static atx_moe_cache_entry * atx_moe_get_cache_entry(
        ggml_backend_sched_t sched,
        const ggml_tensor * input,
        int split_backend_id,
        int64_t n_expert,
        size_t expert_size,
        const std::vector<int> & resident_experts,
        int layer) {
    atx_moe_residency_state & state = atx_moe_state();
    const uint64_t key = atx_cache_key(input, split_backend_id);

    auto found = state.caches.find(key);
    if (found != state.caches.end()) {
        return found->second.init_failed ? nullptr : &found->second;
    }

    atx_moe_cache_entry entry;
    entry.input = input;
    entry.n_expert = n_expert;
    entry.expert_size = expert_size;
    // The legacy staging cache keeps a little padding after each slice because
    // input_cpy may be consumed by kernels that read padded packed tensors.
    // Direct CUDA mode consumes the compact cache as the source tensor, so the
    // hot expert stride must match the original quantized expert block layout.
    entry.slice_size = expert_size + (state.direct_cuda ? 0 : std::min<size_t>(expert_size, 512));
    entry.hot_stride_channel = entry.slice_size / ggml_type_size(input->type);
    entry.layer = layer;
    entry.resident_experts = resident_experts;
    entry.n_hot = (int64_t) resident_experts.size();
    const atx_moe_tensor_name parsed_tensor = atx_parse_moe_tensor_name(ggml_get_name(input));
    if (parsed_tensor.kind == "ffn_gate_exps.weight") {
        entry.tensor_kind = 1;
    } else if (parsed_tensor.kind == "ffn_up_exps.weight") {
        entry.tensor_kind = 2;
    } else if (parsed_tensor.kind == "ffn_down_exps.weight") {
        entry.tensor_kind = 3;
    } else if (parsed_tensor.kind == "ffn_gate_up_exps.weight") {
        entry.tensor_kind = 4;
    }

    const size_t ctx_size = ggml_tensor_overhead() * (resident_experts.size() + 6) + 4096;
    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    entry.ctx = ggml_init(ctx_params);
    if (entry.ctx == nullptr) {
        entry.init_failed = true;
        state.cache_miss_slices++;
        auto inserted = state.caches.emplace(key, entry);
        return inserted.first->second.init_failed ? nullptr : &inserted.first->second;
    }

    const size_t hot_size = entry.slice_size * resident_experts.size();
    const size_t alignment = ggml_backend_buft_get_alignment(sched->bufts[split_backend_id]);
    const size_t map_offset = GGML_PAD(hot_size, alignment);
    const size_t map_size = (size_t) n_expert * sizeof(int32_t);
    const size_t total_size = map_offset + map_size;
    entry.buffer = ggml_backend_buft_alloc_buffer(sched->bufts[split_backend_id], total_size);
    if (entry.buffer == nullptr) {
        entry.init_failed = true;
        state.cache_miss_slices++;
        auto inserted = state.caches.emplace(key, entry);
        return inserted.first->second.init_failed ? nullptr : &inserted.first->second;
    }

    void * base = ggml_backend_buffer_get_base(entry.buffer);
    for (size_t i = 0; i < resident_experts.size(); ++i) {
        ggml_tensor * t = ggml_new_tensor_1d(entry.ctx, GGML_TYPE_I8, entry.slice_size);
        ggml_format_name(t, "atx_moe_cache_L%d_E%d", layer, resident_experts[i]);
        ggml_backend_tensor_alloc(entry.buffer, t, (uint8_t *) base + i * entry.slice_size);
        entry.expert_tensors[resident_experts[i]] = t;
    }
    entry.expert_map_tensor = ggml_new_tensor_1d(entry.ctx, GGML_TYPE_I32, n_expert);
    ggml_format_name(entry.expert_map_tensor, "atx_moe_cache_L%d_expert_map", layer);
    ggml_backend_tensor_alloc(entry.buffer, entry.expert_map_tensor, (uint8_t *) base + map_offset);

    std::vector<int32_t> expert_map((size_t) n_expert, -1);
    for (size_t i = 0; i < resident_experts.size(); ++i) {
        expert_map[(size_t) resident_experts[i]] = (int32_t) i;
    }
    ggml_backend_tensor_set_async(sched->backends[split_backend_id], entry.expert_map_tensor, expert_map.data(), 0, map_size);
    ggml_backend_synchronize(sched->backends[split_backend_id]);

    atx_moe_tensor_mapping mapping;
    mapping.name = ggml_get_name(input);
    mapping.backend = ggml_backend_buft_name(sched->bufts[split_backend_id]);
    mapping.layer = layer;
    mapping.n_expert = n_expert;
    mapping.expert_size = expert_size;
    mapping.resident_experts = resident_experts;
    state.mappings.push_back(std::move(mapping));

    auto inserted = state.caches.emplace(key, std::move(entry));
    return inserted.first->second.init_failed ? nullptr : &inserted.first->second;
}

static bool atx_moe_copy_resident_expert(
        ggml_backend_sched_t sched,
        ggml_backend_t split_backend,
        ggml_tensor * input_cpy,
        const ggml_tensor * input,
        int split_backend_id,
        int32_t expert_id,
        int64_t n_expert,
        size_t expert_size,
        const std::vector<int> & resident_experts,
        int layer,
        bool is_decode,
        bool allow_direct_cuda) {
    atx_moe_residency_state & state = atx_moe_state();
    if (state.direct_cuda && !allow_direct_cuda) {
        return false;
    }

    atx_moe_cache_entry * cache = atx_moe_get_cache_entry(sched, input, split_backend_id, n_expert, expert_size, resident_experts, layer);
    if (cache == nullptr) {
        state.cache_miss_slices++;
        return false;
    }

    auto tensor_it = cache->expert_tensors.find(expert_id);
    if (tensor_it == cache->expert_tensors.end()) {
        state.cache_miss_slices++;
        return false;
    }

    if (state.direct_cuda && cache->expert_map_tensor != nullptr && cache->buffer != nullptr) {
        ggml_atx_moe_direct_cache direct{};
        direct.packed_src = input->data;
        direct.hot_data = ggml_backend_buffer_get_base(cache->buffer);
        direct.expert_map = (const int32_t *) cache->expert_map_tensor->data;
        direct.type = (int) input->type;
        direct.layer = cache->layer;
        direct.tensor_kind = cache->tensor_kind;
        direct.n_expert = cache->n_expert;
        direct.n_hot = cache->n_hot;
        direct.expert_stride_bytes = cache->expert_size;
        direct.hot_stride_bytes = cache->slice_size;
        direct.hot_stride_channel = cache->hot_stride_channel;
        state.direct_caches[input] = direct;
        state.direct_caches[input_cpy] = direct;
        state.direct_cache_registrations++;
    }

    ggml_tensor * cache_tensor = tensor_it->second;
    const size_t expert_offset = (size_t) expert_id * expert_size;
    const size_t padding = std::min<size_t>(expert_size, 512);
    const size_t copy_size = expert_size + (!state.direct_cuda && expert_id < n_expert - 1 ? padding : 0);

    if (cache->hydrated.find(expert_id) == cache->hydrated.end()) {
        ggml_backend_tensor_set_async(split_backend, cache_tensor, (const uint8_t *) input->data + expert_offset, 0, copy_size);
        atx_moe_sync_backend(split_backend, state.resident_cache_hydrate_sync_calls, state.resident_cache_hydrate_sync_ns);
        cache->hydrated.insert(expert_id);
        state.resident_cache_hydrate_slices++;
        state.resident_bytes_hydrated += copy_size;
        if (layer >= 0) {
            atx_moe_layer_stats & layer_stats = state.layer_stats[layer];
            layer_stats.resident_cache_hydrate_slices++;
            layer_stats.resident_bytes_hydrated += copy_size;
        }
    }

    const char * buft_name = ggml_backend_buft_name(sched->bufts[split_backend_id]);
    if (buft_name != nullptr && std::string(buft_name).find("_Private") != std::string::npos) {
        if (!cache->warned_private) {
            GGML_LOG_WARN("ATX MoE residency: backend %s uses private buffers; resident slice reuse falls back to host source copies\n", buft_name);
            cache->warned_private = true;
        }
        state.cache_miss_slices++;
        return false;
    }

    if (state.direct_cuda) {
        state.resident_cache_hit_slices++;
        state.resident_direct_hit_slices++;
        if (is_decode) {
            state.decode_resident_direct_hit_slices++;
        } else {
            state.prompt_resident_direct_hit_slices++;
        }
        if (layer >= 0) {
            atx_moe_layer_stats & layer_stats = state.layer_stats[layer];
            layer_stats.resident_cache_hit_slices++;
            layer_stats.resident_direct_hit_slices++;
            if (is_decode) {
                layer_stats.decode_resident_direct_hit_slices++;
            } else {
                layer_stats.prompt_resident_direct_hit_slices++;
            }
        }
        return true;
    }

    ggml_backend_tensor_set_async(split_backend, input_cpy, cache_tensor->data, expert_offset, copy_size);
    state.resident_cache_hit_slices++;
    state.resident_staging_copy_calls++;
    state.resident_bytes_copied += copy_size;
    state.input_cpy_staging_bytes += copy_size;
    if (layer >= 0) {
        atx_moe_layer_stats & layer_stats = state.layer_stats[layer];
        layer_stats.resident_cache_hit_slices++;
        layer_stats.resident_staging_copy_calls++;
        layer_stats.resident_bytes_copied += copy_size;
    }
    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    atx_moe_residency_init();
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            atx_moe_residency_state & atx_loop_state = atx_moe_state();
            const atx_moe_tensor_name atx_loop_tensor = atx_loop_state.enabled ? atx_parse_moe_tensor_name(ggml_get_name(input)) : atx_moe_tensor_name();
            if (atx_loop_state.enabled) {
                atx_loop_state.split_inputs_seen++;
                if (atx_loop_tensor.matched) {
                    atx_loop_state.moe_named_split_inputs_seen++;
                    if (ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                            ggml_backend_buffer_is_host(input->buffer)) {
                        atx_loop_state.moe_named_host_weight_inputs++;
                    }
                }
            }

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = nullptr;
                if (ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                        ggml_backend_buffer_is_host(input->buffer)) {
                    for (int node_id = 0; node_id < split->graph.n_nodes; ++node_id) {
                        ggml_tensor * candidate = split->graph.nodes[node_id];
                        if (candidate->op == GGML_OP_MUL_MAT_ID && candidate->src[0] == input_cpy) {
                            node = candidate;
                            break;
                        }
                    }
                }
                if (atx_loop_state.enabled && atx_loop_tensor.matched) {
                    if (node != nullptr) {
                        atx_loop_state.moe_named_node_matches++;
                    } else {
                        atx_loop_state.moe_named_node_misses++;
                    }
                }
                if (node != nullptr) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];
                    atx_moe_residency_state & atx_state = atx_moe_state();
                    const atx_moe_tensor_name atx_tensor = atx_parse_moe_tensor_name(ggml_get_name(input));
                    const std::vector<int> atx_resident_experts =
                        atx_state.enabled && atx_tensor.matched ?
                        atx_resident_experts_for_layer(atx_state, atx_tensor.layer, n_expert) :
                        std::vector<int>();
                    const bool atx_use_resident_cache = atx_state.enabled && !atx_resident_experts.empty();
                    const bool atx_direct_cuda_supported_node =
                        atx_state.direct_cuda &&
                        node->op == GGML_OP_MUL_MAT_ID &&
                        ggml_is_quantized(input->type) &&
                        input->type != GGML_TYPE_TQ4_1S &&
                        input->type != GGML_TYPE_TQ3_1S;
                    if (atx_state.direct_cuda && atx_use_resident_cache && !atx_direct_cuda_supported_node) {
                        atx_state.direct_kernel_fallbacks++;
                        if (atx_state.direct_require) {
                            atx_state.direct_require_violations++;
                            GGML_ABORT("ATX MoE direct required but tensor '%s' cannot use direct CUDA (type=%s, op=%d)\n",
                                    ggml_get_name(input), ggml_type_name(input->type), (int) node->op);
                        }
                    }

                    atx_moe_sync_backend(input_backend, atx_state.input_backend_sync_calls, atx_state.input_backend_sync_ns);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        // Metal get_async wraps the destination pointer with newBufferWithBytesNoCopy,
                        // which is not safe for arbitrary std::vector allocations on this path.
                        // A synchronous get is fine here because we immediately need the routed IDs.
                        atx_moe_sync_backend(ids_backend, atx_state.ids_backend_sync_calls, atx_state.ids_backend_sync_ns);
                        ggml_backend_tensor_get(ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        atx_moe_sync_backend(ids_backend, atx_state.ids_backend_sync_calls, atx_state.ids_backend_sync_ns);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                                if (atx_tensor.matched && atx_tensor.layer >= 0) {
                                    atx_state.routed_expert_counts[atx_tensor.layer][id]++;
                                }
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }
                    const bool atx_is_decode_step = (ids_tensor->ne[1] <= 1);

                    atx_state.tensors_seen++;

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        if (atx_state.direct_cuda && atx_state.strict_hot_no_stage && !atx_resident_experts.empty()) {
                            uint64_t hot_slices_in_range = 0;
                            for (const int resident_id : atx_resident_experts) {
                                if (resident_id >= first_id && resident_id <= last_id) {
                                    hot_slices_in_range++;
                                }
                            }
                            if (hot_slices_in_range != 0) {
                                const uint64_t bytes = hot_slices_in_range * (uint64_t) expert_size;
                                atx_state.hot_staging_violations += hot_slices_in_range;
                                atx_state.hot_staging_bytes += bytes;
                                if (atx_tensor.matched) {
                                    atx_state.layer_stats[atx_tensor.layer].resident_staging_copy_calls += hot_slices_in_range;
                                }
                                GGML_ABORT("ATX MoE direct strict hot-no-stage violation: layer=%d range=%d..%d includes %llu hot expert slices\n",
                                        atx_tensor.layer, first_id, last_id, (unsigned long long) hot_slices_in_range);
                            }
                        }
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        const auto copy_t0 = std::chrono::steady_clock::now();
                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                        const auto copy_t1 = std::chrono::steady_clock::now();
                        const uint64_t copied_slices = (uint64_t) (last_id - first_id + 1);
                        const uint64_t copied_bytes = (uint64_t) (expert_size_copy + padding_end);
                        const uint64_t copy_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(copy_t1 - copy_t0).count();
                        atx_state.host_expert_slice_copies += copied_slices;
                        atx_state.host_expert_range_copy_calls++;
                        atx_state.host_copy_submit_ns += copy_ns;
                        atx_state.max_host_range_slices = std::max<uint64_t>(atx_state.max_host_range_slices, copied_slices);
                        if (copied_slices == 1) {
                            atx_state.host_expert_single_copy_calls++;
                        }
                        atx_state.host_bytes_copied += copied_bytes;
                        atx_state.input_cpy_staging_bytes += copied_bytes;
                        if (atx_tensor.matched) {
                            atx_moe_layer_stats & layer_stats = atx_state.layer_stats[atx_tensor.layer];
                            layer_stats.host_expert_slice_copies += copied_slices;
                            layer_stats.host_expert_range_copy_calls++;
                            layer_stats.host_copy_submit_ns += copy_ns;
                            layer_stats.max_host_range_slices = std::max<uint64_t>(layer_stats.max_host_range_slices, copied_slices);
                            if (copied_slices == 1) {
                                layer_stats.host_expert_single_copy_calls++;
                            }
                            layer_stats.host_bytes_copied += copied_bytes;
                        }
                    };

                    if (atx_use_resident_cache) {
                        std::vector<uint8_t> resident_mask((size_t) n_expert, 0);
                        for (const int resident_id : atx_resident_experts) {
                            if (resident_id >= 0 && resident_id < n_expert) {
                                resident_mask[(size_t) resident_id] = 1;
                            }
                        }
                        int32_t cold_first_id = -1;
                        int32_t cold_last_id = -1;
                        int32_t cold_gap_slices = 0;
                        auto flush_cold_experts = [&]() {
                            if (cold_first_id < 0) {
                                return;
                            }
                            atx_state.cold_expert_miss_range_calls++;
                            if (atx_tensor.matched) {
                                atx_state.layer_stats[atx_tensor.layer].cold_expert_miss_range_calls++;
                            }
                            copy_experts(cold_first_id, cold_last_id);
                            cold_first_id = -1;
                            cold_last_id = -1;
                            cold_gap_slices = 0;
                        };
                        for (int32_t id = 0; id < n_expert; ++id) {
                            if (!ggml_bitset_get(used_ids.data(), id)) {
                                if (cold_first_id >= 0 && atx_state.cold_coalesce_gap > 0 &&
                                        cold_gap_slices < atx_state.cold_coalesce_gap &&
                                        !(atx_state.strict_hot_no_stage && resident_mask[(size_t) id] != 0)) {
                                    cold_last_id = id;
                                    cold_gap_slices++;
                                    atx_state.coalesced_unused_gap_slices++;
                                    if (atx_tensor.matched) {
                                        atx_state.layer_stats[atx_tensor.layer].coalesced_unused_gap_slices++;
                                    }
                                    continue;
                                }
                                flush_cold_experts();
                                continue;
                            }
                            atx_state.used_expert_slices_seen++;
                            if (atx_is_decode_step) {
                                atx_state.decode_used_expert_slices_seen++;
                            } else {
                                atx_state.prompt_used_expert_slices_seen++;
                            }
                            if (atx_tensor.matched) {
                                atx_moe_layer_stats & layer_stats = atx_state.layer_stats[atx_tensor.layer];
                                layer_stats.used_expert_slices_seen++;
                                if (atx_is_decode_step) {
                                    layer_stats.decode_used_expert_slices_seen++;
                                } else {
                                    layer_stats.prompt_used_expert_slices_seen++;
                                }
                            }
                            if (resident_mask[(size_t) id] != 0 &&
                                    atx_moe_copy_resident_expert(sched, split_backend, input_cpy, input, split_backend_id,
                                        id, n_expert, expert_size, atx_resident_experts, atx_tensor.layer, atx_is_decode_step, atx_direct_cuda_supported_node)) {
                                if (atx_state.direct_cuda && !atx_state.strict_hot_no_stage && cold_first_id >= 0) {
                                    // Keep an existing cold host-copy range coalesced across
                                    // resident hot experts. Direct CUDA will consume the hot
                                    // cache, but copying through the hot slice is often cheaper
                                    // than splitting the host transfer into tiny ranges.
                                    cold_last_id = id;
                                } else {
                                    flush_cold_experts();
                                }
                                continue;
                            }
                            atx_state.cold_expert_miss_slices++;
                            if (atx_is_decode_step) {
                                atx_state.decode_cold_expert_miss_slices++;
                            } else {
                                atx_state.prompt_cold_expert_miss_slices++;
                            }
                            if (atx_tensor.matched) {
                                atx_moe_layer_stats & layer_stats = atx_state.layer_stats[atx_tensor.layer];
                                layer_stats.cold_expert_miss_slices++;
                                if (atx_is_decode_step) {
                                    layer_stats.decode_cold_expert_miss_slices++;
                                } else {
                                    layer_stats.prompt_cold_expert_miss_slices++;
                                }
                            }
                            if (cold_first_id < 0) {
                                cold_first_id = id;
                                cold_last_id = id;
                                cold_gap_slices = 0;
                            } else if (id == cold_last_id + 1) {
                                cold_last_id = id;
                                cold_gap_slices = 0;
                            } else {
                                flush_cold_experts();
                                cold_first_id = id;
                                cold_last_id = id;
                                cold_gap_slices = 0;
                            }
                        }
                        flush_cold_experts();
                    } else {
                        int id = 0;
                        while (id < n_expert && !ggml_bitset_get(used_ids.data(), id)) {
                            id++;
                        }
                        if (id < n_expert) {
                            int32_t first_id = id;
                            int32_t last_id = first_id;

                            for (++id; id < n_expert; ++id) {
                                if (!ggml_bitset_get(used_ids.data(), id)) {
                                    continue;
                                }

                                if (id == last_id + 1) {
                                    last_id = id;
                                    continue;
                                }

                                copy_experts(first_id, last_id);

                                first_id = id;
                                last_id = id;
                            }
                            copy_experts(first_id, last_id);
                        }
                    }
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
