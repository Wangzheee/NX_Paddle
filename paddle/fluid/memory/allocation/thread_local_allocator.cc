// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/memory/allocation/thread_local_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

namespace paddle {
namespace memory {
namespace allocation {

const int MALLOC_ALIGN = 64;

#define CUDA_CALL(func)                                      \
  {                                                          \
    auto e = (func);                                         \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

void* DirectAllocator::Alloc(size_t unaligned_size) {
  if (platform::is_cpu_place(place_)) {
    size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
    char* p = static_cast<char*>(std::malloc(offset + unaligned_size));
    // Memory checking
    CHECK(p) << "Error occurred in malloc period: available space is not enough "
                "for mallocing "
            << unaligned_size << " bytes.";
    // Byte alignment
    void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                      (~(MALLOC_ALIGN - 1)));
    static_cast<void**>(r)[-1] = p;
    return r;
  } else if (platform::is_gpu_place(place_)) {
    int dev_id = BOOST_GET_CONST(platform::CUDAPlace, place_).GetDeviceId();
    platform::CUDADeviceGuard guard(dev_id);
    void* ptr{};
    CUDA_CALL(cudaMalloc(&ptr, unaligned_size));
    return ptr;
  }
  return nullptr;
}

void DirectAllocator::Free(void* ptr) {
  if (platform::is_cpu_place(place_)) {
    if (ptr) {
      std::free(static_cast<void**>(ptr)[-1]);
    } 
  } else if (platform::is_gpu_place(place_)) {
    int dev_id = BOOST_GET_CONST(platform::CUDAPlace, place_).GetDeviceId();
    platform::CUDADeviceGuard guard(dev_id);
    CUDA_CALL(cudaFree(ptr));
  }
}


ThreadLocalAllocatorImpl::ThreadLocalAllocatorImpl(const platform::Place& p)
    : place_(p) {
  if (platform::is_gpu_place(place_)) {
    direct_allocator_.reset(new DirectAllocator{place_});
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Thread local allocator only supports CUDAPlace now."));
  }
}

std::shared_ptr<ThreadLocalAllocatorImpl> ThreadLocalCUDAAllocatorPool::Get(
    int gpu_id) {
  auto pos = std::distance(devices_.begin(),
                           std::find(devices_.begin(), devices_.end(), gpu_id));
  PADDLE_ENFORCE_LT(
      pos, devices_.size(),
      platform::errors::InvalidArgument(
          "The position of device should be less than the size of devices."));
  std::call_once(*init_flags_[pos], [this, pos, gpu_id] {
    platform::SetDeviceId(devices_[pos]);
    allocators_[pos].reset(
        new ThreadLocalAllocatorImpl(platform::CUDAPlace(gpu_id)));
  });
  return allocators_[pos];
}

ThreadLocalCUDAAllocatorPool::ThreadLocalCUDAAllocatorPool()
    : devices_(platform::GetSelectedDevices()) {
  auto gpu_num = devices_.size();
  allocators_.resize(gpu_num);
  init_flags_.reserve(gpu_num);
  for (size_t i = 0; i < gpu_num; ++i) {
    init_flags_.emplace_back(new std::once_flag());
  }
}

ThreadLocalAllocation* ThreadLocalAllocatorImpl::AllocateImpl(size_t size) {
  VLOG(10) << "ThreadLocalAllocatorImpl::AllocateImpl " << size;
  void* ptr = direct_allocator_->Alloc(size);
  auto* tl_allocation = new ThreadLocalAllocation(ptr, size, place_);
  tl_allocation->SetThreadLocalAllocatorImpl(shared_from_this());
  return tl_allocation;
}

void ThreadLocalAllocatorImpl::FreeImpl(ThreadLocalAllocation* allocation) {
  VLOG(10) << "ThreadLocalAllocatorImpl::FreeImpl " << allocation;
  direct_allocator_->Free(allocation->ptr());
  delete allocation;
}

uint64_t ThreadLocalAllocatorImpl::ReleaseImpl() {
  return direct_allocator_->Release();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
