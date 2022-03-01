op_library(tensorrt_engine_op DEPS tensorrt_engine tensorrt_converter)
file(APPEND ${pybind_file} "USE_NO_KERNEL_OP__(tensorrt_engine);\n")
file(APPEND ${pybind_file} "USE_OP__(load_combine);\n")
file(APPEND ${pybind_file} "USE_OP__(load);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(cast);\n")
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP__(multiclass_nms2);\n")
file(APPEND ${pybind_file} "USE_CPU_ONLY_OP__(multiclass_nms);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(box_coder);\n")
file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP__(density_prior_box);\n")
nv_test(test_tensorrt_engine_op SRCS tensorrt_engine_op_test.cc
  DEPS tensorrt_engine_op
  analysis)
