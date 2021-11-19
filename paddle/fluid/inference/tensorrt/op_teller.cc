// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

// Just tell by the op_types.
struct SimpleOpTypeSetTeller : public Teller {
  SimpleOpTypeSetTeller() {
#if IS_TRT_VERSION_GE(5130)
    teller_set.insert("relu6");
    teller_set.insert("hard_sigmoid");
    teller_set.insert("clip");
    int8_teller_set.insert("relu6");
    int8_teller_set.insert("hard_sigmoid");
    int8_teller_set.insert("clip");
#endif
#if IS_TRT_VERSION_GE(6000)
    teller_set.insert("fused_embedding_eltwise_layernorm");
    teller_set.insert("multihead_matmul");
    teller_set.insert("skip_layernorm");
    teller_set.insert("slice");
#endif
  }

  bool operator()(const std::string& op_type, const framework::OpDesc& desc,
                  bool use_no_calib_int8) override {
    if (use_no_calib_int8) {
      return int8_teller_set.count(op_type);
    } else {
      return teller_set.count(op_type);
    }
  }

 private:
  // use this set for no calib int8.
  std::unordered_set<std::string> int8_teller_set{"mul",
                                                  "conv2d",
                                                  "conv2d_fusion",
                                                  "pool2d",
                                                  "relu",
                                                  "depthwise_conv2d",
                                                  "softmax",
                                                  "sigmoid",
                                                  "batch_norm",
                                                  "elementwise_add",
                                                  "leaky_relu",
                                                  "fc",
                                                  "concat",
                                                  "scale",
                                                  "elementwise_mul",
                                                  "conv2d_transpose",
                                                  "hard_swish"};
  std::unordered_set<std::string> teller_set{
      "mul",
      "matmul",
      "conv2d",
      "conv2d_fusion",
      "pool2d",
      "relu",
      "softmax",
      "sigmoid",
      "hard_swish",
      "depthwise_conv2d",
      "batch_norm",
      "concat",
      "tanh",
      "pad",
      "elementwise_add",
      "elementwise_mul",
      "dropout",
      "prelu",
      "conv2d_transpose",
      "leaky_relu",
      "fc",
      "shuffle_channel",
      "swish",
      "split",
      "instance_norm",
      "gelu",
      "layer_norm",
      "scale",
      "stack",
      "transpose2",
      "reshape2",
      "reshape",
      "flatten2",
      "flatten",
      "nearest_interp",
      "yolo_box",
  };
};

bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc,
                    bool use_no_calib_int8) {
  // do not support the op which is labeled the `skip_quant`
  if ((desc.HasAttr("namescope") &&
       BOOST_GET_CONST(std::string, desc.GetAttr("op_namescope")) ==
           "/skip_quant_2/") ||
      desc.HasAttr("skip_quant"))
    return false;

  for (auto& teller : tellers_) {
    if (op_type == "pool2d" || op_type == "conv2d" ||
        op_type == "depthwise_conv2d" || op_type == "conv2d_transpose") {
      std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));

      std::string padding_algorithm = "EXPLICIT";
      if (desc.HasAttr("padding_algorithm"))
        padding_algorithm =
            BOOST_GET_CONST(std::string, desc.GetAttr("padding_algorithm"));
      if (paddings.size() > 2 ||
          (padding_algorithm == "SAME" && op_type != "pool2d"))
        return false;
    }
    if (op_type == "transpose2") {
      if (!desc.HasAttr("axis")) return false;
    }
/*    if (op_type == "concat") {
      if (!desc.HasAttr("axis")) return false;
      int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));
      if(axis == 2){
        std::cout << "concat axis=2 not supported.\n";
        return false;
      }
    }
*/    if (op_type == "matmul") {
      auto* block = desc.Block();
      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVar(var_name);
          const auto shape = var_desc->GetShape();
          if (shape.size() < 3) {
            VLOG(1)
                << "matmul op dims < 3 not supported in tensorrt, but got dims "
                << shape.size() << ", so jump it.";
            return false;
          }
        }
      }
    }
    if (op_type == "reshape2") {
      if (!desc.HasAttr("shape")) return false;
      std::vector<int> shape =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("shape"));
      auto* block = desc.Block();
      std::string input_name = desc.Input("X")[0];
      auto* var_desc = block->FindVar(input_name);
      const auto input_shape = var_desc->GetShape();
      if (shape.size() <= 1 || input_shape.size() <= 1) return false;
      int input_volume = 1;
      int shape_volume = 1;
      for (size_t i=1; i<input_shape.size(); i++) {
        input_volume *= input_shape[i];
      }
      for (size_t i=1; i<shape.size(); i++) {
        shape_volume *= shape[i];
      }
      if(input_volume != shape_volume) {
        VLOG(2) << "reshape2 input volume not equals to shape volume, not convert to TRT.";
	return false;
      }

/*      std::cout << "input shape: ";
        std::cout << "[";
        for (size_t i=0; i<input_shape.size(); i++) {
          std::cout << input_shape[i] << ", ";
        }
        std::cout << "]\n";
      std::cout << "attr shape: ";
        std::cout << "[";
        for (size_t i=0; i<shape.size(); i++) {
          std::cout << shape[i] << ", ";
        }
        std::cout << "]\n";
*/    }
    if (op_type == "nearest_interp") {
      return false;
    }
    if ((*teller)(op_type, desc, use_no_calib_int8)) return true;
  }
  return false;
}

OpTeller::OpTeller() { tellers_.emplace_back(new SimpleOpTypeSetTeller); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
