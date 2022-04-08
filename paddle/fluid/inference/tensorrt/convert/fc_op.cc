/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
    nvinfer1::ILayer* regist_fc (nvinfer1::ITensor* inputs, int n_output,
                         TensorRTEngine::Weight& weight,
                         TensorRTEngine::Weight& bias,const std::string activation_type ) {
      auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *inputs,
                                            n_output, weight.get(), bias.get());
      if (activation_type == "relu") {
        nvinfer1::IActivationLayer* relu_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Activation, *(fc_layer->getOutput(0)),
                                 nvinfer1::ActivationType::kRELU);
        return relu_layer;
      } else {
        return fc_layer;
      }
    }
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";
    framework::OpDesc op_desc(op, nullptr);

    auto input_names = op_desc.InputNames();
    auto output_name = op_desc.Output("Out").front();
    bool with_bias = input_names.size() >= 3;
    std::string w_name = "Y";
    std::string i_name = "X";
    if (with_bias) {
      w_name = "W";
      i_name = "Input";
    }
    // Declare inputs
    std::string x_name = op_desc.Input(i_name).front();
    auto* X = engine_->GetITensor(x_name);
    std::cout << "fc input name : " << x_name << "\n";
    auto x_dim = X->getDimensions();
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v, platform::errors::NotFound(
                 "Can not find %s presistale var of fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? BOOST_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";
    const bool transpose_Y =
        op_desc.HasAttr("transpose_Y")
            ? BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_Y"))
            : false;
    // This may trigger a GPU->CPU copy, because TRT's weight can only be
    // assigned from CPU memory, which can't be avoided.
    float* weight_data = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    float in_scale = 0.;
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      CHECK(op_desc.HasAttr(i_name + "_scale"));
      in_scale =
          BOOST_GET_CONST(float, op_desc.GetAttr(i_name + "_scale")) * 127;
      auto weight_scale =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("weight_scale"));
      weight_data = engine_->GetWeightCPUData(op_desc.Input(w_name).front(),
                                              Y_t, true, weight_scale);
      engine_->SetTensorDynamicRange(X, in_scale);
#endif
    } else {
      weight_data =
          engine_->GetWeightCPUData(op_desc.Input(w_name).front(), Y_t, false);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The fc's weight should be a matrix with 2 dims, but "
                          "it's %d-dimensional.",
                          Y_t->dims().size()));  // a matrix
    size_t n_output = Y_t->dims()[1];

    int m = Y_t->dims()[0];
    int n = Y_t->dims()[1];

    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };

    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(Y_t->numel());
    memcpy(weight_data_tmp.data(), weight_data, Y_t->numel() * sizeof(float));
    if (!transpose_Y) {
      tranpose_weight(weight_data_tmp.data(), weight_data, m, n);
    }
    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(Y_t->numel())};
    weight.dims.assign({n, m});

    float* bias_data = nullptr;
    int bias_num = 0;
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<framework::LoDTensor>();
      bias_data =
          engine_->GetWeightCPUData(op_desc.Input("Bias").front(), b_t, false);
      bias_num = b_t->numel();
    }
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_num)};

    nvinfer1::Dims reshape_before_fc_dim;
    nvinfer1::Dims reshape_after_fc_dim;
    if (engine_->with_dynamic_shape()) {
      if( x_dim.nbDims ==2 ){
         reshape_before_fc_dim.nbDims = 4;
         reshape_before_fc_dim.d[0] = 0;
         reshape_before_fc_dim.d[1] = 0;
         reshape_before_fc_dim.d[2] = 1;
         reshape_before_fc_dim.d[3] = 1;	 
         auto* reshape_before_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
         reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
         auto* fc_layer = regist_fc(reshape_before_fc_layer->getOutput(0), n_output, weight, bias,activation_type);
         reshape_after_fc_dim.nbDims = 2;
         reshape_after_fc_dim.d[0] = 0;
         reshape_after_fc_dim.d[1] = 0;
	 auto* reshape_after_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(fc_layer->getOutput(0)));
         reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
	 RreplenishLayerAndOutput(reshape_after_fc_layer, "fc",
                                   {output_name}, test_mode);

      }else if(x_dim.nbDims ==3 ){
         reshape_before_fc_dim.nbDims = 4;
         reshape_before_fc_dim.d[0] = 0;
         reshape_before_fc_dim.d[1] = 0;
         reshape_before_fc_dim.d[2] = 0;
         reshape_before_fc_dim.d[3] = 1;
         auto* reshape_before_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
         reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
	 auto* fc_layer = regist_fc(reshape_before_fc_layer->getOutput(0), n_output, weight, bias,activation_type);
         reshape_after_fc_dim.nbDims = 3;
         reshape_after_fc_dim.d[0] = 0;
         reshape_after_fc_dim.d[1] = 0;
	 reshape_after_fc_dim.d[2] = 0;
         auto* reshape_after_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(fc_layer->getOutput(0)));
         reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
         RreplenishLayerAndOutput(reshape_after_fc_layer, "fc",
                                   {output_name}, test_mode);
      } else if( x_dim.nbDims ==4){
         auto* fc_layer = regist_fc(X, n_output, weight, bias,activation_type);
         RreplenishLayerAndOutput(fc_layer, "fc",
                                   {output_name}, test_mode);
      }else{
        PADDLE_THROW(platform::errors::Fatal(
          "fc shape error."));  
      }
    }else{
      if( x_dim.nbDims ==1 ){
         reshape_before_fc_dim.nbDims = 3;
         reshape_before_fc_dim.d[0] = 0;
         reshape_before_fc_dim.d[1] = 1;
         reshape_before_fc_dim.d[2] = 1;
	 auto* reshape_before_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
         reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
         auto* fc_layer = regist_fc(reshape_before_fc_layer->getOutput(0), n_output, weight, bias,activation_type);
         reshape_after_fc_dim.nbDims = 1;
         reshape_after_fc_dim.d[0] = 0;
         auto* reshape_after_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(fc_layer->getOutput(0)));
         reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
         RreplenishLayerAndOutput(reshape_after_fc_layer, "fc",
                                   {output_name}, test_mode);
      }else if(x_dim.nbDims ==2 ){
         reshape_before_fc_dim.nbDims = 3;
         reshape_before_fc_dim.d[0] = 0;
         reshape_before_fc_dim.d[1] = 0;
         reshape_before_fc_dim.d[2] = 1;
	 auto* reshape_before_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
         reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
         auto* fc_layer = regist_fc(reshape_before_fc_layer->getOutput(0), n_output, weight, bias,activation_type);
         reshape_after_fc_dim.nbDims = 2;
         reshape_after_fc_dim.d[0] = 0;
         reshape_after_fc_dim.d[1] = 0;
         auto* reshape_after_fc_layer =
           TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(fc_layer->getOutput(0)));
         reshape_after_fc_layer->setReshapeDimensions(reshape_after_fc_dim);
         RreplenishLayerAndOutput(reshape_after_fc_layer, "fc",
                                   {output_name}, test_mode);
      } else if( x_dim.nbDims ==3){
       auto* fc_layer = regist_fc(X, n_output, weight, bias,activation_type);
       RreplenishLayerAndOutput(fc_layer, "fc",
                                   {output_name}, test_mode);
      }else{
        PADDLE_THROW(platform::errors::Fatal(
          "fc shape error."));
      }
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
