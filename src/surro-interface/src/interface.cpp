// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2020 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>

int main() {
  auto graph = tf_utils::LoadGraph("../models/model.pb");
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
  if (graph == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  const std::vector<std::int64_t> input_dims = {1,11, 4};
  const std::vector<float> input_vals = {
  5.3, 66.4, 2.4, 2.13,
  2.3, 2.1, 1.1, 1.1,
  5.3, 66.4, 2.4, 2.13,
  2.3, 2.1, 55.1, 1.1,
  5.3, 66.4, 2.4, 2.13,
  2.3, 2.1, 1.1, 1.1,
  5.3, 66.4, 2.4, 2.13,
   2.3, 2.1, 1.1, 1.1,
  5.3, 66.4, 2.4, 2.13,
  2.3, 2.1, 1.1, 1.1,
  5.3, 66.4, 2.4, 2.13
  };  

  const std::vector<TF_Output> input_ops = {{TF_GraphOperationByName(graph, "input_layer"), 0}}; 
  const std::vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, input_dims, input_vals)};
  SCOPE_EXIT{ tf_utils::DeleteTensors(input_tensors); }; // Auto-delete on scope exit.

  const std::vector<TF_Output> out_ops = {{TF_GraphOperationByName(graph, "output_layer/BiasAdd"), 0}}; 
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); }; // Auto-delete on scope exit.

  auto session = tf_utils::CreateSession(graph);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); }; // Auto-delete on scope exit.
  if (session == nullptr) {
    std::cout << "Can't create session" << std::endl;
    return 2;
  }
  
  auto code = tf_utils::RunSession(session, input_ops, input_tensors, out_ops, output_tensors);
  
  if (code == TF_OK) {
    auto result = tf_utils::GetTensorData<float>(output_tensors[0]);
    
    std::cout << "output vals:" << std::endl;
    for(int i = 0; i < result.size(); i+=2)
      printf("%.7lf %.7lf\n" , result[i], result[i+1]);
  } else {
    std::cout << "Error run session TF_CODE: " << code;
    return code;
  }

  return 0;
}
