// Adapted from https://github.com/acids-ircam/nn_tilde/blob/master/src/backend/backend.cpp

#include "backend.h"
#include "parsing_utils.h"
#include <algorithm>
#include <iostream>
#include <stdlib.h>


#define CPU torch::kCPU
#define CUDA torch::kCUDA
#define MPS torch::kMPS

//#ifdef WIN_VERSION
#include <filesystem>
//#endif

FCPPN::FCPPN() : m_loaded(0), m_device(CPU), m_use_gpu(false) {
    at::init_num_threads();
  }

Backend::Backend() : m_loaded(0), m_device(CPU), m_use_gpu(false) {
  at::init_num_threads();
}

void Backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer,
                      int n_vec, // m_buffer_size
                      std::string method,
                      int n_batches) {
  c10::InferenceMode guard;

  auto params = get_method_params(method);
  // std::cout << "in_buffer length : " << in_buffer.size() << std::endl;
  // std::cout << "out_buffer length : " << out_buffer.size() << std::endl;

  if (!params.size())
    return;

  auto in_dim = params[0];
  auto in_ratio = params[1];
  auto out_dim = params[2];
  auto out_ratio = params[3];

  if (!m_loaded)
    return;

  // COPY BUFFER INTO A TENSOR
  std::vector<at::Tensor> tensor_in;
  // for (auto buf : in_buffer)
  for (int i(0); i < in_buffer.size(); i++) {
    tensor_in.push_back(torch::from_blob(in_buffer[i], {1, 1, n_vec}));
  }

  auto cat_tensor_in = torch::cat(tensor_in, 1);
  cat_tensor_in = cat_tensor_in.reshape({in_dim, n_batches, -1, in_ratio});
  cat_tensor_in = cat_tensor_in.select(-1, -1); // this line gets rid of the last dimension
  cat_tensor_in = cat_tensor_in.permute({1, 0, 2}); // -> {n_batches, in_dim, n latent per buffer}
    
    
  // SEND TENSOR TO DEVICE
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  cat_tensor_in = cat_tensor_in.to(m_device);
  std::vector<torch::jit::IValue> inputs = {cat_tensor_in};

  // PROCESS TENSOR
  at::Tensor tensor_out;
  try {
    tensor_out = m_model.get_method(method)(inputs).toTensor();
    tensor_out = tensor_out.repeat_interleave(out_ratio).reshape(
        {n_batches, out_dim, -1});
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return;
  }
  model_lock.unlock();

  int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
      out_n_vec(tensor_out.size(2));

  // for (int b(0); b < out_batches; b++) {
  //   for (int c(0); c < out_channels; c++) {
  //     std::cout << b << ";" << c << ";" << tensor_out[b][c].min().item<float>() << std::endl;
  //   }
  // }

  // CHECKS ON TENSOR SHAPE
  if (out_batches * out_channels != out_buffer.size()) {
    std::cout << "bad out_buffer size, expected " << out_batches * out_channels
              << " buffers, got " << out_buffer.size() << "!\n";
    return;
  }

  if (out_n_vec != n_vec) {
    std::cout << "model output size is not consistent, expected " << n_vec
              << " samples, got " << out_n_vec << "!\n";
    return;
  }

  tensor_out = tensor_out.to(CPU);
  tensor_out = tensor_out.reshape({out_batches * out_channels, -1});
  auto out_ptr = tensor_out.contiguous().data_ptr<float>();

  for (int i(0); i < out_buffer.size(); i++) {
    memcpy(out_buffer[i], out_ptr + i * n_vec, n_vec * sizeof(float));
  }
}

int Backend::load(std::string path) {
  try {
    auto model = torch::jit::load(path);
    model.eval();
    model.to(m_device);

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    m_model = model;
    m_loaded = 1;
    model_lock.unlock();

    m_available_methods = get_available_methods();
    m_path = path;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

int Backend::reload() {
  auto return_code = load(m_path);
  return return_code;
}

bool Backend::has_method(std::string method_name) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &m : m_model.get_methods()) {
    if (m.name() == method_name)
      return true;
  }
  return false;
}

bool Backend::has_settable_attribute(std::string attribute) {
  for (const auto &a : get_settable_attributes()) {
    if (a == attribute)
      return true;
  }
  return false;
}

std::vector<std::string> Backend::get_available_methods() {
  std::vector<std::string> methods;
  try {
    std::vector<c10::IValue> dumb_input = {};

    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto methods_from_model =
        m_model.get_method("get_methods")(dumb_input).toList();
    model_lock.unlock();

    for (int i = 0; i < methods_from_model.size(); i++) {
      methods.push_back(methods_from_model.get(i).toStringRef());
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    for (const auto &m : m_model.get_methods()) {
      try {
        auto method_params = m_model.attr(m.name() + "_params");
        methods.push_back(m.name());
      } catch (...) {
      }
    }
    model_lock.unlock();
  }
  return methods;
}

std::vector<std::string> Backend::get_available_attributes() {
  std::vector<std::string> attributes;
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  for (const auto &attribute : m_model.named_attributes())
    attributes.push_back(attribute.name);
  return attributes;
}

std::vector<std::string> Backend::get_settable_attributes() {
  std::vector<std::string> attributes;
  try {
    std::vector<c10::IValue> dumb_input = {};
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto methods_from_model =
        m_model.get_method("get_attributes")(dumb_input).toList();
    model_lock.unlock();
    for (int i = 0; i < methods_from_model.size(); i++) {
      attributes.push_back(methods_from_model.get(i).toStringRef());
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    for (const auto &a : m_model.named_attributes()) {
      try {
        auto method_params = m_model.attr(a.name + "_params");
        attributes.push_back(a.name);
      } catch (...) {
      }
    }
    model_lock.unlock();
  }
  return attributes;
}

std::vector<c10::IValue> Backend::get_attribute(std::string attribute_name) {
  std::string attribute_getter_name = "get_" + attribute_name;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto attribute_getter = m_model.get_method(attribute_getter_name);
    model_lock.unlock();
  } catch (...) {
    throw "getter for attribute " + attribute_name + " not found in model";
  }
  std::vector<c10::IValue> getter_inputs = {}, attributes;
  try {
    try {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      attributes = m_model.get_method(attribute_getter_name)(getter_inputs)
                       .toList()
                       .vec();
      model_lock.unlock();
    } catch (...) {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      auto output_tuple =
          m_model.get_method(attribute_getter_name)(getter_inputs).toTuple();
      attributes = (*output_tuple.get()).elements();
      model_lock.unlock();
    }
  } catch (...) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    attributes.push_back(
        m_model.get_method(attribute_getter_name)(getter_inputs));
    model_lock.unlock();
  }
  return attributes;
}

std::string Backend::get_attribute_as_string(std::string attribute_name) {
  std::vector<c10::IValue> getter_outputs = get_attribute(attribute_name);
  // finstringd arguments
  torch::Tensor setter_params;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    setter_params = m_model.attr(attribute_name + "_params").toTensor();
    model_lock.unlock();
  } catch (...) {
    throw "parameters to set attribute " + attribute_name +
        " not found in model";
  }
  std::string current_attr = "";
  for (int i = 0; i < setter_params.size(0); i++) {
    int current_id = setter_params[i].item().toInt();
    switch (current_id) {
    // bool case
    case 0: {
      current_attr += (getter_outputs[i].toBool()) ? "true" : "false";
      break;
    }
    // int case
    case 1: {
      current_attr += std::to_string(getter_outputs[i].toInt());
      break;
    }
    // float case
    case 2: {
      float result = getter_outputs[i].to<float>();
      current_attr += std::to_string(result);
      break;
    }
    // str case
    case 3: {
      current_attr += getter_outputs[i].toStringRef();
      break;
    }
    default: {
      throw "bad type id : " + std::to_string(current_id) + "at index " +
          std::to_string(i);
      break;
    }
    }
    if (i < setter_params.size(0) - 1)
      current_attr += " ";
  }
  return current_attr;
}

void Backend::set_attribute(std::string attribute_name,
                            std::vector<std::string> attribute_args) {
  // find setter
  std::string attribute_setter_name = "set_" + attribute_name;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto attribute_setter = m_model.get_method(attribute_setter_name);
    model_lock.unlock();
  } catch (...) {
    throw "setter for attribute " + attribute_name + " not found in model";
  }
  // find arguments
  torch::Tensor setter_params;
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    setter_params = m_model.attr(attribute_name + "_params").toTensor();
    model_lock.unlock();
  } catch (...) {
    throw "parameters to set attribute " + attribute_name +
        " not found in model";
  }
  // process inputs
  std::vector<c10::IValue> setter_inputs = {};
  for (int i = 0; i < setter_params.size(0); i++) {
    int current_id = setter_params[i].item().toInt();
    switch (current_id) {
    // bool case
    case 0:
      setter_inputs.push_back(c10::IValue(to_bool(attribute_args[i])));
      break;
    // int case
    case 1:
      setter_inputs.push_back(c10::IValue(to_int(attribute_args[i])));
      break;
    // float case
    case 2:
      setter_inputs.push_back(c10::IValue(to_float(attribute_args[i])));
      break;
    // str case
    case 3:
      setter_inputs.push_back(c10::IValue(attribute_args[i]));
      break;
    default:
      throw "bad type id : " + std::to_string(current_id) + "at index " +
          std::to_string(i);
      break;
    }
  }
  try {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    auto setter_out = m_model.get_method(attribute_setter_name)(setter_inputs);
    model_lock.unlock();
    int setter_result = setter_out.toInt();
    if (setter_result != 0) {
      throw "setter returned -1";
    }
  } catch (...) {
    throw "setter for " + attribute_name + " failed";
  }
}

std::vector<int> Backend::get_method_params(std::string method) {
  std::vector<int> params;

  if (std::find(m_available_methods.begin(), m_available_methods.end(),
                method) != m_available_methods.end()) {
    try {
      std::unique_lock<std::mutex> model_lock(m_model_mutex);
      auto p = m_model.attr(method + "_params").toTensor();
      model_lock.unlock();
      for (int i(0); i < 4; i++)
        params.push_back(p[i].item().to<int>());
    } catch (...) {
    }
  }
  return params;
}


int Backend::get_higher_ratio() {
  int higher_ratio = 1;
  for (const auto &method : m_available_methods) {
    auto params = get_method_params(method);
    if (!params.size())
      continue; // METHOD NOT USABLE, SKIPPING
    int max_ratio = std::max(params[1], params[3]);
    higher_ratio = std::max(higher_ratio, max_ratio);
  }
  return higher_ratio;
}

bool Backend::is_loaded() { return m_loaded; }

void Backend::use_gpu(bool value) {
  std::unique_lock<std::mutex> model_lock(m_model_mutex);
  if (value) {
    if (torch::hasCUDA()) {
      std::cout << "sending model to cuda" << std::endl;
      m_device = CUDA;
    } else if (torch::hasMPS()) {
      std::cout << "sending model to mps" << std::endl;
      m_device = MPS;
    } else {
      std::cout << "sending model to cpu" << std::endl;
      m_device = CPU;
    }
  } else {
    m_device = CPU;
  }
  m_model.to(m_device);
}

void FCPPN::use_gpu(bool value) {
    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    if (value) {
        if (torch::hasCUDA()) {
          std::cout << "sending model to cuda" << std::endl;
          m_device = CUDA;
        } else if (torch::hasMPS()) {
          std::cout << "sending model to mps" << std::endl;
          m_device = MPS;
        } else {
          std::cout << "sending model to cpu" << std::endl;
          m_device = CPU;
        }
    } else {
        m_device = CPU;
    }
    m_model->to(m_device);
    sample_tensor = sample_tensor.to(m_device);
    model_lock.unlock();
}

bool FCPPN::save(std::string save_path, std::string save_name, int m_in_dim, int m_out_dim,
                    int m_cmax, float m_gauss_scale, int m_mapping_size) {
//    std::unique_lock<std::mutex> model_lock(m_model_mutex);
    try {
        m_model->to(CPU);
        // TODO: this is not a save path:
        torch::serialize::OutputArchive archive;
        m_model->save(archive);
//        archive << value;
        
        archive.write("m_in_dim", torch::tensor(m_in_dim));
        archive.write("m_out_dim", torch::tensor(m_out_dim));
        archive.write("m_cmax", torch::tensor(m_cmax));
        archive.write("m_gauss_scale", torch::tensor(m_gauss_scale));
        archive.write("m_mapping_size", torch::tensor(m_mapping_size));

        // Save the archive to the specified file path
        std::string src_path_str = (std::filesystem::path(save_path) / save_name).string();
        archive.save_to(src_path_str);
        
        m_model->to(m_device);
        return true;
    } catch (const std::exception &e) {
        m_model->to(m_device);
        throw e;
        return false;
    }
}

void FCPPN::init_optimizer(float lr) {
    optimizer.reset();
    optimizer = std::make_unique<torch::optim::Adam>(m_model->parameters(), lr);
    optimizer_init = true;
}

float FCPPN::train_for(int epoch){
    size_t batch_count = 0;
    float loss_log = 0.0;
    for (int i(0); i < epoch; i++){
        for (auto& batch : *data_loader) {
            // Lock per batch (not per cycle) so the audio thread's try_lock in
            // cppn_infer keeps succeeding and the live latent updates during
            // training. PReLU/Linear-only net => train()/eval() forward are
            // identical, so an interleaved audio read between batches is safe.
            torch::Tensor loss;
            {
                std::unique_lock<std::mutex> model_lock(m_model_mutex);
                m_model->train();
                optimizer->zero_grad();
                torch::Tensor batch_data = batch.data.to(m_device);
                torch::Tensor prediction = m_model->forward(batch_data);

                torch::Tensor batch_target = batch.target.to(m_device);

                loss = torch::mse_loss(prediction, batch_target);
                loss.backward();

                optimizer->step();
                m_model->eval();
            }
            // sync the scalar loss outside the lock to avoid stalling inference
            loss_log += loss.to(torch::kCPU).item<float>();
            batch_count++;
        }
    }
    if (batch_count == 0){
        throw std::runtime_error("no batches were found for training");
    }
    loss_log /= batch_count;
    return loss_log;
}
