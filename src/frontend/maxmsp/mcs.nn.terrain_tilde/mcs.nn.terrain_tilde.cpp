#include "../../../backend/backend.h"
#include "../shared/circular_buffer.h"

#include "c74_min.h"

#include <stdio.h>
#include <iostream>
#include <format>

#include <algorithm>
#include <chrono>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#include "../shared/min_path.h"
#include "../shared/min_dictionary.h"

#include "../shared/utils.h"

#include <math.h>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

#define CPU torch::kCPU
#define CUDA torch::kCUDA
#define MPS torch::kMPS

#define None at::indexing::None
#define Slice torch::indexing::Slice

using namespace c74::min;
using namespace c74::min::ui;

std::string min_devkit_path() {
#ifdef WIN_VERSION
    char    pathstr[4096];
    HMODULE hm = nullptr;
    if (!GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)&min_devkit_path, &hm)) {
        int ret = GetLastError();
        fprintf(stderr, "GetModuleHandle() returned %d\n", ret);
    }
    GetModuleFileNameA(hm, pathstr, sizeof(pathstr));
    // path now is the path to this external's binary, including the binary filename.
    auto filename = strrchr(pathstr, '\\');
    if (filename)
        *filename = 0;
    auto externals = strrchr(pathstr, '\\');
    if (externals)
        *externals = 0;
    path p{ pathstr };    // convert to Max path
    return p;
#endif    // WIN_VERSION

#ifdef MAC_VERSION
    CFBundleRef this_bundle = CFBundleGetBundleWithIdentifier(CFSTR("com.jasperzheng.mcs-nn-terrain-"));
    CFURLRef    this_url = CFBundleCopyExecutableURL(this_bundle);
    char        this_path[4096];
    CFURLGetFileSystemRepresentation(this_url, true, reinterpret_cast<UInt8*>(this_path), 4096);
    string this_path_str{ this_path };
    CFRelease(this_url);
    // we now have a path like this:
    // /Users/tim/Materials/min-devkit/externals/min.project.mxo/Contents/MacOS/min.project"
    // so we need to chop off 5 slashes from the end
    auto iter = this_path_str.find("/externals/mcs.nn.terrain~.mxo/Contents/MacOS/mcs.nn.terrain~");
    this_path_str.erase(iter, strlen("/externals/mcs.nn.terrain~.mxo/Contents/MacOS/mcs.nn.terrain~"));
    return this_path_str;
#endif    // MAC_VERSION
}

// Max multichannel negotiation callbacks (registered in maxclass_setup)
long simplemc_multichanneloutputs(c74::max::t_object *x, long index, long count);
long simplemc_inputchanged(c74::max::t_object *x, long index, long count);

class mcs_nn_terrain : public object<mcs_nn_terrain>, public mc_operator<> {
    
public:
    MIN_DESCRIPTION{"Latent terrain priors for neural audio autoencoders, for generating latent vectors"};
    MIN_TAGS        { "" };
    MIN_AUTHOR{"Jasper Shuoyang Zheng"};
    MIN_RELATED        { "nn~, nn.terrain.gui, nn.terrain.encode, nn.terrain.record" };


    // INLETS OUTLETS
    std::vector<std::unique_ptr<outlet<>>> m_outlets;
    std::vector<std::unique_ptr<inlet<>>> m_inlets;

    mcs_nn_terrain(const atoms &args = {});
    ~mcs_nn_terrain();
    
    std::vector<std::unique_ptr<buffer_reference>> buffers;
    
    
    // BACKEND
    std::unique_ptr<FCPPN> cppn_model;
    
    bool cppn_init = false;
    c74::min::path t_path;
    
    int m_in_dim{2}, m_in_ratio{2048}, m_out_dim{8}, m_out_ratio{1}, m_cmax{128}, m_mapping_size{256};
    float m_gauss_scale{0.1};
    
    
    // BUFFER RELATED MEMBERS
    int m_buffer_size{2048};
    int m_in_count{0};                                          // counts collected input samples until a full buffer
    std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;
    std::vector<std::unique_ptr<float[]>> m_latent_out;
    std::vector<float> m_in_point;                              // flat control points, row-major [voice][m_in_dim]
    std::vector<float> m_out_flat;                              // flat inference output, row-major [voice][m_out_dim]

    // MULTICHANNEL (mcs): one inlet per voice, each carrying `m_in_dim` channels;
    // one outlet per voice, each carrying `m_out_dim` channels. Voice count fixed by arg.
    int m_voices{1};
    int m_batch{1};
    int get_batches() { return m_voices; }

    // AUDIO PERFORM
    bool m_use_thread{true}, m_should_stop_perform_thread{false};
    std::unique_ptr<std::thread> m_train_thread{nullptr}, m_compute_thread{nullptr};
    std::binary_semaphore m_should_train_lock{0}, m_should_plot_lock{0};
    std::binary_semaphore m_finish_train_lock{1}, m_finish_plot_lock{1};
    

    atoms create_dataloader();
    bool load_param_from_file(string model_path, torch::serialize::InputArchive &archive);
    void operator()(audio_bundle input, audio_bundle output);
    void perform(audio_bundle input, audio_bundle output);
    void cppn_infer(float *, float *);
//    atoms freeze_terrain(int width_x, int height_y, string terrain_name, int stride);
    
    static void model_train_loop(mcs_nn_terrain *nn_instance);
    static void model_plot_loop(mcs_nn_terrain *nn_instance);
    
//    at::Tensor sample_tensor;
    min_dict sampled_dict{symbol(true)};
    min_dict terrain_dict{symbol(true)};
    string sampled_str;
    float x_lo_prev{0.0}, x_hi_prev{0.0}, y_lo_prev{0.0}, y_hi_prev{0.0}, stride_prev{0.0};
    int x_res_prev{0}, y_res_prev{0}, plot_resolution_prev{0}, c_prev{0};
    void sample_interval(float x_lo, float x_hi, int x_res,
                         float y_lo, float y_hi, int y_res, int c);
    void create_sample_tensor();
    
    float twoPi = 4 * acos(0.0);
    
    int inputPhase{0};
    int lastIdx{0};
    
    bool time_to_refresh = false;
    
    
    argument<int> voices_arg {this, "voices", "Number of voices (batches). Must be the FIRST argument.", true};
    argument<int> control_dim_arg {this, "control_dim", "Dimensionality of the input space.", true};
    argument<int> latent_dim_arg {this, "latent_dim", "Dimensionality of the autoencoder's latent space.", true};
    argument<float> t_gauss_scale_arg {this, "gauss_scale", "(Optional) Gaussian scale of the neural network. A float value between 0-1 is suggested, by default is 0.1. A higher Gaussian scale leads to a noisy terrain. If the Gaussian scale is 0, the Fourier feature mapping layer will be removed, will result in a smooth (low frequency) terrain."};
    argument<int> t_cmax_arg {this, "network_channel", "(Optional) Number of channels in the neural network's fully connected layers, by default is 128"};
    argument<int> mapping_size_arg {this, "feature_size", "(Optional) Size of the random Fourier feature mapping, higher feature size is suggested when using high dimensional control space, by default is 256."};
    argument<int> buffer_arg {this, "buffer_size", "(Optional) Size of the internal buffer (the same as decoder's compression ratio is suggested)."};
    
    argument<symbol> model_path_arg {this, "checkpoint_path", "If only one argument is given, it should be the path to a model checkpoint."};
    
    attribute<bool> enable_cppn {this, "enable_terrain", false, title{"Enable Terrain"},
        description{"Enable / disable cppn computation"},
        setter{
            [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
                if (cppn_init && args[0]){
                    return {true};
                } else {
                    return {false};
                }
                return args;
            }
        }
    };
    
    message<> m_sample_interval {this, "plot_interval", "Sample the terrain across a closed interval to plot into the GUI, only 2D interval dimension is supported at the moment.   <br /><br />Args:  <br />1D [not implemented]: lo, hi, resolution;  <br />2D: values at left, top, right, bottom, x_resolution, y_resolution, latent_channel (optional, default 0)",
        MIN_FUNCTION {
            //args:
            // x_lo, x_hi, y_lo, y_hi, stride, latent_clamp_min, latent_clamp_max
            // lo, hi, stride, latent_clamp_min, latent_clamp_max
            if (args.size() == 3) {
                cerr << "plot_interval: 1D sampling not implemented" << endl;
                return {};
            } else if (args.size() == 6) {
                sample_interval(args[0], args[2], args[4], args[1], args[3], args[5], 0);
                return {};
            } else if (args.size() == 7) {
                sample_interval(args[0], args[2], args[4], args[1], args[3], args[5], args[6]);
                return {};
            }
            cerr << "plot_interval: wrong number of arguments" << endl;
            return {};
        }
    };

    attribute<int,threadsafe::undefined,limit::clamp> plot_resolution {this, "plot_resolution", 1, title{"Plot Resolution"}, description {"(int) stride 1~16"},range{{1, 16}}};
    attribute<bool> plot_colour {this, "plot_multi_channel", false, title{"Plot Multi-Channel"}, description {"(bool) Plot multiple latent channels at once (only support the adjacent 4 channels at the moment)"}};
    
    attribute<number,threadsafe::undefined,limit::clamp> lr{ this, "training_lr", 0.001, title{"Training Learning Rate"}, description{"learning rate when training the nn"}, range{{0.0001, 0.1}}, category{"Training"}};
    attribute<int,threadsafe::undefined,limit::clamp> batch_size{ this, "training_batchsize", 32, title{"Training Batch Size"}, description{"batch size when training the nn"}, range{{4, 128}}, category{"Training"}};
    attribute<int,threadsafe::undefined,limit::clamp> worker{ this, "training_worker", 2, title{"Training Workers"},  description{"number of worker in dataloader"}, range{{0, 4}}, category{"Training"}};

    attribute<int> epoch_per_cycle {this, "training_epoch", 10, title{"Training Epoch"}, description {"(int) train the model for this many epochs when one train message is received"}, category{"Training"}};
    
    message<> maxclass_setup{
        this, "maxclass_setup", [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
            cout << "mcs.nn.terrain~ version: 1.5.6.2 Jul-2026 - torch version: " << TORCH_VERSION << endl;
            // register multichannel handlers
            c74::max::t_class *c = args[0];
            c74::max::class_addmethod(c, (c74::max::method)simplemc_multichanneloutputs,
                                      "multichanneloutputs", c74::max::A_CANT, 0);
            c74::max::class_addmethod(c, (c74::max::method)simplemc_inputchanged,
                                      "inputchanged", c74::max::A_CANT, 0);
            return {};
        }
    };
    
    message<> dspsetup {this, "dspsetup",
        MIN_FUNCTION {
           m_one_over_samplerate = 1.0 / samplerate();
           return {};
       }
    };
    
    at::Tensor training_coords;
    at::Tensor training_latents;
    
    min_dict training_data = {symbol(true)};
    min_dict coord_dict = {symbol(true)};
    min_dict latent_dict = {symbol(true)};
    int coord_len = 0;
    int latent_len = 0;
    
    bool dataset_updated = true;
    bool dataloader_ready = false;
    
    message<> dictionary { this, "dictionary",
        "Use a dictionary to gather training data for the terrain (will replace what has been loaded before). ",
        MIN_FUNCTION {
            c74::max::t_symbol    **keys = NULL;
            long        numkeys = 0;
            
            min_dict d = {args[0]};
            c74::max::dictionary_getkeys(d.m_instance, &numkeys, &keys);
            
            for (int i = 0; i < numkeys; i++){
                string key_str = std::string(keys[i]->s_name);
                atom d_data = d[key_str];
                min_dict d_data_dict = {d_data};
                atom count = c74::max::dictionary_getentrycount(d_data_dict.m_instance);
                
                if (key_str == "coordinates"){
                    coord_len = count;
                    if (coord_len == 0) {
                        coord_dict.clear();
                    } else {
                        coord_dict = d_data_dict;
                        cout << coord_len << " spatial trajectories loaded." << endl;
                    }
                    dataset_updated = true;
                } else if (key_str == "latents"){
                    latent_len = count;
                    if (latent_len==0){
                        latent_dict.clear();
                    } else {
                        latent_dict = d_data_dict;
                        cout << latent_len << " latent trajectories loaded." << endl;
                    }
                    dataset_updated = true;
                } else if (key_str == "anchors"){
                    
                } else {
                    cerr << "unknown key, or some dictionaries are empty: " << key_str << endl;
                }
            }
            // freeing memory for keys
            if (keys){
                c74::max::dictionary_freekeys(d.m_instance, numkeys, keys);
            }
            training_data.touch();
            m_outlets[m_outlets.size()-1]->send("dataset","dictionary", training_data.name());
            return {};
        }
    };
    

//    std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<LatentDataset, torch::data::transforms::Stack<torch::data::Example<>>>, torch::data::samplers::RandomSampler>> data_loader;
    
//    std::unique_ptr<torch::optim::Adam> optimizer;
    
    message<> model_summary {this, "model_summary", "Model summary",
        MIN_FUNCTION {
            min_dict net_args_dict {symbol(true)};
            
            net_args_dict["input_dimension"] = m_in_dim;
            net_args_dict["output_dimension"] = m_out_dim;
            net_args_dict["gaussian_scale"] = m_gauss_scale;
            net_args_dict["number_of_channels"] = m_cmax;
            net_args_dict["features_size"] = m_mapping_size;
            net_args_dict["number_of_parameters"] = static_cast<int>(cppn_model->m_model->parameters().size());
            
            m_outlets[m_outlets.size()-1]->send("summary", "dictionary", net_args_dict.name());
            return {};
        }
    };
    
    min_dict dataset_info_dict {symbol(true)};
    min_dict dataset_co_dict {symbol(true)};
    min_dict dataset_lt_dict {symbol(true)};
    atoms dataset_summary;
    message<> dataset_summary_m {this, "dataset_summary", "Coordinates-latents pairs summary",
        MIN_FUNCTION {
            try{
                if (dataset_updated){
                    dataset_summary = create_dataloader();
                }
                if (dataset_summary[0] == 0){
                    cwarn << "no data loaded" << endl;
                    return {};
                }
                symbol shape_key{"shape"};
                symbol range_key{"range"};
                
                dataset_info_dict["dataset_len"] = static_cast<int>(dataset_summary[0]);
                
                atoms shape_co = {dataset_summary[3], dataset_summary[4]};
                c74::max::dictionary_appendatoms(dataset_co_dict.m_instance, shape_key, shape_co.size(), &shape_co[0]);
                atoms range_co = {dataset_summary[5], dataset_summary[6]};
                c74::max::dictionary_appendatoms(dataset_co_dict.m_instance, range_key, range_co.size(), &range_co[0]);
                
                atoms shape_lt = {dataset_summary[7], dataset_summary[8]};
                c74::max::dictionary_appendatoms(dataset_lt_dict.m_instance, shape_key, shape_lt.size(), &shape_lt[0]);
                atoms range_lt = {dataset_summary[9], dataset_summary[10]};
                c74::max::dictionary_appendatoms(dataset_lt_dict.m_instance, range_key, range_lt.size(), &range_lt[0]);
                
                dataset_info_dict.touch();
                m_outlets[m_outlets.size()-1]->send("summary", "dictionary", dataset_info_dict.name());
            } catch (const std::exception &e) {
                cerr << e.what() << endl;
            }
            return {};
        }
    };
    
    attribute<bool> cppn_gpu{this, "gpu", false,
        description{"Enable / disable gpu usage when available"},
        setter{
            [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
                if (cppn_init){
                    if (cppn_model->is_loaded()){
                        cppn_model->use_gpu(bool(args[0]));
//                        optimizer = std::make_unique<torch::optim::Adam>(cppn_model->m_model->parameters(), static_cast<float>(lr));
                        cppn_model->init_optimizer(static_cast<float>(lr));
                        
//                        if (bool(args[0])){
//                            cwarn << "Terrain is a very small neural network, it's performance on CPU can be better in some cases" << endl;
//                        }
                    }
                }
                return args;
            }
        }
    };
    
    message<> new_cppn {this, "new", "Create a new terrain model.   <br /><br />Args:  <br />path (optional): path to a model checkpoint, if not given, a new model will be created;  <br />gaussian_scale: Gaussian scale of the neural network;  <br />network_channel: number of channels in the neural network's fully connected layers;  <br />feature_size: size of the random Fourier feature mapping.",
        MIN_FUNCTION {
            //args:
            // x_lo, x_hi, y_lo, y_hi, stride, latent_clamp_min, latent_clamp_max
            // lo, hi, stride, latent_clamp_min, latent_clamp_max
            torch::serialize::InputArchive archive;
            
            if (args.size() != 3 && args.size() != 1) {
                cerr << "wrong number of argments" << endl;
                return {};
            }
            if (args.size() == 1) {
                if (args[0].a_type == 3){
                    std::string model_path = std::string(args[0]);
                    if (!load_param_from_file(model_path, archive)) {
                        cerr << "error loading model from file: " << model_path << endl;
                        return {};
                    }
                } else {
                    cerr << "argument should be a path" << endl;
                    return {};
                }
            } else if (args.size() == 3) {
                m_cmax = int(args[1]);
                m_gauss_scale = float(args[0]);
                m_mapping_size = int(args[2]);
            }
            cppn_model.reset();
            cppn_model = std::make_unique<FCPPN>();
            
            if (!cppn_model->create(m_in_dim, m_out_dim, m_cmax, m_gauss_scale, m_mapping_size)) {
                cerr << "error during creating model" << endl;
                error();
                return {};
            }
            
            if (args.size() == 1) { // if loading from a checkpoint
                try{
                    cppn_model->m_model->load(archive);
                } catch (const std::exception& e) {
                    cerr << "Error loading weights: " << e.what() << endl;
                    return {};
                }
            }
            
            cppn_model->m_model->eval();
            
            torch::Tensor test_inputs = torch::ones({1, m_in_dim}, torch::kFloat);
            
            auto output_tensor = cppn_model->m_model->forward(test_inputs);
            
//            optimizer.reset();
//            optimizer = std::make_unique<torch::optim::Adam>(cppn_model->m_model->parameters(), static_cast<float>(lr));
            cppn_model->init_optimizer(static_cast<float>(lr));
            cout << "terrain setup finished" << endl;
            return {};
        }
    };
    
    
    int total_epochs = 0;
    float loss_log = 0.0f;
    message<> train_cycle {this, "train", "Train the terrain model with one batch",
        MIN_FUNCTION {
            if (!cppn_model->optimizer_init){
                cerr << "optimizer not initialized" << endl;
                return {};
            }
            if (dataset_updated){
                try{
                    dataset_summary = create_dataloader();
                    if (dataset_summary[0]==0){
                        cerr << "no data loaded" << endl;
                        return {};
                    }
                } catch (const std::exception &e) {
                    cerr << e.what() << endl;
                    return {};
                }
            }
            m_should_train_lock.release();
            return {};
        }
    };
    
    timer<timer_options::defer_delivery> m_train_timer { this,
        MIN_FUNCTION {
            if (m_finish_train_lock.try_acquire()) {
                m_outlets[m_outlets.size()-1]->send("loss", loss_log);
                m_outlets[m_outlets.size()-1]->send("epoch", total_epochs);
            }
            if (m_finish_plot_lock.try_acquire()) {
                atom count = c74::max::dictionary_getentrycount(terrain_dict.m_instance);
                if (count == 0){
                    m_train_timer.delay(100);
                    return {};
                }
                sampled_dict.touch();
                m_outlets[m_outlets.size()-2]->send("dictionary", sampled_dict.name());
            }
            m_train_timer.delay(100);
            return {};
        }
    };
    
    message<> train_cycle_thread {this, "train_test", "Train the terrain model with one batch",
        MIN_FUNCTION {
            m_should_train_lock.release();
            return {};
        }
    };
    
    attribute<symbol> external_path{ this, "saving_path", "none", title{"Saving Path"}, category{"Saving"}, setter{ MIN_FUNCTION{
        if (args[0] == "none") {
            symbol min_path_this = min_devkit_path();
            return { min_path_this };
        } else {
            try {
                // Don't delete this:
                min_path min_path_this = min_path(static_cast<string>(args[0]));
                return { args[0] };
            } catch (...) {
                symbol min_path_this = min_devkit_path();
                cwarn << "Error reading folder, set to default" << endl;
                return { min_path_this };
            }
        }
        } }
    };
    
    attribute<symbol> saving_name{ this, "saving_name", "UntitledTerrain", title{"Saving Name"}, category{"Saving"}, setter{ MIN_FUNCTION{
        std::string str = std::string(args[0]);
        str.erase(std::remove(str.begin(), str.end(), ' '), str.end()); // remove spaces
        if (str.empty()) {
            return { "UntitledTerrain" };
        } else {
            return { args[0] };
        }
        } }
    };
    
    message<> save_cppn {this, "checkpoint", "Checkpoint and save the terrain model",
        MIN_FUNCTION {
            if (!cppn_init){
                return {};
            }
            std::string saved_name = std::to_string(saving_name);
            if (saved_name.substr(saved_name.length() - 3) != ".pt"){
                saved_name = saved_name + ".pt";
            }
            try {
                if(cppn_model->save(std::to_string(external_path), saved_name, m_in_dim, m_out_dim, m_cmax, m_gauss_scale, m_mapping_size)){
                    cout << saved_name << " saved to: " << std::to_string(external_path) << endl;
                }
            } catch (const std::exception &e) {
                cerr << "error saving the model:" << e.what() << endl;
            }
            return {};
        }
    };
    
    
private:
    double m_one_over_samplerate    { 1.0 };
};

atoms mcs_nn_terrain::create_dataloader() {
    int coord_count = static_cast<int>(c74::max::dictionary_getentrycount(coord_dict.m_instance));
    int latent_count = static_cast<int>(c74::max::dictionary_getentrycount(latent_dict.m_instance));
    
    if (coord_count==0 || latent_count==0){
        cwarn << "coordinates or latents not set" << endl;
        return {{0}};
    }
    
    int dataset_len = 0;
    int traj_len = 0;
    
    if (latent_count > coord_count){
        cwarn << "latent trejectories "<<latent_count<<" > audio trejectories "<<coord_count<<", using all audio with " << latent_count - coord_count << " latent trejectories remained" << endl;
        traj_len = coord_count;
    } else if (latent_count < coord_count){
        cwarn << "audio trejectories "<<coord_count<<" > latent trejectories"<<latent_count<<", using all latent trejectories with " << coord_count - latent_count << "audio trejectories remained" << endl;
        traj_len = latent_count;
    } else {
        traj_len = latent_count;
    }
    
    std::vector<at::Tensor> coord_trim;
    std::vector<at::Tensor> latent_trim;
    
    for (int i(0); i < traj_len; i++){
        
        std::vector<float> coords;
        std::vector<float> latents;
        
        int point_count = 0;
        atom c_data = coord_dict[std::to_string(i+1)];
        min_dict c_seg_dict = {c_data};
        atoms seg_c_atoms_tp = c_seg_dict["0"];
        
        atom l_data = latent_dict[std::to_string(i+1)];
        min_dict l_seg_dict = {l_data};
        atoms seg_l_atoms_tp = l_seg_dict["0"];
        
        point_count = seg_c_atoms_tp.size() > seg_l_atoms_tp.size() ? seg_l_atoms_tp.size() : seg_c_atoms_tp.size();
        
        int coord_dim = static_cast<int>(c74::max::dictionary_getentrycount(c_seg_dict.m_instance));
        int latent_dim_here = static_cast<int>(c74::max::dictionary_getentrycount(l_seg_dict.m_instance));
        
        if (m_out_dim != latent_dim_here){
            cerr << "latents dimension " << latent_dim_here << " doesn't match latent_dim" << endl;
            return {{0}};
        }
        for (int k(0); k < m_in_dim; k++){
            atoms seg_c_atoms;
            try {
                seg_c_atoms = c_seg_dict[std::to_string(k)];
            } catch (const std::exception &e) {
                cerr << "error getting coordinates from dictionary: " << e.what() << endl;
                return {{0}};
            }
            for (int j(0); j < point_count; j++){
                coords.push_back(static_cast<float>(seg_c_atoms[j]));
            }
        }
        for (int k(0); k < m_out_dim; k++){
            atoms seg_l_atoms;
            try {
                seg_l_atoms = l_seg_dict[std::to_string(k)];
            } catch (const std::exception &e) {
                cerr << "error getting latents from dictionary: " << e.what() << endl;
                return {{0}};
            }
            for (int j(0); j < point_count; j++){
                latents.push_back(static_cast<float>(seg_l_atoms[j]));
            }
        }
        coord_trim.push_back(torch::from_blob(coords.data(), {m_in_dim, point_count}, torch::kFloat).clone());
        latent_trim.push_back(torch::from_blob(latents.data(), {m_out_dim, point_count}, torch::kFloat).clone());
        dataset_len += point_count;
    }

    
    if (coord_trim.size() == 0 || latent_trim.size() == 0){
        cerr << "error converting to tensor" << endl;
        return {{0}};
    }
    training_coords = torch::cat(coord_trim, 1).permute({1,0});
    training_latents = torch::cat(latent_trim, 1).permute({1,0});
    
    std::vector<int> traj_info = {
        static_cast<int>(training_coords.size(0)),
        static_cast<int>(training_coords.size(1))
    };
    std::vector<float> traj_range_info = {
        training_coords.min().item<float>(),
        training_coords.max().item<float>()
    };
    std::vector<int> lat_info = {
        static_cast<int>(training_latents.size(0)),
        static_cast<int>(training_latents.size(1))
    };
    std::vector<float> lat_range_info = {
        training_latents.min().item<float>(),
        training_latents.max().item<float>()
    };
    
    auto dataset = LatentDataset(training_coords, training_latents).map(torch::data::transforms::Stack<>());
    m_outlets[m_outlets.size()-1]->send("dataset_length", static_cast<float>(dataset.size().value()));
    cppn_model->data_loader = torch::data::make_data_loader(std::move(dataset),
                                                            torch::data::DataLoaderOptions().batch_size(batch_size).workers(worker).drop_last(false));
    dataloader_ready = true;
    dataset_updated = false;
    
    return {{
        dataset_len,
        coord_count,
        latent_count,
        traj_info[0], // trajs.size(0)
        traj_info[1], // trajs.size(1)
        traj_range_info[0], // traj_range_info.min()
        traj_range_info[1], // traj_range_info.max()
        lat_info[0],  // latents.size(0)
        lat_info[1],  // latents.size(1)
        lat_range_info[0],  // lat_range_info.min()
        lat_range_info[1]   // lat_range_info.max()
    }};
}

void mcs_nn_terrain::model_train_loop(mcs_nn_terrain *nn_instance) {
  while (!nn_instance->m_should_stop_perform_thread) {
      if (nn_instance->m_should_train_lock.try_acquire_for(std::chrono::milliseconds(100))) {
          try {
              float loss_log = nn_instance->cppn_model->train_for(nn_instance->epoch_per_cycle);
              nn_instance->loss_log = loss_log;
              nn_instance->total_epochs += nn_instance->epoch_per_cycle;
              nn_instance->m_finish_train_lock.release();
          } catch (const std::exception &e) {
              nn_instance->cerr << e.what() << endl;
          }
      }
  }
}
void mcs_nn_terrain::model_plot_loop(mcs_nn_terrain *nn_instance) {
  while (!nn_instance->m_should_stop_perform_thread) {
      if (nn_instance->m_should_plot_lock.try_acquire_for(std::chrono::milliseconds(100))) {
          nn_instance->terrain_dict.clear();
          std::unique_lock<std::mutex> model_lock(nn_instance->cppn_model->m_model_mutex);
          c10::InferenceMode guard; // plotting only needs forward; skip autograd
          int c = nn_instance->c_prev;
          bool multi_channel = c + 4 < nn_instance->m_out_dim ? nn_instance->plot_colour : false;
          try {
              if (multi_channel){
                  for (int i(0); i < nn_instance->cppn_model->sample_tensor.size(0); i++){ // y
                      min_dict row_dict = {};
                      at::Tensor tensor_in = nn_instance->cppn_model->sample_tensor.index({i});
                      at::Tensor tensor_out = nn_instance->cppn_model->m_model->forward(tensor_in).to(torch::kFloat);
                      
                      for (int j(0); j < 4; j++){
                          
                          at::Tensor a_out = tensor_out.index({Slice(None), c+j}).to(torch::kCPU);
                          
                          auto ten_out_ptr = a_out.contiguous().data_ptr<float>();
                          atoms result(ten_out_ptr, ten_out_ptr + nn_instance->cppn_model->sample_tensor.size(1));
                          
                          symbol skey{j};
                          c74::max::dictionary_appendatoms(row_dict.m_instance, skey, result.size(), &result[0]);
                      }
                      nn_instance->terrain_dict[std::to_string(i)] = row_dict;
                  }
              } else {
                  for (int i(0); i < nn_instance->cppn_model->sample_tensor.size(0); i++){ // y
                      at::Tensor tensor_in = nn_instance->cppn_model->sample_tensor.index({i});
                      at::Tensor tensor_out = nn_instance->cppn_model->m_model->forward(tensor_in).to(torch::kFloat);
                      tensor_out = tensor_out.index({Slice(None), c}).to(torch::kCPU);
                      auto ten_out_ptr = tensor_out.contiguous().data_ptr<float>();
                      atoms result(ten_out_ptr, ten_out_ptr + nn_instance->cppn_model->sample_tensor.size(1));
                      
                      symbol skey{i};
                      c74::max::dictionary_appendatoms(nn_instance->terrain_dict.m_instance, skey, result.size(), &result[0]);
                  }
              }
              nn_instance->m_finish_plot_lock.release();
          } catch (const std::exception &e) {
              nn_instance->cerr << e.what() << endl;
          }
          model_lock.unlock();
      }
  }
}


void fill_with_zero(audio_bundle output) {
    for (int c(0); c < output.channel_count(); c++) {
        auto out = output.samples(c);
        for (int i(0); i < output.frame_count(); i++) {
            out[i] = 0.;
        }
    }
}

void mcs_nn_terrain::operator()(audio_bundle input, audio_bundle output) {
//  CHECK IF MODEL IS LOADED AND ENABLED
    if (!cppn_init || !enable_cppn) {
        fill_with_zero(output);
        return;
    }
    perform(input, output);
}

void mcs_nn_terrain::sample_interval(float x_lo, float x_hi, int x_res, float y_lo, float y_hi, int y_res, int c){
    if (x_lo == x_lo_prev && x_hi == x_hi_prev && y_lo == y_lo_prev && y_hi == y_hi_prev && y_res == y_res_prev && x_res == x_res_prev && plot_resolution == plot_resolution_prev){
        // do nothing
    } else {
        x_lo_prev = x_lo;
        x_hi_prev = x_hi;
        y_lo_prev = y_lo;
        y_hi_prev = y_hi;
        x_res_prev = x_res;
        y_res_prev = y_res;
        plot_resolution_prev = plot_resolution;
        create_sample_tensor();
    }
    if (c < 0 || c >= m_out_dim){
        cerr << "latent channel out of range" << endl;
        return;
    }
    c_prev = c;
    if (cppn_model->sample_tensor.numel() == 0){
        return;
    }
    m_should_plot_lock.release();
}

void mcs_nn_terrain::create_sample_tensor(){
    vector<float> tensor_in_data;
    
    float x_stride = (x_hi_prev-x_lo_prev)/x_res_prev;
    float y_stride = (y_hi_prev-y_lo_prev)/y_res_prev;
    
    for (int i(0); i < y_res_prev; i+=plot_resolution){
        for (int j(0); j < x_res_prev; j+=plot_resolution){
            tensor_in_data.push_back(x_lo_prev + j * x_stride);
            tensor_in_data.push_back(y_lo_prev + i * y_stride);
        }
    }
    int canvas_width = static_cast<int>(std::ceil(x_res_prev / static_cast<float>(plot_resolution)));
    int canvas_height = static_cast<int>(std::ceil(y_res_prev / static_cast<float>(plot_resolution)));
    int canvas_size = canvas_width * canvas_height * 2;
    
    if (tensor_in_data.size() != canvas_size){
        cerr << "tensor_in_data size mismatch: " << tensor_in_data.size() << ", " << canvas_size << endl;
        return;
    }
    cppn_model->sample_tensor = torch::from_blob(tensor_in_data.data(), {canvas_height, canvas_width, 2}, torch::kFloat).clone().to(cppn_model->m_device);
//    cout << "sampled tensor created: " << cppn_model->sample_tensor.sizes() << endl;
}

void mcs_nn_terrain::perform(audio_bundle input, audio_bundle output) {
    auto vec_size = input.frame_count();// 128
    int batch = m_batch;
  // COUNT INPUT SAMPLES UNTIL A FULL BUFFER WORTH HAS BEEN COLLECTED
    m_in_count += static_cast<int>(vec_size);

    if (m_in_count >= m_buffer_size) { // BUFFER IS FULL // 2048
        m_in_count -= m_buffer_size;   // vec_size divides m_buffer_size, so this stays aligned

        // (A) gather one control point per voice (first sample of the current block).
        // inlet b occupies channels [b*m_in_dim .. b*m_in_dim + m_in_dim).
        for (int b(0); b < batch; b++){
            for (int d(0); d < m_in_dim; d++){
                m_in_point[b * m_in_dim + d] = static_cast<float>(*input.samples(b * m_in_dim + d));
            }
        }

        cppn_infer(m_in_point.data(), m_out_flat.data());

        // (B) scatter held latent values. output channel for (voice b, dim d) is b*m_out_dim + d,
        // which matches the row-major order of m_out_flat exactly.
        for (int i(0); i < batch * m_out_dim; i++){
            float val = m_out_flat[i];
            std::fill(m_latent_out[i].get(), m_latent_out[i].get() + m_buffer_size, val);
            m_out_buffer[i].put(m_latent_out[i].get(), m_buffer_size);
        }
    }

    // COPY CIRCULAR BUFFER TO OUTPUT (buffers are indexed by output audio channel)
    for (int c(0); c < output.channel_count(); c++) {
        auto out = output.samples(c);
        if (enable_cppn){
            m_out_buffer[c].get(out, vec_size);
        }
    }
}

void mcs_nn_terrain::cppn_infer(float* float_in, float* out_flat){
    if (!cppn_init){
        return;
    }

    // NON-BLOCKING: never stall the audio thread. If training currently holds the
    // model, bail out and let the caller keep streaming the previous latent.
    std::unique_lock<std::mutex> model_lock(cppn_model->m_model_mutex, std::try_to_lock);
    if (!model_lock.owns_lock()){
        return;
    }

    c10::InferenceMode guard; // no autograd graph is built on the audio thread

    at::Tensor tensor_out;
    try {
        // batched forward: {batch, m_in_dim} -> {batch, m_out_dim}
        at::Tensor tensor_in = torch::from_blob(float_in, {m_batch, m_in_dim}, torch::kFloat)
                                   .to(cppn_model->m_device);
        tensor_out = cppn_model->m_model->forward(tensor_in);
        tensor_out = tensor_out.clamp_min({-100.0f}).clamp_max({100.0f});
        // each output channel is constant across the buffer, so only ship the
        // batch * m_out_dim scalars back to the host (cheap, GPU/MPS-friendly).
        tensor_out = tensor_out.to(torch::kCPU).contiguous(); // -> {batch, m_out_dim}
    } catch (const std::exception &e) {
        cout << e.what() << endl;
        return;
    }
    model_lock.unlock();

    auto out_ptr = tensor_out.data_ptr<float>();
    int n = m_batch * m_out_dim;            // row-major [batch][m_out_dim]
    std::copy(out_ptr, out_ptr + n, out_flat);
}


bool mcs_nn_terrain::load_param_from_file(string model_path, torch::serialize::InputArchive &archive) {
    if (model_path.substr(model_path.length() - 3) != ".pt")
        model_path = model_path + ".pt";
    min_path m_path = min_path(model_path);
    
    try {
        // Load the archive from the specified file path
        archive.load_from(std::string(m_path));
        
        // Load initialization arguments
        torch::Tensor in_dim_tensor, out_dim_tensor, c_max_tensor, gauss_scale_tensor, mapping_size_tensor;
        archive.read("m_in_dim", in_dim_tensor);
        archive.read("m_out_dim", out_dim_tensor);
        archive.read("m_cmax", c_max_tensor);
        archive.read("m_gauss_scale", gauss_scale_tensor);
        archive.read("m_mapping_size", mapping_size_tensor);
        
        m_in_dim = in_dim_tensor.item<int>();
        m_out_dim = out_dim_tensor.item<int>();
        m_cmax = c_max_tensor.item<int>();
        m_gauss_scale = gauss_scale_tensor.item<float>();
        m_mapping_size = mapping_size_tensor.item<int>();
    } catch (const std::exception& e) {
        cerr << "Error loading from checkpoint: " << e.what() << endl;
        return false;
    }
    return true;
}

mcs_nn_terrain::mcs_nn_terrain(const atoms &args){
    
    // arguments (voices is ALWAYS first for mcs):
    // (voices, path)
    // (voices, in_dim, out_dim)
    // (voices, in_dim, out_dim, gauss_scale)
    // (voices, in_dim, out_dim, gauss_scale, c_max)
    // (voices, in_dim, out_dim, gauss_scale, c_max, mapping_size)
    // (voices, in_dim, out_dim, gauss_scale, c_max, mapping_size, buffer_size)
    torch::serialize::InputArchive archive;

    // CHECK ARGUMENTS
    if (!args.size()) {
        return;
    }
    // FIRST ARGUMENT = number of voices (batches)
    if (args[0].a_type != 1) {
        cerr << "first argument (voices) should be an integer" << endl;
        return;
    }
    m_voices = int(args[0]);
    if (m_voices < 1 || m_voices > 32) {
        cerr << "voices should be a positive integer <= 32" << endl;
        return;
    }
    m_batch = m_voices;

    // remaining args follow the standard nn.terrain~ scheme
    atoms rest(args.begin() + 1, args.end());

    if (!rest.size()) {
        // only voices given: no model spec -> object stays disabled (cppn_init = false)
        return;
    }
    if (rest.size() == 1) { // ONE ARGUMENT IS GIVEN
        if (rest[0].a_type == 3){
            std::string model_path = std::string(rest[0]);
            if (!load_param_from_file(model_path, archive)) {
                cerr << "error loading model from file: " << model_path << endl;
                return;
            }
        } else {
            cerr << "argument should be a path" << endl;
            return;
        }
    }
    if (rest.size() >= 2) { // TWO ARGUMENTS ARE GIVEN
        if (rest[0].a_type == 1 && rest[1].a_type == 1) {
            if (int(rest[0]) > 0 && int(rest[1]) > 0 && int(rest[0]) <= 6 && int(rest[1]) <= 1024) {
                m_in_dim = int(rest[0]);
                m_out_dim = int(rest[1]);
            } else {
                cerr << "control_dim should be positive integers <= 6, latent_dim should be positive integers <= 1024" << endl;
                return;
            }
        } else {
            cerr << "first and second args should be integers" << endl;
            return;
        }
    }
    if (rest.size() >= 3) { // THREE ARGUMENTS ARE GIVEN
        if (rest[2].a_type == 2) {
            if (float(rest[2]) >= 0 && float(rest[2]) <= 10) {
                m_gauss_scale = float(rest[2]);
            } else {
                cerr << "third arg should be non-negative float <= 10" << endl;
                return;
            }
        } else {
            cerr << "third arg should be a float" << endl;
            return;
        }
    }
    if (rest.size() >= 4) { // Four ARGUMENTS ARE GIVEN
        if (rest[3].a_type == 1) {
            if (int(rest[3]) <= 1024) {
                m_cmax = int(rest[3]);
            } else {
                cerr << "fourth arg should be positive integers <= 1024" << endl;
                return;
            }
        } else {
            cerr << "fourth arg be an integer" << endl;
            return;
        }
    }
    if (rest.size() >= 5) { // Five ARGUMENTS ARE GIVEN
        if (rest[4].a_type == 1) {
            if (int(rest[4]) <= 2048 && int(rest[4]) >= 16) {
                m_mapping_size = int(rest[4]);
            } else {
                cerr << "fifth arg should be positive integers between [16, 2048]" << endl;
                return;
            }
        } else {
            cerr << "fifth arg be an integer" << endl;
            return;
        }
    }
    if (rest.size() >= 6) {
        m_buffer_size = power_ceil(int(rest[5]));
    }
    if (rest.size() >= 7) {
        cerr << "too many arguments" << endl;
        return;
    }

    cppn_model = std::make_unique<FCPPN>();

    if (!cppn_model->create(m_in_dim, m_out_dim, m_cmax, m_gauss_scale, m_mapping_size)) {
        cerr << "error during creating model" << endl;
        error();
        return;
    }

    if (rest.size() == 1) { // if loading from a checkpoint
        try{
            cppn_model->m_model->load(archive);
        } catch (const std::exception& e) {
            cerr << "Error loading weights: " << e.what() << endl;
            return;
        }
    }
    
    // TODO: do this:
    cppn_model->m_model->eval();
    
    torch::Tensor test_inputs = torch::ones({1, m_in_dim}, torch::kFloat);
    
    
    auto output_tensor = cppn_model->m_model->forward(test_inputs);
    
    cppn_model->init_optimizer(static_cast<float>(lr));
    cppn_init = true;

  m_use_thread = false;

  // CREATE INLETS, OUTLETS and BUFFERS (multichannel: batch is on the inlets)
  m_batch = m_voices;
  // one multichannelsignal inlet per voice; channels within = m_in_dim (control point)
  for (int b(0); b < m_voices; b++) {
    std::string input_label = "(multichannelsignal) voice " + std::to_string(b) + " control point (channels = control_dim)";
    m_inlets.push_back(std::make_unique<inlet<>>(this, input_label, "multichannelsignal"));
  }
  m_in_point.assign(m_batch * m_in_dim, 0.0f);
  m_out_flat.assign(m_batch * m_out_dim, 0.0f);

  // one multichannelsignal outlet per voice; channels within = m_out_dim (latent vector)
  for (int b(0); b < m_voices; b++) {
      std::string output_label = "(multichannelsignal) voice " + std::to_string(b) + " latent (channels = latent_dim)";
      m_outlets.push_back(std::make_unique<outlet<>>(this, output_label, "multichannelsignal"));
  }
  m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(m_batch * m_out_dim);
  for (int i(0); i < m_batch * m_out_dim; i++) {
      m_out_buffer[i].initialize(m_buffer_size);
      m_latent_out.push_back(std::make_unique<float[]>(m_buffer_size));
  }
    m_outlets.push_back(std::make_unique<outlet<>>(this, "(dictionary) plotted interval", "dictionary"));
    m_outlets.push_back(std::make_unique<outlet<>>(this, "(message) logging information, route option: summary, dataset, dataset_length, epoch, loss", "message"));
    

    external_path = min_devkit_path();
//    cout << "path " << external_path << endl;
    
    create_sample_tensor();
    
    training_data["coordinates"] = coord_dict;
    training_data["latents"] = latent_dict;
    
    sampled_dict["terrain"] = terrain_dict;
    
    dataset_info_dict["coordinates"] = dataset_co_dict;
    dataset_info_dict["latents"] = dataset_lt_dict;
    
    m_train_thread = std::make_unique<std::thread>(model_train_loop, this);
    m_compute_thread = std::make_unique<std::thread>(model_plot_loop, this);
    
    m_train_timer.delay(200);
    
//    cout << "terrain setup finished" << endl;
}

mcs_nn_terrain::~mcs_nn_terrain() {
    
    terrain_dict.clear();
    latent_dict.clear();
    coord_dict.clear();
    
    m_should_stop_perform_thread = true;
    if (m_train_thread){
        m_train_thread->join();
    }
    if (m_compute_thread){
        m_compute_thread->join();
    }
    m_train_timer.stop();


}

long simplemc_multichanneloutputs(c74::max::t_object *x, long index, long count) {
    minwrap<mcs_nn_terrain> *ob = (minwrap<mcs_nn_terrain> *)(x);
    // each signal outlet (one per voice) carries m_out_dim channels
    return index < ob->m_min_object.m_voices ? ob->m_min_object.m_out_dim : 1;
}

long simplemc_inputchanged(c74::max::t_object *x, long index, long count) {
    minwrap<mcs_nn_terrain> *ob = (minwrap<mcs_nn_terrain> *)(x);
    auto chan_number = ob->m_min_object.m_in_dim;
    // each voice inlet must carry exactly m_in_dim channels (batch is fixed)
    if (chan_number != count) {
        c74::max::object_error(
            x, (std::string("invalid channel number for input ") +
                std::to_string(index) + std::string("; should be ") +
                std::to_string(chan_number))
                   .c_str());
    }
    return false;
}

MIN_EXTERNAL(mcs_nn_terrain);
