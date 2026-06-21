/// Author: Jasper Shuoyang Zheng
/// Modified from the multitouch example in Min-DevKit.

#include "c74_min.h"

#include <stdio.h>
#include <iostream>

#include <thread>
#include <semaphore>
#include <mutex>
#include <atomic>
#include <array>

#ifdef MAC_VERSION
#include <ApplicationServices/ApplicationServices.h>
#endif    // MAC_VERSION

#include "pen.h"
#include "terrain.h"
#include "../shared/min_path.h"
#include "../shared/min_dictionary.h"
#include "../shared/circular_buffer.h"
#include "../shared/utils.h"

#include <algorithm>
#include <chrono>

#include <fstream>

using namespace c74::min;
using namespace c74::min::ui;

typedef struct event_info_type {
    number X;
    number Y;
    number P;
    symbol phase;
    color color;
    std::chrono::duration<float> ms;
    int modifier{0}; // 0: none, 1: shift, 2: command, 3: else
} event_info;

typedef struct segment_info_type {
    number x_start;
    number y_start;
    number x_end;
    number y_end;
    number distance;
    float ms;
    bool completed;
} segment_info;

typedef struct traj_info_type {
    vector<segment_info *> segments;
    int target_latents;
    int filled_latents;
    float ms;
    bool completed;
} traj_info;

typedef struct click_check_type {
    int traj_idx;
    int seg_idx;
    bool front; // true: start, false: end
    float x;
    float y;
} click_check_info;

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
    CFBundleRef this_bundle = CFBundleGetBundleWithIdentifier(CFSTR("com.jasperzheng.nn-terrain-gui"));
    CFURLRef    this_url = CFBundleCopyExecutableURL(this_bundle);
    char        this_path[4096];
    CFURLGetFileSystemRepresentation(this_url, true, reinterpret_cast<UInt8*>(this_path), 4096);
    string this_path_str{ this_path };
    CFRelease(this_url);
    // we now have a path like this:
    // /Users/tim/Materials/min-devkit/externals/min.project.mxo/Contents/MacOS/min.project"
    // so we need to chop off 5 slashes from the end
    auto iter = this_path_str.find("/externals/nn.terrain.gui.mxo/Contents/MacOS/nn.terrain.gui");
    this_path_str.erase(iter, strlen("/externals/nn.terrain.gui.mxo/Contents/MacOS/nn.terrain.gui"));
    return this_path_str;
#endif    // MAC_VERSION
}

class stylus : public object<stylus>, public ui_operator<600, 150>, public vector_operator<> {
private:
    
    // TERRAIN PLOTTING WORKER (keeps the heavy canvas build off the message thread)
    std::unique_ptr<std::thread> m_render_thread{nullptr};
    bool m_should_stop_render_thread{false};
    std::binary_semaphore m_should_render_lock{0}; // handler -> worker: work pending
    std::binary_semaphore m_finish_render_lock{0}; // worker -> main: canvas ready
    std::atomic<bool> m_render_pending{false};     // guards binary_semaphore over-release

    std::mutex m_canvas_mutex;        // guards terrain_canvas / terrain_canvas_c / terrain_loaded
    std::mutex m_render_input_mutex;  // guards the snapshot handed to the worker
    std::vector<std::vector<float>> m_in_mono;
    std::vector<std::array<std::vector<float>, 4>> m_in_color;
    int m_in_mode{0};                 // 1: mono, 4: color
    float m_in_cmin{0.0f}, m_in_cmax{0.0f};
    color m_in_blo, m_in_bhi;

    static void terrain_render_loop(stylus *st);

    float cursor_x = -21.0f;
    float cursor_y = -21.0f;
    float cursor_p = 0.0f;
    
    vector<traj_info *> m_trajs;
    int traj_ptr = 0;
    int seg_ptr = 0;
    
    vector<traj_info *> m_points;
    int point_ptr = 0;
    
    vector<traj_info *> m_plays;
    int play_traj_ptr = 0;
    int play_seg_ptr = 0;
    int is_adding_new = 0; // 0: no, 1: adding to the front, 2: adding to the end
    bool continue_dragging = false;
    bool drag_once = false;
    
    click_check_info cc;
    
    color playing_selected{0.941f, 0.68f, 0.392f, 1.0f};
    color playing_standby{0.7f, 0.7f, 0.7f, 1.0f};
    
    color dataset_complete{0.345f, 0.753f, 0.824f, 1.0f};
    color dataset_selected{0.95f, 0.95f, 0.95f, 1.0f};
    color dataset_incomplete{0.7f, 0.7f, 0.7f, 1.0f};
    
    vector<event_info *> e_inks;
    vector<vector<event_info *>> e_slices;
    vector<vector<vector<event_info *>>> e_pages;
    
    std::unique_ptr<circular_buffer<double, float>[]> m_in_buffer;
    std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;
    
    std::unique_ptr<float[]> playheads;
    
    bool process_vec_now = false;
    
    vector<vector<float>> terrain_canvas;
    std::vector<std::vector<c74::max::t_jrgb>> terrain_canvas_c;
    
    
    double ms_to_distance    { 1.0 };
    double dist_to_ms    { 116.0 };
    
    
    bool stylus_to_refresh = true;
    bool traj_to_refresh = true;
    

    enum class tasks {dataset, play, stylus, enum_count};
    enum_map tasks_info = {"Dataset", "Play", "Stylus"};
    attribute<tasks> m_task { this, "task", tasks::dataset, tasks_info, title{"UI Tasks"}, description {"The task to be performed on the UI"}, category{"Behavior"}, setter{ MIN_FUNCTION{

        if (static_cast<int>(args[0]) != 2 /*tasks::stylus*/ && !e_inks.empty()) {
            e_inks.back()->phase = "up";
        }
        if (static_cast<int>(args[0]) == 1 /*tasks::play*/) {
            traj_len_ms.clear();
            for (int i = 0; i < m_plays.size(); i++){
                const auto& traj {m_plays[i]};
                traj_len_ms.push_back(traj->ms);
                if (selected_traj == i){
                    m_traj_len.send(traj->ms);
                }
            }
            m_traj_len_all.send(traj_len_ms);
        }
        traj_to_refresh = true;
        return { args[0] };
    }} };
    
    enum class modes {points, audio, enum_count};
    enum_map modes_info = {"Points", "Trajectories"};
    
    attribute<modes> m_mode { this, "mode", modes::audio, modes_info, title{"Dataset Mode"}, description {"The mode for gathering dataset"}, category{"Behavior"}, setter{ MIN_FUNCTION{
        traj_to_refresh = true;
        return { args[0] };
    }} };

    number m_anchor_x {};
    number m_anchor_y {};
    number x_prev {};
    number y_prev {};
    number start_x {};
    number start_y {};

    attribute<number> m_in_ratio{ this, "in_ratio", 2048.0, title{"AutoEncoder Ratio"}, description{"(float) autoencoder's compression ratio in the time domain."}};
    float v_left = -16.0f;
    float v_top = -4.0f;
    float v_right = 16.0f;
    float v_bottom = 4.0f;
    attribute<numbers> v_box{ this, "values_boundary", {-16.0f, -4.0f, 16.0f, 4.0f}, title{"Values at Left-Top-Right-Bottom"}, description{"(float) value at the left, top, right, bottom of the canvas (values won't be clamped, it behaves more like a scale.)"}, setter{ MIN_FUNCTION{
        if (args.size() != 4) {
            cerr << "v_box requires 4 arguments: left, top, right, bottom" << endl;
            return v_box;
        }
        v_left = args[0];
        v_top = args[1];
        v_right = args[2];
        v_bottom = args[3];
        return {args[0], args[1], args[2], args[3]};
    }}};
    
    attribute<int,threadsafe::undefined,limit::clamp> plot_resolution {this, "plot_resolution", 1, title{"Plot Resolution"}, description {"(int) 1~16, the resolution to display the terrain"},range{{1, 16}}, category{"Appearance"}};
    
    attribute<number,threadsafe::undefined,limit::clamp> traj_density{ this, "latent_density", 0.4f, title{"Latent Density"}, description{"Density of latents along a trajectory"}, range{{0.1, 10.0}}, setter{ MIN_FUNCTION{
        float density = static_cast<float>(args[0]);
        ms_to_distance = samplerate()/m_in_ratio*0.001*density;
        dist_to_ms = m_in_ratio/(0.001f*samplerate()*density);
//        cout << "dist_to_ms: " << dist_to_ms << endl;
        return { args[0] };
    }}};
    
    double lt_to_ms(int x){
        return static_cast<double>(x) * m_in_ratio/samplerate() * 1000.0;
    }
    int ms_to_lt(double x){
        return static_cast<int>(std::ceil(x * samplerate() / 1000.0 / m_in_ratio));
    }
    
    attribute<bool> show_terrain{ this, "terrain_display", false, description{"(bool)"}, title{"Terrain Display"}, category{"Appearance"}, setter{ MIN_FUNCTION{
            if (args[0] && terrain_loaded){
                return {true};
            } else {
                return {false};
            }
        } }
    };
    
    attribute<numbers> terrain_clamp { this, "terrain_clamp", {-1.0f, 4.0f}, description{"(min, max)"}, title{"Terrain Value Clamp"}, category{"Appearance"}, setter{ MIN_FUNCTION{
        if (args.size() != 2) {
            cerr << "terrain_clamp requires 2 arguments: min, max" << endl;
            return terrain_clamp;
        }
        return {args[0], args[1]};
    }}};
    attribute<color> t_rgb { this, "terrain_rgb_h", color{1.0,1.0,1.0,1.0}, description{"terrain colour higher bound"}, title{"Terrain Colour Hi"},category{"Appearance"}};
    attribute<color> b_rgb { this, "terrain_rgb_l", color{0.9,0.75,0.6,0.0}, description{"terrain colour lower bound"}, title{"Terrain Colour Lo"},category{"Appearance"}};
    attribute<bool> show_stylus{ this, "stylus_display", true, description{"(bool)"}, title{"Stylus Display"}, category{"Appearance"}, setter{ MIN_FUNCTION{
        return { args[0] };
    }}};
    attribute<int,threadsafe::undefined,limit::clamp> m_buffer_size{ this, "buffer_size", 2048, description{"(int) Buffer size when playing the trajectories. Same values with the terrain and the autoencoder is suggested."}, title{"Buffer Size"}, range{31, 4097}, setter{ MIN_FUNCTION{
        return {int(power_ceil(int(args[0])))};
        } }
    };
    
    attribute<number,threadsafe::undefined,limit::clamp> m_line_width_scale{ this, "stylus_line_width", 3.0, title{"Stylus Line Width"},range{0.1, 10.0}, description{"(float) line width of the drawing in the stylus mode"},category{"Appearance"}};
        
    
    number m_line_width_min = 0.01;

    attribute<color>  ink_color{ this, "stylus_ink_color", color{1.0,1.0,1.0,1.0}, title{"Stylus Ink Colour"}, description{"(color) colour of the drawing in the stylus mode"},category{"Appearance"}};
    number page_refresh = 1000;

    
    float canvas_width = 600.0f;
    float canvas_height = 150.0f;
    
    bool is_using_pen = false;
    bool is_touching = false;
    int terrain_loaded = 0; // 0: no, 1: mono, 4: color
    

    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    attribute<symbol> note_name{ this, "log_name", "UntitledTerrainMap", title{"Saving Name"}, category{"Saving"}};
    
    bool is_refreshing_page = false;

    symbol m_fontname{ "lato-bold" };
    number m_fontsize{ 9.0 };
    
    
    numbers itof(const number x, const number y){
        float dst_range_x = v_right - v_left;
        float dst_range_y = v_bottom - v_top;
        return {x * (dst_range_x / canvas_width) + v_left, y * (dst_range_y / canvas_height) + v_top};
    }
    numbers ftoi(const number x, const number y){
        float dst_range_x = v_right - v_left;
        float dst_range_y = v_bottom - v_top;
        return {canvas_width * (x - v_left) / dst_range_x, canvas_height * (y - v_top) / dst_range_y};
    }
    
    
public:
    MIN_DESCRIPTION    { "Graphic User Interface for Latent Terrain Synth" };
    MIN_TAGS        { "dictionary, nn, ui, multitouch" }; // if multitouch tag is defined then multitouch is supported but mousedragdelta is not supported
    MIN_AUTHOR        { "Jasper Shuoyang Zheng" };
    MIN_RELATED{ "nn.terrain~, nn~, nn.terrain.encode, nn.terrain.record, play~" };

    inlet<>     m_input_sig { this, "(signal) trajectory position to play (in ms)" };
    
    outlet<>    m_output_sig1{ this, "(signal) Control signal 1", "signal" };
    outlet<>    m_output_sig2{ this, "(signal) Control signal 2", "signal" };
    outlet<>    m_traj_len_all{ this, "(list) Total times of each trajectory (msec)", "list"};
    outlet<>    m_traj_len{ this, "(float) Total time of the selected trajectory (msec)", "number" };
    outlet<>    m_traj_end{ this, "(int) Index of the trajectory when the playhead reach its end (1-based index).", "int" };
    outlet<>    m_outlet_pen{ this, "(float) Mouse/pen events: x, y, pen_pressure (if using a stylus)" };
    outlet<>    m_outlet_strokes{ this, "(dictionary) x,y coordinates from trajectories", "dictionary"};
    
    bool out_of_zone_banged = false;
    bool out_of_zone = false;
    
    message<> maxclass_setup{
        this, "maxclass_setup", [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
            cout << "nn.terrain.gui version: 1.5.6.2 Jul-2026" << endl;
            return {};
        }
    };
    message<> dspsetup {this, "dspsetup",
        MIN_FUNCTION {
            ms_to_distance = samplerate()/m_in_ratio*0.001*traj_density;
            dist_to_ms = m_in_ratio/(0.001f*samplerate()*traj_density);
            for (int i = 0; i < m_plays.size(); ++i) {
                const auto& traj {m_plays[i]};
                traj->ms = 0.0f;
                for (segment_info *seg : traj->segments){
                    seg->distance = sqrt(pow(seg->x_end - seg->x_start, 2) + pow(seg->y_end - seg->y_start, 2));
                    seg->ms = seg->distance * dist_to_ms;
                    traj->ms += seg->ms;
                }
                traj_len_ms[play_traj_ptr] = traj->ms;
                if (i == play_traj_ptr){
                    m_traj_len.send(traj->ms);
                }
                m_traj_len_all.send(traj_len_ms);
            }
           return {};
       }
    };
    
    attribute<symbol> external_path{ this, "saving_path", "none", title{"Saving Path"}, category{"Saving"}, setter{ MIN_FUNCTION{
        if (args[0] == "none") {
            symbol min_path_this = min_devkit_path();
            return { min_path_this };
        } else {
            try {
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
    
    stylus(const atoms& args = {});
    ~stylus();
    void clear_trajs(tasks t, modes m);
    
    
    int selected_traj = 0;
    message<> m_selected_traj { this, "setselected",
        "Set the selected trajectory (using 1-based index)",
        MIN_FUNCTION {
            if (args.size() != 1){
                cerr << "args size " << args.size() << " error, args size should be 1" << endl;
                return {};
            }
            if (args[0].a_type != 1){
                cerr << "args type should be int" << endl;
                return {};
            }
            if (m_plays.empty()){
                cerr << "no trajectory available" << endl;
                return {};
            }
            if (static_cast<int>(args[0]) <= 0 || static_cast<int>(args[0]) > m_plays.size()){
                cerr << "selected trajectory index should be in the range [1, " << m_plays.size() << "]" << endl;
                return {};
            }
            selected_traj = static_cast<int>(args[0]) - 1;
            
            process_vec_now = true;
            const auto& traj {m_plays[selected_traj]};
            
            traj_to_refresh = true;
            
            m_traj_len.send(traj->ms);
            
            return {};
        }
    };
    
    vector<double> latent_lens;
    message<> list { this, "list",
        "Length of audio samples (in ms) to be plotted along trajectories",
        MIN_FUNCTION {
            if (m_mode != modes::audio){
                cwarn << "Defining length of audio samples only works in audio dataset mode" << endl;
                return {};
            }
            latent_lens.clear();
            traj_len_ms.clear();
            for (auto i = 0; i < args.size(); ++i) {
                latent_lens.push_back(ms_to_lt(args[i]));
                if (i < m_trajs.size()){
                    const auto& traj {m_trajs[i]};
                    traj_len_ms.push_back(traj->ms);
                    traj->completed = traj->filled_latents >= latent_lens[i];
                }
            }
            const vector<atom> traj_len_ms_num(latent_lens.begin(), latent_lens.end());
            latent_len_attr.set(traj_len_ms_num);
            
            if (m_trajs.size() > args.size()){
                for (auto i = args.size(); i < m_trajs.size(); ++i){
                    const auto& traj {m_trajs[i]};
                    traj->completed = false;
                }
            }
            traj_to_refresh = true;
            return {};
        }
    };
    
    message<> append_m {this, "append", "Add an audio sample length (in ms) to be plotted along trajectories",
        MIN_FUNCTION {
            if (args.size() != 1){
                cerr << "args size " << args.size() << " error, args size should be 1" << endl;
                return {};
            }
            if (args[0].a_type == 3){
                cerr << "arguments need to be int or float" << endl;
                return {};
            }
            if (m_mode != modes::audio){
                cwarn << "Defining length of audio samples only works in audio dataset mode" << endl;
                return {};
            }
            latent_lens.push_back(ms_to_lt(args[0]));
            if (latent_lens.size() <= m_trajs.size()){
                const auto& traj {m_trajs[latent_lens.size()-1]};
                traj_len_ms.push_back(traj->ms);
                traj->completed = traj->filled_latents >= latent_lens.back();
            }
            const vector<atom> traj_len_ms_num(latent_lens.begin(), latent_lens.end());
            latent_len_attr.set(traj_len_ms_num);
            
            traj_to_refresh = true;
            return {};
        }
    };
    
    message<> float_m { this, "float",
        "Length of an audio sample (in ms) to be plotted along a trajectory",
        MIN_FUNCTION {
            if (m_mode != modes::audio){
                cwarn << "Defining length of audio samples only works in audio dataset mode" << endl;
                return {};
            }
            latent_lens.clear();
            traj_len_ms.clear();
            
            latent_lens.push_back(ms_to_lt(args[0]));
            if (0 < m_trajs.size()){
                const auto& traj {m_trajs[0]};
                traj_len_ms.push_back(traj->ms);
                traj->completed = traj->filled_latents >= latent_lens[0];
            }
            
            if (m_trajs.size() > 1){
                for (auto i = 1; i < m_trajs.size(); ++i){
                    const auto& traj {m_trajs[i]};
                    traj->completed = false;
                }
            }
            
            const vector<atom> traj_len_ms_num(latent_lens.begin(), latent_lens.end());
            latent_len_attr.set(traj_len_ms_num);
            
            traj_to_refresh = true;
            return {};
        }
    };
    
    message<> dictionary { this, "dictionary",
        "Plot sampled interval of a terrain, or coordinates of trajectories (depending on the key name of the dictionary)",
        MIN_FUNCTION {
            c74::max::t_symbol    **keys = NULL;
            long        numkeys = 0;
            min_dict d = {args[0]};
            c74::max::dictionary_getkeys(d.m_instance, &numkeys, &keys);
            for (int k = 0; k < numkeys; k++){
                string key_str = std::string(keys[k]->s_name);
                if (key_str == "clips"){
                    atoms clips = d["clips"];
                    for (int i = 0; i < clips.size(); i++){
                        min_dict d2 = {clips[i]};
                        atoms ms = d2["durationms"];
                    }
                } else if (key_str == "terrain"){
                    atom d_data = d[key_str];
                    min_dict terrain_dict = {d_data};
                    int count = static_cast<int>(c74::max::dictionary_getentrycount(terrain_dict.m_instance));
                    if (count == 0) {
                        {
                            std::unique_lock<std::mutex> lk(m_canvas_mutex);
                            terrain_loaded = 0;
                        }
                        if (keys){
                            c74::max::dictionary_freekeys(d.m_instance, numkeys, keys);
                        }
                        return {};
                    }

                    // Extract the raw latent values on this (message) thread only, then
                    // hand off to m_render_thread. The heavy scale()/colour math and the
                    // terrain_canvas build run on the worker; the jgraphics redraw is
                    // triggered by m_redraw_timer once the worker signals completion.
                    symbol skey{"0"};
                    std::vector<std::vector<float>> in_mono;
                    std::vector<std::array<std::vector<float>, 4>> in_color;
                    int in_mode;
                    if (!c74::max::dictionary_entryisdictionary(terrain_dict.m_instance, skey)){
                        // mono terrain
                        in_mode = 1;
                        in_mono.resize(count);
                        for (int y = 0; y < count; y++){
                            atoms latents = terrain_dict[y];
                            int latents_size = static_cast<int>(latents.size());
                            in_mono[y].resize(latents_size);
                            for (int x = 0; x < latents_size; x++){
                                in_mono[y][x] = static_cast<float>(latents[x]);
                            }
                        }
                    } else {
                        // color terrain (4 adjacent latent channels)
                        in_mode = 4;
                        in_color.resize(count);
                        for (int y = 0; y < count; y++){
                            atom line_data = terrain_dict[y];
                            min_dict line_dict = {line_data};
                            atoms latents_0 = line_dict[0];
                            atoms latents_1 = line_dict[1];
                            atoms latents_2 = line_dict[2];
                            atoms latents_3 = line_dict[3];
                            int latents_size = static_cast<int>(latents_0.size());
                            for (int c = 0; c < 4; c++){
                                in_color[y][c].resize(latents_size);
                            }
                            for (int x = 0; x < latents_size; x++){
                                in_color[y][0][x] = static_cast<float>(latents_0[x]);
                                in_color[y][1][x] = static_cast<float>(latents_1[x]);
                                in_color[y][2][x] = static_cast<float>(latents_2[x]);
                                in_color[y][3][x] = static_cast<float>(latents_3[x]);
                            }
                        }
                    }

                    {
                        std::unique_lock<std::mutex> lk(m_render_input_mutex);
                        m_in_mode = in_mode;
                        m_in_mono.swap(in_mono);
                        m_in_color.swap(in_color);
                        m_in_cmin = terrain_clamp[0];
                        m_in_cmax = terrain_clamp[1];
                        m_in_blo = b_rgb.get();
                        m_in_bhi = t_rgb.get();
                    }
                    // signal the worker (guarded so we never over-release the semaphore)
                    if (!m_render_pending.exchange(true)) {
                        m_should_render_lock.release();
                    }

                } else if (key_str == "anchors"){
                    clear_trajs(m_task, m_mode);
                    auto& type_traj = m_task == tasks::play ? m_plays : m_mode == modes::points ? m_points : m_trajs;
                    
                    atom d_data = d[key_str];
                    min_dict coord_dict = {d_data};
                    int count = static_cast<int>(c74::max::dictionary_getentrycount(coord_dict.m_instance));
                    if (count == 0) {
                        if (keys){
                            c74::max::dictionary_freekeys(d.m_instance, numkeys, keys);
                        }
                        return {};
                    }
                    traj_len_ms.clear();
                    for (int i = 0; i < count; i++){
                        atom t_data = coord_dict[std::to_string(i+1)];
                        min_dict traj_dict = {t_data};
                        
                        atoms anchor_x = traj_dict["anchor_x"];
                        atoms anchor_y = traj_dict["anchor_y"];
                        if (anchor_x.size() != anchor_y.size()){
                            cerr << "anchor_x and anchor_y should have the same size" << endl;
                            return {};
                        }
                        vector<segment_info *> new_segments;
                        float traj_ms = 0.0f;
                        int filled_latents = 1;
                        for (int j = 0; j < anchor_x.size()-1; j++){
                            number distance = sqrt(pow(static_cast<double>(anchor_x[j+1])-static_cast<double>(anchor_x[j]), 2) + pow(static_cast<double>(anchor_y[j+1])-static_cast<double>(anchor_y[j]), 2));
                            float ms = static_cast<double>(distance * dist_to_ms);
                            new_segments.push_back(new segment_info{
                                anchor_x[j], anchor_y[j], anchor_x[j+1], anchor_y[j+1],
                                distance,
                                ms,
                                true
                            });
                            traj_ms += ms;
                            filled_latents += ceil((distance-0.001f)/traj_density);
                        }
                        if (m_task == tasks::dataset && m_mode == modes::audio){
                            if (latent_lens.size() > i){
                                bool completed = latent_lens[i] < filled_latents;
                                type_traj.push_back(new traj_info{ new_segments, 0, filled_latents, traj_ms, completed });
                            } else {
                                type_traj.push_back(new traj_info{ new_segments, 0, filled_latents, traj_ms, false });
                            }
                        } else {
                            type_traj.push_back(new traj_info{ new_segments, 0, 1, traj_ms, true });
                        }
                        if (selected_traj == i && m_task == tasks::play){
                            m_traj_len.send(traj_ms);
                        }
                        traj_len_ms.push_back(traj_ms);
                    }
                    
                    if (m_task == tasks::play) {
                        m_traj_len_all.send(traj_len_ms);
                    }
                }
            }
            
            if (keys){
                c74::max::dictionary_freekeys(d.m_instance, numkeys, keys);
            }
            return {};
        }
    };
    
    atoms traj_len_ms;
    attribute<std::vector<int>> latent_len_attr{ this, "latent_length", {1}, title{"Latent Length"}, description{"(list) Number of latents in each trajectory"}, setter{ MIN_FUNCTION{
        if (m_mode != modes::audio){
            cwarn << "Defining length of audio samples only works in audio dataset mode" << endl;
            return {args};
        }
        latent_lens.clear();
        traj_len_ms.clear();
        for (auto i = 0; i < args.size(); ++i) {
            latent_lens.push_back(args[i]);
            if (i < m_trajs.size()){
                const auto& traj {m_trajs[i]};
                traj_len_ms.push_back(traj->ms);
                traj->completed = traj->filled_latents >= latent_lens[i];
            }
        }
        
        if (m_trajs.size() > args.size()){
            for (auto i = args.size(); i < m_trajs.size(); ++i){
                const auto& traj {m_trajs[i]};
                traj->completed = false;
            }
        }
        traj_to_refresh = true;
        return {args};
    }} };
    
    
    number prev_x = 0.0f;
    number prev_y = 0.0f;
    number prev_p = 0.0f;
    symbol prev_phase = "none";
    void send(symbol message_name, const event& e) {
        if (prev_x == e.x() && prev_y == e.y() && prev_p == e.pen_pressure() && prev_phase == message_name){
            return;
        } else {
            prev_x = e.x();
            prev_y = e.y();
            prev_p = e.pen_pressure();
            prev_phase = message_name;
        }
        if (e.x() > canvas_width * 1.05f || e.y() > canvas_height * 1.1f) {
            if (message_name == "up"){
                continue_dragging = true;
                return;
            }
        }
        
        
        numbers xy = itof(e.x(), e.y());
        cursor_x = xy[0];
        cursor_y = xy[1];
        cursor_p = e.pen_pressure();
        
        event_info ei;
        ei.X = xy[0];
        ei.Y = xy[1];
        ei.P = (e.pen_pressure() == 1.0)|| (e.pen_pressure() == 0.0) ? 0.4 : e.pen_pressure();
        ei.phase = message_name;
//        ei.ms = std::chrono::system_clock::now() - now;
        
        if (e.is_shift_key_down()){
            ei.modifier = 1;
        } else if (e.is_command_key_down()){
            ei.modifier = 2;
        } else {
            ei.modifier = 0;
        }
        
        // TODO: make this better:
        ei.color = color{ ink_color.get().red(), ink_color.get().green(), ink_color.get().blue(), ink_color.get().alpha() };
        process_event(ei);

//        m_outlet_main.send(message_name, event_type, e.is_command_key_down(), e.is_shift_key_down());
        m_outlet_pen.send(ei.X, ei.Y, e.pen_pressure());
    }
    
    void process_event(const event_info& ei) {
        if (m_task == tasks::stylus){
            
            // =============== if using stylus ===============
            
            e_inks.push_back(new event_info(ei));
//            redraw();
            stylus_to_refresh = true;
            return;
        } else if (m_task == tasks::dataset && !latent_lens.empty() && m_mode == modes::audio){
            
            // =============== if gathering dataset ===============
            
//            if (ei.X > v_right || ei.Y > v_bottom || ei.X < v_left || ei.Y < v_top) {
//                return;
//            }
            
            if (ei.phase == "down"){
                traj_to_refresh = true;
                if (continue_dragging){
                    continue_dragging = false;
                    return;
                }
                cc = check_click(ei.X, ei.Y);
//                cout << cc.traj_idx << ", " << cc.seg_idx << ", " << cc.front << endl;
                if (cc.traj_idx == -1 && m_trajs.size() < latent_lens.size()){
                    // not clicked on any points, starting a new trajectory
                    vector<segment_info *> new_segments;
                    new_segments.push_back(new segment_info{ ei.X, ei.Y, ei.X, ei.Y, 0, 0.0f, false });
                    m_trajs.push_back(new traj_info{ new_segments, 0, 1, 0.0f, false });
                    traj_ptr = m_trajs.size() - 1;
                    seg_ptr = 0;
                    traj_len_ms.push_back(0.0f);
                    traj_to_refresh = true;
                    is_adding_new = 2; // adding to the end
                    return;
                } else if (cc.traj_idx < latent_lens.size()) {
                    drag_once = true;
                    const auto& traj {m_trajs[cc.traj_idx]};
                    traj_ptr = cc.traj_idx;
                    if (cc.traj_idx >= 0 && ei.modifier > 0){
                        // clicked on a trajectory + shift/command pressed:
                        // TODO: check if the clicked traj is completed
                        
                        if ((cc.seg_idx == traj->segments.size()-1 || traj->segments.size() == 1) && !cc.front){
                            numbers prev_xy{traj->segments.back()->x_end, traj->segments.back()->y_end};
                            while (traj->filled_latents > latent_lens[cc.traj_idx]){
                                // remove segments that contains extra latents
                                number last_dist = traj->segments.back()->distance;
                                int last_seg = ceil((last_dist-0.001f)/traj_density);
                                prev_xy[0] = traj->segments.back()->x_start;
                                prev_xy[1] = traj->segments.back()->y_start;
                                traj->segments.pop_back();
                                traj->filled_latents -= last_seg;
                                traj->filled_latents = fmax(traj->filled_latents, 1);
                                seg_ptr = traj->segments.size()-1;
                            }
                            traj->segments.push_back(new segment_info{ prev_xy[0], prev_xy[1], ei.X, ei.Y, 0, 0.0f, false });
                            seg_ptr = traj->segments.size()-1;
                            is_adding_new = 2; // adding to the end
                        } else if (cc.seg_idx == 0 && cc.front){
                            numbers prev_xy{traj->segments.front()->x_start, traj->segments.front()->y_start};
                            while (traj->filled_latents > latent_lens[cc.traj_idx]){
                                // remove segments that contains extra latents
                                number last_dist = traj->segments.front()->distance;
                                int last_seg = ceil((last_dist-0.001f)/traj_density);
                                prev_xy[0] = traj->segments.front()->x_end;
                                prev_xy[1] = traj->segments.front()->y_end;
                                traj->segments.erase(traj->segments.begin());
                                traj->filled_latents -= last_seg;
                                traj->filled_latents = fmax(traj->filled_latents, 1);
                            }
                            traj->segments.insert(traj->segments.begin(), new segment_info{ ei.X, ei.Y, prev_xy[0], prev_xy[1], 0, 0.0f, false });
                            seg_ptr = 0;
                            is_adding_new = 1; // adding to the front
                        } else {
                            is_adding_new = 0; // just moving something
                            seg_ptr = cc.seg_idx;
                        }
                        traj_to_refresh = true;
                    } else if (cc.traj_idx >= 0 && ei.modifier == 0){
                        // clicked on a trajectory (no key is pressed):
                        
                        seg_ptr = cc.seg_idx;
                        if ((cc.seg_idx == traj->segments.size()-1 || traj->segments.size() == 1) && !cc.front){
                            // if moving the end of the segment:
                            number last_dist = traj->segments.back()->distance;
                            int last_seg = ceil((last_dist-0.001f)/traj_density);
                            traj->filled_latents -= last_seg;
                            traj->filled_latents = fmax(traj->filled_latents, 1);
                            numbers prev_xy{traj->segments.back()->x_start, traj->segments.back()->y_start};
                            
                            traj->segments.pop_back();
                            
                            while (traj->filled_latents > latent_lens[cc.traj_idx]){
                                // remove segments that contains extra latents
                                last_dist = traj->segments.back()->distance;
                                last_seg = ceil((last_dist-0.001f)/traj_density);
                                prev_xy[0] = traj->segments.back()->x_start;
                                prev_xy[1] = traj->segments.back()->y_start;
                                traj->segments.pop_back();
                                traj->filled_latents -= last_seg;
                                traj->filled_latents = fmax(traj->filled_latents, 1);
                                seg_ptr = traj->segments.size()-1;
                            }
                            
                            traj->segments.push_back(new segment_info{ prev_xy[0], prev_xy[1], ei.X, ei.Y, 0, 0.0f, false });
                            
                            seg_ptr = traj->segments.size()-1;
                            
                        } else if (cc.seg_idx == 0 && cc.front){
                            // if moving the start of the segment:
                            number last_dist = traj->segments.front()->distance;
                            int last_seg = ceil((last_dist-0.001f)/traj_density);
                            traj->filled_latents -= last_seg;
                            traj->filled_latents = fmax(traj->filled_latents, 1);
                            numbers prev_xy{traj->segments.front()->x_end, traj->segments.front()->y_end};
                            traj->segments.erase(traj->segments.begin());
                            
                            while (traj->filled_latents > latent_lens[cc.traj_idx]){
                                // remove segments that contains extra latents
                                last_dist = traj->segments.front()->distance;
                                last_seg = ceil((last_dist-0.001f)/traj_density);
                                prev_xy[0] = traj->segments.front()->x_end;
                                prev_xy[1] = traj->segments.front()->y_end;
                                traj->segments.erase(traj->segments.begin());
                                traj->filled_latents -= last_seg;
                                traj->filled_latents = fmax(traj->filled_latents, 1);
                            }
                            traj->segments.insert(traj->segments.begin(), new segment_info{ ei.X, ei.Y, prev_xy[0], prev_xy[1], 0, 0.0f, false });
                            
                            seg_ptr = 0;
                        }
                        
                        
                        is_adding_new = 0;
                        traj_to_refresh = true;
                    }
                } else {
                    is_adding_new = -2;
                    if (latent_lens.empty()){
                        cerr << "the length (ms) of trajectories are not defined 1" << endl;
                    } else if (m_trajs.size() >= latent_lens.size()){
                        cerr << "all " << latent_lens.size() << " audio trajectories have been filled, add more audio samples to create new trajectories" << endl;
                    }
                }
            }
            if (ei.phase == "drag" || continue_dragging || drag_once){
                drag_once = false;
//                cout << is_adding_new<< endl;
                if (is_adding_new == -2){
                    return;
                }
//                cout << "traj_ptr: " << traj_ptr << ", seg_ptr: " << seg_ptr << endl;
                if (traj_ptr >= 0 && traj_ptr < m_trajs.size()){
                    traj_to_refresh = true;
                    const auto& traj {m_trajs[traj_ptr]};
                    
                    traj->ms = 0.0f;
                    for (int i = 0; i < traj->segments.size(); i++){
                        const auto& seg {traj->segments[i]};
                        if (cc.traj_idx>= 0){
                            // if the segment is clicked:
                            if (i == seg_ptr){
                                if (is_adding_new == 2){
                                    // if adding to the end
                                    number distance = sqrt(pow(ei.X - seg->x_start, 2) + pow(ei.Y - seg->y_start, 2));
                                    number target_distance = fmax(traj_density * (latent_lens[traj_ptr] - traj->filled_latents + 1), 0.0f) - 0.001f;
                                    number x_end, y_end;
                                    if (distance <= target_distance){
                                        x_end = ei.X;
                                        y_end = ei.Y;
                                        seg->distance = distance;
                                    } else {
                                        x_end = seg->x_start + (ei.X - seg->x_start) * target_distance / distance;
                                        y_end = seg->y_start + (ei.Y - seg->y_start) * target_distance / distance;
                                        seg->distance = target_distance;
                                    }
                                    seg->x_end = x_end;
                                    seg->y_end = y_end;
                                } else if (is_adding_new == 1){
                                    // if adding to the front
                                    number distance = sqrt(pow(ei.X - seg->x_end, 2) + pow(ei.Y - seg->y_end, 2));
                                    number target_distance = fmax(traj_density * (latent_lens[traj_ptr] - traj->filled_latents + 1), 0.0f) - 0.001f;
                                    number x_end, y_end;
                                    if (distance <= target_distance){
                                        x_end = ei.X;
                                        y_end = ei.Y;
                                        seg->distance = distance;
                                    } else {
                                        x_end = seg->x_end + (ei.X - seg->x_end) * target_distance / distance;
                                        y_end = seg->y_end + (ei.Y - seg->y_end) * target_distance / distance;
                                        seg->distance = target_distance;
                                    }
                                    seg->x_start = x_end;
                                    seg->y_start = y_end;
                                } else if (is_adding_new == 0) {
                                    // if moving the clicked segment (no adding)
                                    if (cc.front){
                                        // if in the front of the trajectory
                                        number distance = sqrt(pow(ei.X - seg->x_end, 2) + pow(ei.Y - seg->y_end, 2));
                                        number target_distance = fmax(traj_density * (latent_lens[traj_ptr] - traj->filled_latents + 1), 0.0f) - 0.001f;
                                        number x_end, y_end;
                                        if (distance <= target_distance){
                                            x_end = ei.X;
                                            y_end = ei.Y;
                                            seg->distance = distance;
                                        } else {
                                            x_end = seg->x_end + (ei.X - seg->x_end) * target_distance / distance;
                                            y_end = seg->y_end + (ei.Y - seg->y_end) * target_distance / distance;
                                            seg->distance = target_distance;
                                        }
                                        seg->x_start = x_end;
                                        seg->y_start = y_end;
                                    } else if (i != traj->segments.size()-1) {
                                        // if in the middle of the trajectory
                                        const auto& seg_next {traj->segments[i+1]};
                                        seg->x_end = ei.X;
                                        seg->y_end = ei.Y;
                                        seg_next->x_start = ei.X;
                                        seg_next->y_start = ei.Y;
                                        seg->distance = sqrt(pow(seg->x_end - seg->x_start, 2) + pow(seg->y_end - seg->y_start, 2));
                                        seg_next->distance = sqrt(pow(seg_next->x_end - seg_next->x_start, 2) + pow(seg_next->y_end - seg_next->y_start, 2));
                                    } else {
                                        // if in the end of the trajectory
                                        number distance = sqrt(pow(ei.X - seg->x_start, 2) + pow(ei.Y - seg->y_start, 2));
                                        number target_distance = fmax(traj_density * (latent_lens[traj_ptr] - traj->filled_latents + 1), 0.0f) - 0.001f;
                                        number x_end, y_end;
                                        if (distance <= target_distance){
                                            x_end = ei.X;
                                            y_end = ei.Y;
                                            seg->distance = distance;
                                        } else {
                                            x_end = seg->x_start + (ei.X - seg->x_start) * target_distance / distance;
                                            y_end = seg->y_start + (ei.Y - seg->y_start) * target_distance / distance;
                                            seg->distance = target_distance;
                                        }
                                        seg->x_end = x_end;
                                        seg->y_end = y_end;
                                    }
                                }
                            }
                        } else if (cc.traj_idx == -1){
//                            cout << "not clicked on any segments" << endl;
                            // if the segment is not clicked
                            number distance = sqrt(pow(ei.X - seg->x_start, 2) + pow(ei.Y - seg->y_start, 2));
                            number target_distance = fmax(traj_density * (latent_lens[m_trajs.size() - 1] - traj->filled_latents + 1), 0.0f) - 0.001f;
                            number x_end, y_end;
                            if (distance <= target_distance){
                                x_end = ei.X;
                                y_end = ei.Y;
                                seg->distance = distance;
                            } else {
                                x_end = seg->x_start + (ei.X - seg->x_start) * target_distance / distance;
                                y_end = seg->y_start + (ei.Y - seg->y_start) * target_distance / distance;
                                seg->distance = target_distance;
                            }
                            seg->x_end = x_end;
                            seg->y_end = y_end;
                        }
                        seg->ms = seg->distance * dist_to_ms;
                        traj->ms += seg->ms;
                    }
                    traj_len_ms[traj_ptr] = traj->ms;
                    return;
                }
            }
            if (ei.phase == "up"){
                if (is_adding_new == -2){
                    return;
                }
                if (ei.X > v_right || ei.Y > v_bottom || ei.X < v_left || ei.Y < v_top) {
                    continue_dragging = true;
//                    cout << "continue dragging set from dataset" << endl;
                    return;
                }
                is_adding_new = 0;
                if (traj_ptr >= 0 && traj_ptr < m_trajs.size()){
                    const auto& traj {m_trajs[traj_ptr]};
                    traj_to_refresh = true;
                    
                    traj->ms = 0.0f;
                    traj->filled_latents = 1;
                    for (segment_info *seg : traj->segments){
                        float effective_distance = (floor(seg->distance / traj_density)+1) * traj_density - 0.01f;
                        
                        traj->ms += effective_distance * dist_to_ms;
                        traj->filled_latents += ceil((seg->distance-0.001f)/traj_density);
                        seg->completed = true;
                    }
                    
                    traj->completed = traj->filled_latents >= latent_lens[traj_ptr];
                    
                    if (traj_ptr < latent_lens.size()){
                        traj_len_ms[traj_ptr] = traj->ms;
                    }
                    dump_strokes(m_task, false);
                    return;
                }
                traj_ptr = 0;
            }
        } else if (m_task == tasks::dataset && (latent_lens.empty()) && m_mode == modes::audio){
//            cerr << "here" << endl;
            if (ei.phase == "up" || ei.phase == "down"){
                if (latent_lens.empty()){
                    cerr << "the length (ms) of trajectories are not defined" << endl;
                } else if (traj_ptr >= latent_lens.size()){
                    cerr << "ran out of audio sample" << endl;
                }
            }
            return;
        } else if (m_task == tasks::dataset && m_mode == modes::points){
            
            // =============== if gathering dataset in points mode ===============
            
            if (ei.phase == "down"){
                if (continue_dragging){
                    continue_dragging = false;
                    return;
                }
                cc = check_click(ei.X, ei.Y);
                if (cc.traj_idx == -1){
                    vector<segment_info *> new_segments;
                    new_segments.push_back(new segment_info{ ei.X, ei.Y, ei.X, ei.Y, 0, 0.0f, false });
                    m_points.push_back(new traj_info{ new_segments, 0, 1, 0.0f, false });
                    traj_ptr = m_points.size()-1;
                    traj_to_refresh = true;
                    return;
                } else {
                    traj_ptr = cc.traj_idx;
                    traj_to_refresh = true;
                    return;
                }
            }
            if (ei.phase == "drag" || continue_dragging){
                if (traj_ptr >= 0 && traj_ptr < m_points.size()){
                    const auto& traj {m_points[traj_ptr]};
                    const auto& seg {traj->segments.back()};
                    seg->x_start = ei.X;
                    seg->y_start = ei.Y;
                    seg->x_end = ei.X;
                    seg->y_end = ei.Y;
                    traj_to_refresh = true;
                    return;
                }
            }
            if (ei.phase == "up"){
                if (ei.X > v_right || ei.Y > v_bottom || ei.X < v_left || ei.Y < v_top) {
                    continue_dragging = true;
//                    cout << "continue dragging set from points" << endl;
                    return;
                }
                if (traj_ptr >= 0 && traj_ptr < m_points.size()){
                    const auto& traj {m_points[traj_ptr]};
                    const auto& seg {traj->segments.back()};
                    seg->x_start = ei.X;
                    seg->y_start = ei.Y;
                    seg->x_end = ei.X;
                    seg->y_end = ei.Y;
                    seg->completed = true;
                    seg->distance = 0.001f;
                    traj->completed = true;
                    traj_to_refresh = true;
                    dump_strokes(m_task, false);
                    return;
                }
            }
                
        } else if (m_task == tasks::play){
            
            // =============== if playing ===============
            
            if (ei.phase == "down"){
                if (continue_dragging){
                    continue_dragging = false;
                    return;
                }
                cc = check_click(ei.X, ei.Y);
                
                if (m_plays.empty()){
                    vector<segment_info *> new_segments;
                    new_segments.push_back(new segment_info{ ei.X, ei.Y, ei.X, ei.Y, 0, 0.0f, false });
                    m_plays.push_back(new traj_info{ new_segments, 0, 1, 0.0f, false });
                    play_traj_ptr = m_plays.size() - 1;
                    traj_to_refresh = true;
                    traj_len_ms.push_back(0.0f);
                    is_adding_new = 2;
                    return;
                }
                
                if (cc.traj_idx == -1){
                    // not clicked on any points, starting a new trajectory
                    vector<segment_info *> new_segments;
                    new_segments.push_back(new segment_info{ ei.X, ei.Y, ei.X, ei.Y, 0, 0.0f, false });
                    m_plays.push_back(new traj_info{ new_segments, 0, 1, 0.0f, false });
                    play_traj_ptr = m_plays.size() - 1;
                    play_seg_ptr = 0;
                    traj_len_ms.push_back(0.0f);
                    traj_to_refresh = true;
                    return;
                } else if (cc.traj_idx >= 0 && ei.modifier > 0){
                    // clicked on a trajectory + shift/command pressed:
                    const auto& traj {m_plays[cc.traj_idx]};
                    
                    play_traj_ptr = cc.traj_idx;
                    if ((cc.seg_idx == traj->segments.size()-1 || traj->segments.size() == 1) && !cc.front){
                        traj->segments.push_back(new segment_info{ cc.x, cc.y, ei.X, ei.Y, 0, 0.0f, false });
                        play_seg_ptr = cc.seg_idx + 1;
                        is_adding_new = 2; // adding to the end
                    } else if (cc.seg_idx == 0 && cc.front){
                        traj->segments.insert(traj->segments.begin(), new segment_info{ ei.X, ei.Y, cc.x, cc.y, 0, 0.0f, false });
                        play_seg_ptr = 0;
                        is_adding_new = 1; // adding to the front
                    } else {
                        is_adding_new = 0; // just moveing something
                        play_seg_ptr = cc.seg_idx;
                    }
                    traj_to_refresh = true;
                } else if (cc.traj_idx >= 0 && ei.modifier == 0){
                    // clicked on a trajectory (no key is pressed):
                    play_traj_ptr = cc.traj_idx;
                    play_seg_ptr = cc.seg_idx;
                    is_adding_new = 0;
                }
            }
            if (ei.phase == "drag" || continue_dragging){
                if (play_traj_ptr >= 0 && play_traj_ptr < m_plays.size()){
                    const auto& traj {m_plays[play_traj_ptr]};
                    traj->ms = 0.0f;
                    for (int i = 0; i < traj->segments.size(); i++){
                        const auto& seg {traj->segments[i]};
                        if (cc.traj_idx>= 0){
                            // if the segment is clicked:
                            if (i == play_seg_ptr){
                                if (is_adding_new == 2){
                                    // if adding to the end
                                    seg->x_end = ei.X;
                                    seg->y_end = ei.Y;
                                } else if (is_adding_new == 1){
                                    // if adding to the front
                                    seg->x_start = ei.X;
                                    seg->y_start = ei.Y;
                                } else {
                                    // if moving the clicked segment (no adding)
                                    if (cc.front){
                                        seg->x_start = ei.X;
                                        seg->y_start = ei.Y;
                                    } else if (i != traj->segments.size()-1) {
                                        const auto& seg_next {traj->segments[i+1]};
                                        seg->x_end = ei.X;
                                        seg->y_end = ei.Y;
                                        seg_next->x_start = ei.X;
                                        seg_next->y_start = ei.Y;
                                    } else {
                                        seg->x_end = ei.X;
                                        seg->y_end = ei.Y;
                                    }
                                }
                            }
                        } else {
                            // if the segment is not clicked
                            seg->x_end = ei.X;
                            seg->y_end = ei.Y;
                        }
                        seg->distance = sqrt(pow(seg->x_end - seg->x_start, 2) + pow(seg->y_end - seg->y_start, 2));
                        seg->ms = seg->distance * dist_to_ms;
                        traj->ms += seg->ms;
                    }
                    traj_len_ms[play_traj_ptr] = traj->ms;
                    if (selected_traj == play_traj_ptr){
                        m_traj_len.send(traj->ms);
                    }
                    
                    traj_to_refresh = true;
                    return;
                }
            }
            if (ei.phase == "up"){
                if (ei.X > v_right || ei.Y > v_bottom || ei.X < v_left || ei.Y < v_top) {
                    continue_dragging = true;
//                    cout << "continue dragging set from playing" << endl;
                    return;
                }
                is_adding_new = 0;
                if (play_traj_ptr >= 0 && play_traj_ptr < m_plays.size()){
                    const auto& traj {m_plays[play_traj_ptr]};
                    // TODO: if the distance is too small, don't count it as a segment
                    traj->completed = true;
                    traj_to_refresh = true;
                    
                    traj->ms = 0.0f;
                    for (segment_info *seg : traj->segments){
                        traj->ms += seg->ms;
                        seg->completed = true;
                    }
                    traj_len_ms[play_traj_ptr] = traj->ms;
                    if (selected_traj == play_traj_ptr){
                        m_traj_len.send(traj->ms);
                    }
                    dump_strokes(m_task, true);
                    m_traj_len_all.send(traj_len_ms);
                    return;
                }
                
            }
            if (ei.phase == "move"){
                traj_to_refresh = true;
            }
            return;
        }
    }
    
    click_check_info check_click(const number x, const number y){
        click_check_info cc;
        cc.traj_idx = -1;
        cc.seg_idx = 0;
        const auto& type_traj = m_task == tasks::play ? m_plays : m_mode == modes::points ? m_points : m_trajs;
        
        for (int i = 0; i < type_traj.size(); i++){
            // check if it clicked on any point at the end of the trajectory
            const auto& traj {type_traj[i]};
            for (int j = 0; j < traj->segments.size(); j++){
                const auto& seg {traj->segments[j]};
                if (abs(seg->x_start - x) < traj_density * 2.0f && abs(seg->y_start - y) < traj_density * 2.0f){
                    cc.traj_idx = i;
                    cc.seg_idx = j;
                    cc.x = seg->x_start;
                    cc.y = seg->y_start;
                    cc.front = true;
//                    cout << "clicked front: [" << i << ", " << j << "]" << endl;
                    return cc;
                }
                if (abs(seg->x_end - x) < traj_density * 2.0f && abs(seg->y_end - y) < traj_density * 2.0f){
                    cc.traj_idx = i;
                    cc.seg_idx = j;
                    cc.x = seg->x_end;
                    cc.y = seg->y_end;
                    cc.front = false;
//                    cout << "clicked end: [" << i << ", " << j << "]" << endl;
                    return cc;
                }
            }
        }
        return cc;
    }
    
    bool hover_check(const number x, const number y){
        return abs(x - cursor_x) < traj_density * 2.0f && abs(y - cursor_y) < traj_density * 2.0f;
    }
    
    
    message<> m_clear{ this, "clear",
        MIN_FUNCTION {
            vector<event_info *> new_e_phase_slice = e_inks;
            e_slices.push_back(new_e_phase_slice);
            vector<vector<event_info *>> new_e_phase = e_slices;
            e_pages.push_back(new_e_phase);
            
            e_inks.clear();
            e_slices.clear();
            
            m_pen.clear_history();
            traj_ptr = 0;
            clear_trajs(m_task, m_mode);
            if (m_mode == modes::points && m_task == tasks::dataset){
                m_points.clear();
            }
            if (m_task == tasks::play){
                dump_strokes(m_task, true);
                m_outlet_strokes.send("dictionary",coordinates_dict.name());
            }
            stylus_to_refresh = true;
            traj_to_refresh = true;
            return {};
        }
    };

    message<> m_mouseenter { this, "mouseenter", MIN_FUNCTION{send("enter", args);return {};}};
    message<> m_mousemove { this, "mousemove", MIN_FUNCTION{send("move", args);return {};}};
    message<> m_mouseleave { this, "mouseleave", MIN_FUNCTION{send("leave", args);return {};}};
    message<> m_mousedown{ this, "mousedown",
        MIN_FUNCTION {
            is_touching = true;
            send("down", args);
            return {};
        }
    };
    message<> m_mouseup{ this, "mouseup",
        MIN_FUNCTION {
            is_touching = false;
            send("up", args);
            return {};
        }
    };
    message<> m_mousedrag{ this, "mousedrag",MIN_FUNCTION {send("drag", args);return {};}};
    
    min_dict coordinates_dict { symbol(true) };

    message<> m_send_strokes{ this, "getcontent", "Get all trajectories in a dictionary" , MIN_FUNCTION {
        if (m_task == tasks::stylus){
            cwarn << "please use the 'log' message to get stylus drawings" << endl;
            return {};
        }
        dump_strokes(m_task, false);
        return {};
    } };
    void dump_strokes(tasks m_task, bool anchor_only){
        coordinates_dict.clear();
        min_dict coordinates_dict_data { symbol(true) };
        min_dict anchor_dict_data { symbol(true) };
        const auto& type_traj = m_task == tasks::play ? m_plays : m_mode == modes::points ? m_points : m_trajs;
        
        for (int i(0); i < type_traj.size(); i++) {
            if (i >= latent_lens.size() && (m_mode == modes::audio && m_task == tasks::dataset)){
                continue;
            }
            const auto& traj {type_traj[i]};
            min_dict traj_dict = {};
            min_dict achor_point_dict = {};
            atoms anchor_x = {};
            atoms anchor_y = {};
            
            
            for (int z(0); z < 2 ; z++){
                atoms result = {};
                int p_counts = 0;
                int max_points = m_mode == modes::audio && m_task == tasks::dataset ? latent_lens[i] : 999;
                for (int s(0); s < traj->segments.size(); s++){
                    const auto& seg {traj->segments[s]};
                    if (!anchor_only){
                        for (float j(0.0f); j < seg->distance+0.001f; j+=traj_density) {
                            if (p_counts <= max_points){
                                if (z == 0){
                                    result.push_back(seg->x_start + (seg->x_end - seg->x_start) * j / (seg->distance+0.00001f));
                                } else {
                                    result.push_back(seg->y_start + (seg->y_end - seg->y_start) * j / (seg->distance+0.00001f));
                                }
                                p_counts += 1;
                            }
                        }
                    }
                    if (z == 0){
                        if (s == 0){
                            anchor_x.push_back(seg->x_start);
                            anchor_y.push_back(seg->y_start);
                        }
                        anchor_x.push_back(seg->x_end);
                        anchor_y.push_back(seg->y_end);
                    }
                }
                if (!anchor_only){
                    symbol skey{z};
                    c74::max::dictionary_appendatoms(traj_dict.m_instance, skey, result.size(), &result[0]);
                }
            }
            if (!anchor_only){
                coordinates_dict_data[std::to_string(i+1)] = traj_dict;
            }
            symbol skey1{"anchor_x"};
            c74::max::dictionary_appendatoms(achor_point_dict.m_instance, skey1, anchor_x.size(), &anchor_x[0]);
            symbol skey2{"anchor_y"};
            c74::max::dictionary_appendatoms(achor_point_dict.m_instance, skey2, anchor_y.size(), &anchor_y[0]);
            anchor_dict_data[std::to_string(i+1)] = achor_point_dict;
        }
        if (!anchor_only){
            coordinates_dict["coordinates"] = coordinates_dict_data;
        }
        coordinates_dict["anchors"] = anchor_dict_data;
        coordinates_dict.touch();
        
        m_outlet_strokes.send("dictionary",coordinates_dict.name());
    }
    terrain m_terrain{ this, 600.0, 150.0, [this](const c74::min::atoms& args, const int inlet) -> c74::min::atoms{
        target t { args };
        // read the canvas the worker built; worker only holds this lock for an O(1) swap
        std::unique_lock<std::mutex> canvas_lock(m_canvas_mutex);
        if (terrain_loaded==1){
            for (int i = 0; i < terrain_canvas.size(); i++) {
                for (int j = 0; j < terrain_canvas[0].size(); j++) {
                    float v = terrain_canvas[i][j];
                    rect<fill> {
                        t,
                        color{ v, v, v, 1.0 },
                        position{ j*plot_resolution, i*plot_resolution },
                        size{ static_cast<double>(plot_resolution), static_cast<double>(plot_resolution) }
                    };
                }
            }
        } else if (terrain_loaded==4){
            for (int i = 0; i < terrain_canvas_c.size(); i++) {
                for (int j = 0; j < terrain_canvas_c[0].size(); j++) {
                        c74::max::t_jrgb c = terrain_canvas_c[i][j];
                        rect<fill> {
                            t,
                            color{ c.red, c.green, c.blue, 1.0 },
                            position{ j*plot_resolution, i*plot_resolution },
                            size{ static_cast<double>(plot_resolution), static_cast<double>(plot_resolution) }
                        };
                }
            }
        }
        return {};
    } };
    
    
    
    pen m_pen{ this, MIN_FUNCTION{
        target t { args };
        x_prev = 0.0;
        y_prev = 0.0;
        float ink = 0.75;
        number radius = 2;
        for (auto i = e_inks.size() ; i > 0; --i) {
            const auto& phase {e_inks[i-1]};
            numbers phase_xy = ftoi(phase->X, phase->Y);
            if ((phase->phase == "up") || (phase->phase == "drag")) {
                is_using_pen = true;
            }
            bool permanent = (phase->phase == "drag") || (phase->phase == "down") || (phase->phase == "up");
            color c { phase->color.red(), phase->color.green(), phase->color.blue(), ink};
            m_anchor_x = phase_xy[0];
            m_anchor_y = phase_xy[1];
            if ((i == e_inks.size()) || (phase->phase == "up") ){
                start_x = m_anchor_x;
                start_y = m_anchor_y;
            } else {
                start_x = x_prev;
                start_y = y_prev;
            }
            number line_width_now = m_line_width_min + m_line_width_scale * phase->P;
            if (i == e_inks.size()-1){
                ellipse<fill> {t, color{ c }, position{ phase_xy[0] - 4, phase_xy[1] - 4 }, size{ 8, 8 } };
            }
            if (permanent) {
                line<> { t, color{ c }, position{ start_x, start_y }, destination{ m_anchor_x, m_anchor_y }, line_width{ line_width_now } };
            } else {
                ellipse<fill> { t, color{ c }, position{ phase_xy[0] - radius, phase_xy[1] - radius }, size{ radius * 2, radius * 2 } };
                e_inks.erase(e_inks.begin() + (i - 1));
            }
            x_prev = m_anchor_x;
            y_prev = m_anchor_y;
        }

        if (is_refreshing_page) {
//            cout << "page refresh" << endl;
            e_inks.back()->phase = "up";
            m_pen.lock_canvas();
            e_slices.push_back(e_inks);
            e_inks.clear();
            event_info ei;
            numbers xy = ftoi(e_slices.back().back()->X, e_slices.back().back()->Y);
            ei.X = xy[0];
            ei.Y = xy[1];
            ei.phase = "down";
            ei.color = color{ 0.8, 0.8, 0.8, 0.0 };
//            ei.ms = e_slices.back().back()->ms;
            e_inks.push_back(new event_info(ei));
            is_refreshing_page = false;
        }
        if ((e_inks.size() >= page_refresh)) {
            is_refreshing_page = true;
        }
        return{ {ink} };
    }, MIN_FUNCTION{
        // trajectories for dataset gathering
        target t { args };
        const auto& type_traj = m_mode == modes::audio ? m_trajs : m_points;
        for (int i(0); i < type_traj.size(); i++) {
            const auto& traj {type_traj[i]};
            numbers text_position;
            bool hovered = false;
            
            for (int j(0); j < traj->segments.size(); j++) {
                const auto& seg {traj->segments[j]};
                numbers xy_start = ftoi(seg->x_start, seg->y_start);
                numbers xy_end = ftoi(seg->x_end, seg->y_end);
                color cs = traj->completed ? dataset_complete : dataset_incomplete;
                line<> {
                    t, cs, position{ xy_start[0], xy_start[1] }, destination{ xy_end[0], xy_end[1] }, line_width{ 1.5 } };
                color c;
                if (j == 0){
                    if (hover_check(seg->x_start, seg->y_start) && !hovered){
                        hovered = true;
                        c = dataset_selected;
                    } else {
                        c = cs;
                    }
                    ellipse<fill> {
                        t, c, position{ xy_start[0]-6, xy_start[1]-6 }, size{ 12, 12 } };
                }
                if (j > 0){
                    if (hover_check(seg->x_start, seg->y_start) && !hovered){
                        hovered = true;
                        c = dataset_selected;
                    } else {
                        c = cs;
                    }
                    ellipse<fill> {
                        t, c, position{ xy_start[0]-3, xy_start[1]-3 }, size{ 6, 6 } };
                }
                if (j == traj->segments.size()-1){
                    if (hover_check(seg->x_end, seg->y_end) && !hovered){
                        hovered = true;
                        c = dataset_selected;
                    } else {
                        c = cs;
                    }
                    ellipse<fill> {
                        t, c, position{ xy_end[0]-6, xy_end[1]-6 }, size{ 12, 12 } };
                }
                if (j == 0){
                    text_position = {xy_start[0]-3, xy_start[1]+3};
                }
            }
            text{
                t, color {color::predefined::black},
                position {text_position[0], text_position[1]},
                fontface {m_fontname}, fontsize {m_fontsize},
                content {std::to_string(i+1)} };
        }
            
        return {{0}};
    }, MIN_FUNCTION{
        // trajectories for playing
        target t { args };
        for (int i(0); i < m_plays.size(); i++) {
            const auto& traj {m_plays[i]};
            numbers text_position;
            color cc = i == selected_traj ? playing_selected : playing_standby;
            color c_fade = color{cc.red()*0.8, cc.green()*0.8, cc.blue()*0.8, cc.alpha()};
            bool hovered = false;
            for (int j(0); j < traj->segments.size(); j++) {
                const auto& seg {traj->segments[j]};
                numbers xy_start = ftoi(seg->x_start, seg->y_start);
                numbers xy_end = ftoi(seg->x_end, seg->y_end);
                if (seg->completed) {
                    color c;
                    line<> { t, cc, position{ xy_start[0], xy_start[1] }, destination{ xy_end[0], xy_end[1] }, line_width{ 1.5 } };
                    if (j == 0){
                        if (hover_check(seg->x_start, seg->y_start) && !hovered){
                            hovered = true;
                            c = dataset_selected;
                        } else {
                            c = cc;
                        }
                        ellipse<fill> {t, c, position{ xy_start[0]-6, xy_start[1]-6 }, size{ 12, 12 } };
                    }
                    if (j > 0){
                        if (hover_check(seg->x_start, seg->y_start) && !hovered){
                            hovered = true;
                            c = dataset_selected;
                        } else {
                            c = cc;
                        }
                        ellipse<fill> { t, c, position{ xy_start[0]-3, xy_start[1]-3 }, size{ 6, 6 } };
                    }
                    if (j == traj->segments.size()-1){
                        if (hover_check(seg->x_end, seg->y_end) && !hovered){
                            hovered = true;
                            c = dataset_selected;
                        } else {
                            c = cc;
                        }
                        ellipse<fill> { t, c, position{ xy_end[0]-6, xy_end[1]-6 }, size{ 12, 12 } };
                    }
                    
                } else {
                    line<> {
                        t, c_fade, position{ xy_start[0], xy_start[1] }, destination{ xy_end[0], xy_end[1] }, line_width{ 1.5 } };
                    ellipse<fill> {
                        t, c_fade, position{ xy_start[0]-3, xy_start[1]-3 }, size{ 6, 6 } };
                    ellipse<fill> {
                        t, c_fade, position{ xy_end[0]-3, xy_end[1]-3 }, size{ 6, 6 } };
                }
                if (j == 0){
                    text_position = {xy_start[0]-3, xy_start[1]+3};
                }
            }
            if (traj->completed) {
                text{
                    t, color {color::predefined::black},
                    position {text_position[0], text_position[1]},
                    fontface {m_fontname}, fontsize {m_fontsize},
                    content {std::to_string(i+1)} };
            }
        }
            
        return {{0}};
    }
    };

    message<> m_save_log{ this, "log",
        MIN_FUNCTION {
            try {
                atoms results = create_log_and_save(std::to_string(note_name) + "", std::to_string(external_path), "not implemented yet");
                cout << "saved log to: " << results[0] << endl;
                
                // careful if the logs folder is not created
                min_path m_log_path = min_path(std::to_string(external_path));
                m_terrain.save_png(static_cast<string>(results[1])+"_terrain.png", m_log_path, 150);
                m_pen.save_png(static_cast<string>(results[1])+".png", m_log_path, 150);
            } catch (const std::exception &e) {
                cerr << "Error saving png:" << e.what() << endl;
            }
            return {};
        }
    };
    std::mutex t_mutex;
    message<> m_paint{ this, "paint",
        MIN_FUNCTION {
            std::unique_lock<std::mutex> t_lock(t_mutex);
            
            target t { args };
            
            if (canvas_width != t.width() || canvas_height != t.height()) {
//                cout << "paint" << t.width() << endl;
                canvas_width = t.width();
                canvas_height = t.height();
                m_terrain.redraw(canvas_width, canvas_height);
            }
            
            if (show_terrain) {
//                if(m_finish_render_lock.try_acquire()){
                    m_terrain.draw_surface(t, t.width(), t.height());
//                }
            } else {
                rect<fill>{t, color{ 0.1, 0.1, 0.1, 1.0 }};
            }
            if (m_task == tasks::dataset){
                if (stylus_to_refresh || traj_to_refresh) {
                    for (int y(0); y < floor(canvas_height/20);y++){
                        for (int x(0); x < floor(canvas_width/20);x++){
                            rect<fill>{t, color{ 0.5, 0.5, 0.5, 1.0 }, position{ 15+x*20, 15+y*20 }, size{ 1.0, 1.0 }};
                        }
                    }
                }
            }
            
            numbers xy = ftoi(cursor_x, cursor_y);
            if (m_task != tasks::stylus) {
                ellipse<stroke> {
                    t,
                    color{ 0.8, 0.8, 0.8, 1.0 },
                    position{ xy[0] - 10, xy[1] - 10 },
                    size{ 10 * 2, 10 * 2 }
                };
            } else {
                ellipse<fill> {
                    t,
                    color{ 0.8, 0.8, 0.8, 1.0 },
                    position{ xy[0] - 2, xy[1] - 2 },
                    size{ 4, 4 }
                };
            }
            
            if (m_task == tasks::stylus && stylus_to_refresh) {
                m_pen.draw_inks(t.width(), t.height());
            }
            if (show_stylus){
                m_pen.draw_surface(t, t.width(), t.height());
            }
            if (m_task == tasks::stylus){
                return {};
            }
            
            if (m_task == tasks::dataset) {
                m_pen.draw_trajs(t.width(), t.height());
            } else if (m_task == tasks::play) {
                m_pen.draw_plays(t.width(), t.height());
            }
            
            m_pen.draw_trajs_surface(t);
            
            numbers playheads_xy = ftoi(playheads[0], playheads[1]);
            ellipse<stroke> {
                t,
                color{ 0.976, 0.737, 0.255, 1.0 },
                position{ playheads_xy[0] - 8, playheads_xy[1] - 8 },
                size{ 16, 16 }
            };
            t_lock.unlock();
            return {};
        }
    };
    
    
    attribute<number,threadsafe::undefined,limit::clamp> m_display_rate{ this, "interval", 43.4, description{"The rate at which the display is updated."}, title{"Update Interval in Milliseconds"}, range{20, 100}, category{"Appearance"}
    };
    
    timer<timer_options::defer_delivery> m_redraw_timer { this,
        MIN_FUNCTION {
            // when the worker has a fresh canvas, render it into the surface here on
            // the main thread (jgraphics must not run off the main thread)
            if (m_finish_render_lock.try_acquire()) {
                m_terrain.redraw(canvas_width, canvas_height);
                show_terrain = true;
            }
            redraw();
            m_redraw_timer.delay(m_display_rate);
            return {};
        }
    };
    timer<timer_options::defer_delivery> m_playhead_timer { this,
        MIN_FUNCTION {
            if (out_of_zone && !out_of_zone_banged){
                m_traj_end.send(selected_traj+1);
                out_of_zone_banged = true;
            } else if (!out_of_zone && out_of_zone_banged){
                out_of_zone_banged = false;
            }
            m_playhead_timer.delay(5);
            return {};
        }
    };
    
    void operator()(audio_bundle input, audio_bundle output) {
        if (m_plays.empty()) {
            return;
        }
        int vec_size = input.frame_count();// 128
        
        auto in = input.samples(0);
        m_in_buffer[0].put(in, vec_size);
        
        if (m_in_buffer[0].full() || process_vec_now) {
            
            float play = *input.samples(0);
            
            update_playheads(play);
            
            process_vec_now = false;
            m_in_buffer[0].reset();
        }
        for (int c(0); c < output.channel_count(); c++) {
            auto out = output.samples(c);
            std::fill(out, out + vec_size, playheads[c]);
        }
    }
    
    bool update_playheads(float p){
        const auto& traj {m_plays[selected_traj]};
        if (traj->segments.empty()){
            return false;
        }
        // we can't do any outlet-sending in the audio thread
        if (p >= traj->ms && traj->ms > 0.0f){
            out_of_zone = true;
        } else {
            out_of_zone = false;
        }
        
        p = fmax(0.0f, fmin(p, traj->ms));
        float played_distance = p * ms_to_distance;
        for (segment_info* seg : traj->segments) {
                float dist = seg->distance;
                if (played_distance > dist){
                    played_distance -= dist;
                } else {
                    dist = dist==0.0f ? 0.0001f : dist;
                    playheads[0] = seg->x_start + played_distance / dist * (seg->x_end - seg->x_start);
                    playheads[1] = seg->y_start + played_distance / dist * (seg->y_end - seg->y_start);
                    return true;
                }
        }
        return false;
    }
};


stylus::stylus(const atoms& args) : ui_operator::ui_operator {this, args} {
    
    m_in_buffer = std::make_unique<circular_buffer<double, float>[]>(1);
    for (int i(0); i < 1; i++) {
      m_in_buffer[i].initialize(m_buffer_size); //2048
    }
    m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(2);
    for (int i(0); i < 2; i++) {
        m_out_buffer[i].initialize(m_buffer_size);
    }

    m_redraw_timer.delay(1000);
    m_playhead_timer.delay(1000);
    
    playheads = std::make_unique<float[]>(2);
    playheads[0] = -32.0f;
    playheads[1] = -32.0f;

    m_render_thread = std::make_unique<std::thread>(terrain_render_loop, this);
}

stylus::~stylus() {
    for (int i = 0; i < e_inks.size(); i++){
        delete (e_inks[i]);
    }
    e_inks.clear();
    for (int i = 0; i < e_slices.size(); i++){
        for (int j = 0; j < e_slices[i].size(); j++){
            delete (e_slices[i][j]);
        }
    }
    e_slices.clear();
    for (int i = 0; i < e_pages.size(); i++){
        for (int j = 0; j < e_pages[i].size(); j++){
            for (int k = 0; k < e_pages[i][j].size(); k++){
                delete (e_pages[i][j][k]);
            }
        }
    }
    e_pages.clear();
    
    for (auto i = 0; i < m_trajs.size(); i++) {
        const auto& traj {m_trajs[i]};
        for (auto j = 0; j < traj->segments.size(); j++) {
            delete traj->segments[j];
        }
        traj->segments.clear();
        delete traj;
    }
    m_trajs.clear();
    
    for (auto i = 0; i < m_points.size(); i++) {
        const auto& traj {m_points[i]};
        for (auto j = 0; j < traj->segments.size(); j++) {
            delete traj->segments[j];
        }
        traj->segments.clear();
        delete traj;
    }
    m_points.clear();
    
    for (auto i = 0; i < m_plays.size(); i++) {
        const auto& traj {m_plays[i]};
        for (auto j = 0; j < traj->segments.size(); j++) {
            delete traj->segments[j];
        }
        traj->segments.clear();
        delete traj;
    }
    m_plays.clear();
    m_redraw_timer.stop();
    m_playhead_timer.stop();

    m_should_stop_render_thread = true;
    if (m_render_thread && m_render_thread->joinable()){
        m_render_thread->join();
    }
}

void stylus::terrain_render_loop(stylus *st) {
    using namespace std::chrono_literals;
    while (!st->m_should_stop_render_thread) {
        if (!st->m_should_render_lock.try_acquire_for(100ms)) {
            continue;
        }
        // re-arm: a terrain arriving while we build below will re-signal us
        st->m_render_pending.store(false);

        // take the latest snapshot handed over by the dictionary handler
        int mode;
        std::vector<std::vector<float>> in_mono;
        std::vector<std::array<std::vector<float>, 4>> in_color;
        float cmin, cmax;
        color blo, bhi;
        {
            std::unique_lock<std::mutex> lk(st->m_render_input_mutex);
            mode = st->m_in_mode;
            in_mono.swap(st->m_in_mono);
            in_color.swap(st->m_in_color);
            cmin = st->m_in_cmin;
            cmax = st->m_in_cmax;
            blo  = st->m_in_blo;
            bhi  = st->m_in_bhi;
        }

        // heavy part: scale()/colour math + canvas build, into local temporaries
        std::vector<std::vector<float>> tmp_mono;
        std::vector<std::vector<c74::max::t_jrgb>> tmp_color;

        if (mode == 1) {
            tmp_mono.resize(in_mono.size());
            for (size_t y = 0; y < in_mono.size(); y++) {
                const auto& row = in_mono[y];
                auto& out_row = tmp_mono[y];
                out_row.resize(row.size());
                for (size_t x = 0; x < row.size(); x++) {
                    out_row[x] = scale<float>(fmin(fmax(row[x], cmin), cmax), cmin, cmax, 0.0f, 1.0f);
                }
            }
        } else if (mode == 4) {
            const float lo_a = static_cast<float>(blo.alpha()), hi_a = static_cast<float>(bhi.alpha());
            const float lo_r = static_cast<float>(blo.red()),   hi_r = static_cast<float>(bhi.red());
            const float lo_g = static_cast<float>(blo.green()), hi_g = static_cast<float>(bhi.green());
            const float lo_b = static_cast<float>(blo.blue()),  hi_b = static_cast<float>(bhi.blue());
            tmp_color.resize(in_color.size());
            for (size_t y = 0; y < in_color.size(); y++) {
                const auto& ch = in_color[y];
                const size_t latents_size = ch[0].size();
                auto& out_row = tmp_color[y];
                out_row.resize(latents_size);
                for (size_t x = 0; x < latents_size; x++) {
                    float brightness = scale<float>(fmin(fmax(ch[0][x], cmin), cmax), cmin, cmax, lo_a, hi_a);
                    float r = scale<float>(fmin(fmax(ch[1][x], cmin), cmax), cmin, cmax, lo_r, hi_r);
                    float g = scale<float>(fmin(fmax(ch[2][x], cmin), cmax), cmin, cmax, lo_g, hi_g);
                    float b = scale<float>(fmin(fmax(ch[3][x], cmin), cmax), cmin, cmax, lo_b, hi_b);
                    out_row[x] = c74::max::t_jrgb{ r * brightness, g * brightness, b * brightness };
                }
            }
        } else {
            continue; // nothing valid to publish
        }

        // publish: O(1) swap under the canvas lock (read by the m_terrain draw lambda)
        {
            std::unique_lock<std::mutex> lk(st->m_canvas_mutex);
            if (mode == 1) {
                st->terrain_canvas.swap(tmp_mono);
            } else {
                st->terrain_canvas_c.swap(tmp_color);
            }
            st->terrain_loaded = mode;
        }
        st->m_finish_render_lock.release();
    }
}

void stylus::clear_trajs(tasks t, modes m){
    
    if (t == tasks::dataset){
        auto& type_traj = m == modes::audio ? m_trajs : m_points;
        for (auto i = 0; i < type_traj.size(); i++) {
            const auto& traj {type_traj[i]};
            for (auto j = 0; j < traj->segments.size(); j++) {
                delete traj->segments[j];
            }
            traj->segments.clear();
            traj->completed = false;
            traj->filled_latents = 1;
        }
        type_traj.clear();
//        traj_len_ms.clear();
//        latent_lens.clear();
        traj_ptr = 0;
    } else if (t == tasks::play){
        selected_traj = 0;
        for (auto i = 0; i < m_plays.size(); i++) {
            const auto& traj {m_plays[i]};
            for (auto j = 0; j < traj->segments.size(); j++) {
                delete traj->segments[j];
            }
            traj->segments.clear();
            delete traj;
        }
        m_plays.clear();
        traj_len_ms.clear();
//        m_traj_sel.send(selected_traj+1);
        m_traj_len.send(0.0f);
        m_traj_len_all.send(traj_len_ms);
    }
}


MIN_EXTERNAL(stylus);
