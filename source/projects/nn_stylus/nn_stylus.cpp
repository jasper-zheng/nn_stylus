/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2020 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.
//#include <windows.h>
//#include <winuser.h>



#include "c74_min.h"
#include "torch/torch.h"
#include "torch/script.h"

#include "utils.h"
#include "pen.h"
#include "min_path.h"
#include <algorithm>

using namespace c74::min;
using namespace c74::min::ui;

class nn_stylus : public object<nn_stylus>, public ui_operator<948, 490> {
private:

    vector<event> m_events_slice;
    vector<symbol> m_event_phases_slice;
    vector<color> m_event_colors_slice;

    vector<vector<event>> m_events;
    vector<vector<symbol>> m_event_phases;
    vector<vector<color>> m_event_colors;

    vector<vector<vector<event>>> m_pages;
    vector<vector<vector<symbol>>> m_pages_phases;
    vector<vector<vector<color>>> m_pages_colors;

    vector<float> canvas_line;
    vector<vector<float>> canvas;

    boolean show_canvas = false;

    number osc_1 {};

    number m_anchor_x {};
    number m_anchor_y {};
    number x_prev {};
    number y_prev {};
    number start_x {};
    number start_y {};

    float fading_speed = 0.3;

    string	m_text;
    symbol m_fontname{ "lato-light" };
    attribute<number>  m_fontsize{ this, "fontsize", 8.0 };
    attribute<bool>  is_fading{ this, "fade history", true };
    attribute<color>  ink_color{ this, "ink color", color{1.0,1.0,1.0,1.0} };
    attribute<number> page_refresh{ this, "page refresh", 1000 };

    number pen_pressure {};
    boolean is_using_pen = false;
    boolean is_touching = false;
    boolean model_loaded = false;

    float m_scale = 3.0f;
    number m_scale_num = 3.0f;

    boolean is_logging = true;
    int log_page_count = 0;
    int log_point_count = 0;
    string patch_path = min_devkit_path();

    min_path m_path;
    torch::jit::script::Module module;

    at::Tensor output_tensor = torch::zeros({1,8});

    numbers stroke_info = { 0.0, 0.0, 0.0, 0.0, 0.0 };



public:
    MIN_DESCRIPTION{ "Pen sketching interface" };
    MIN_TAGS{ "ui, multitouch, math" }; // if multitouch tag is defined then multitouch is supported but mousedragdelta is not supported
    MIN_AUTHOR{ "Cycling '74" };
    MIN_RELATED{ "mousestate, jit.clip" };

    inlet<>     m_input{ this, "(anything) ignored" };
    inlet<>     m_osc{ this, "(float) ignored" };

    outlet<>    m_outlet_main{ this, "primary stuff: type phase x y modifiers" };
    outlet<>    m_outlet_pen{ this, "(float) pen pressure between 0. and 1." };
    outlet<>    m_outlet_index{ this, "int with index of the touch, not useful for mouse or pen" };
    outlet<>    m_outlet_module{ this, "outputs from the module inference" };

    void inference() {
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> test_inputs;

        // CAUTION
        //test_inputs.push_back(torch::tensor({ { stroke_info[0], stroke_info[1], 0.0, 0.0, osc_1}}));
        test_inputs.push_back(torch::tensor({ { stroke_info[0], stroke_info[1]} }));

        // Execute the model and turn its output into a tensor.
        output_tensor = module.forward(test_inputs).toTensor();
    }


    nn_stylus(const atoms& args = {})
        : ui_operator::ui_operator{ this, args } {
        if (torch::hasCUDA()) {
            cout << "Torch has CUDA" << endl;
        } else {
            cout << "Torch is using CPU" << endl;
        }
        m_timer.delay(1000);
    }
    timer<timer_options::defer_delivery> m_timer{ this,
        MIN_FUNCTION {
            if (model_loaded && is_touching)
                inference();
            
            torch::Tensor ten = output_tensor.to(torch::kFloat);
            atoms result(ten.data_ptr<float>(), ten.data_ptr<float>() + ten.numel());

            m_outlet_module.send(result);

            if (is_logging) {
                update_text();
			}

            m_timer.delay(25);
            return {};
        }
    };
    ~nn_stylus() {
		m_timer.stop();
        //if (m_surface) {
        //    c74::max::jgraphics_surface_destroy(m_surface);
        //    m_surface = nullptr;
        //}
        //if (m_paint_surface) {
        //    c74::max::jgraphics_surface_destroy(m_paint_surface);
        //    m_paint_surface = nullptr;
        //}
        cout << "destructed" << endl;
	}

    void send(symbol message_name, const event& e) {
        show_canvas = false;
        m_outlet_index.send(e.index());

        m_outlet_pen.send(e.pen_pressure());
        
        symbol event_type;
        if (e.type() == event::input_type::mouse)
            event_type = c74::max::gensym("mouse");
        else if (e.type() == event::input_type::touch)
            event_type = c74::max::gensym("touch");
        else if (e.type() == event::input_type::pen)
            event_type = c74::max::gensym("pen");
        else
            event_type = c74::max::gensym("unknown");

        number d = (e.target().width() - e.target().height())/2.0;
        stroke_info[0] = scale(e.x(), d, e.target().width()-d, -m_scale_num, m_scale_num);
        stroke_info[1] = scale(e.y(), 0.0, e.target().height(), -m_scale_num, m_scale_num);
        stroke_info[2] = scale(e.pen_pressure(), 0.0, 1.0, -3.0, 3.0, 2.0);

        m_outlet_main.send(message_name, event_type, stroke_info[0], stroke_info[1], e.is_command_key_down(), e.is_shift_key_down());

        m_events_slice.push_back(e);
        m_event_phases_slice.push_back(message_name);
        m_event_colors_slice.push_back(color{ ink_color.get().red(), ink_color.get().green(), ink_color.get().blue(), ink_color.get().alpha() });
        redraw();
    }

    argument<symbol> model_arg{ this, "model", "path to a torchscript file",
        MIN_ARGUMENT_FUNCTION {
            cout << "model read from: " << arg << endl;
        }
    };

    // the actual attribute for the message
    attribute<symbol> model{ this, "model", "none",
        description {
            "Model to be loaded."
            "The path to a torchscript file."
        },
        setter { MIN_FUNCTION{
            //target t { args };
            if (args[0] == "none"){
                model_loaded = false;
                cout << "waiting to load model" << endl;
            } else {
                auto model_path = std::string(args[0]);
                
                m_path = min_path(model_path);
                try {
                    // Deserialize the ScriptModule from a file using torch::jit::load().
                    module = torch::jit::load(std::string(m_path));
                }
                catch (const c10::Error& e) {
                    cerr << "error loading the model from " << m_path << endl;
                    return { -1 };
                }
                cout << "loaded model from " << std::string(m_path) << endl;
                std::vector<torch::jit::IValue> test_inputs;
                // CAUTION
                //test_inputs.push_back(torch::tensor({ { -0.8681, 3.5148, -1.6512, -5.0145, 0.0} }));
                test_inputs.push_back(torch::tensor({ { -0.8681, 3.5148} }));
                output_tensor = module.forward(test_inputs).toTensor();
                model_loaded = true;
                //m_outlet_module.send(output_tensor);
                int w = this->default_width();
                int h = this->default_height();
                float d = (w - h) / 2.0;

                cout << "compiling mapping: h: " << h << ", w: " << w << endl;
                canvas.clear();
                for (int i = 0; i < h; i++) {
                    std::vector<float> line;
                    float y = scale(static_cast<float>(i), 0.0f, static_cast<float>(h), -m_scale, m_scale);
					for (int j = 0; j < w; j++) {
						//line.push_back(0.0);
                        std::vector<torch::jit::IValue> test_inputs;
                        test_inputs.push_back(torch::tensor({ { scale(static_cast<float>(j), d, static_cast<float>(w - d), -m_scale, m_scale), y} }));
                        output_tensor = module.forward(test_inputs).toTensor();
                        torch::Tensor ten = output_tensor.to(torch::kFloat);
                        atoms result(ten.data_ptr<float>(), ten.data_ptr<float>() + 1);
                        line.push_back(scale(clamp(static_cast<float>(result[0]), -3.0f, 3.0f), -3.0f, 3.0f, 0.0f, 1.0f));
                    }
                    canvas.push_back(line);
				}
                show_canvas = true;
                cout << "mapping compiled: " << canvas.size() << ", " << canvas[0].size() << endl;
                redraw();
			}
            
            return {args[0]};
        }}
    };
    attribute<symbol> canvas_background{ this, "canvas_background", "none",
        description {
            "Background image"
            "The path to the background image."
        },
        setter { MIN_FUNCTION{
            if (args[0] == "none") {
                m_image.clear_background();
            } else {
                m_image.set_background(std::string(args[0]).c_str());
            }
            redraw();
            return {args[0]};
        }}
	};
    message<threadsafe::yes> m_ints{
    this, "int", "OSC information", MIN_FUNCTION {
        switch (inlet) {
            case 0:
                break;
            case 1:
                osc_1 = scale(static_cast<float>(args[0]), 0.0f, 1024.0f, -3.0f, 3.0f);
                break;
            default:
                assert(false);
        }
        return {};
    }
    };
    message<> m_save_log{ this, "log",
        MIN_FUNCTION {

            string src_content = "";
            
            for (int i = 0; i < m_pages.size(); i++) {
				const auto& page {m_pages[i]};
				for (int j = 0; j < page.size(); j++) {
                    const auto& event_slices{ page[j] };
                    //const auto& phase{ m_event_phases[j] };
                    for (int k = 0; k < event_slices.size(); k++) {
                        const auto& e{ event_slices[k] };
                    	const auto& phase{ m_event_phases_slice[k]};
                        src_content += std::to_string(i) + "," + std::to_string(static_cast<int>(e.x())) + "," + std::to_string(static_cast<int>(e.y())) + "," + std::to_string(e.pen_pressure()).substr(0, 5) + "," + phase.c_str() + "\n";
                    }
				}
			}

            for (int i = 0; i < m_events.size(); i++) {
				const auto& event_slices{m_events[i]};
                for (int k = 0; k < event_slices.size(); k++) {
                    const auto& e{ event_slices[k] };
                    const auto& phase{ m_event_phases_slice[k] };
                    src_content += std::to_string(m_pages.size()) + "," + std::to_string(static_cast<int>(e.x())) + "," + std::to_string(static_cast<int>(e.y())) + "," + std::to_string(e.pen_pressure()).substr(0, 5) + "," + phase.c_str() + "\n";
                }
            }

            for (int k = 0; k < m_events_slice.size(); k++) {
				const auto& e{ m_events_slice[k] };
				const auto& phase{ m_event_phases_slice[k] };
				src_content += std::to_string(m_pages.size()) + "," + std::to_string(static_cast<int>(e.x())) + "," + std::to_string(static_cast<int>(e.y())) + "," + std::to_string(e.pen_pressure()).substr(0, 5) + "," + phase.c_str() + "\n";
			}

            atoms results = create_log_and_save(patch_path, src_content);
            cout << "saved log to: " << results[0] << endl;
            m_image.write_and_lock(static_cast<string>(results[1])+".png", 150);
            return {};
        }
    };
    message<> m_clear{ this, "clear",
        MIN_FUNCTION {
            vector<event> new_e_slice = m_events_slice;
            vector<symbol> new_e_phase_slice = m_event_phases_slice;
            m_events.push_back(new_e_slice);
            m_event_phases.push_back(new_e_phase_slice);

            vector<vector<event>> new_e = m_events;
            vector<vector<symbol>> new_e_phase = m_event_phases;

            m_pages.push_back(new_e);
            m_pages_phases.push_back(new_e_phase);
            m_pages_colors.push_back(m_event_colors);

            m_events_slice.clear();
            m_event_phases_slice.clear();
            m_event_colors_slice.clear();
            m_events.clear();
            m_event_phases.clear();
            m_event_colors.clear();

            m_image.clear_history();

            update_text();
            redraw();
            return {};
        }
    };
    message<> m_mouseenter{ this, "mouseenter",
        MIN_FUNCTION {
            send("enter", args);
            return {};
        }
    };
    message<> nameaaa{ this, "hello",
        MIN_FUNCTION {
            cout << "nameaaa" << endl;
            return {};
        }
    };
    message<> m_mousemove{ this, "mousemove",
        MIN_FUNCTION {
            //cout << "mousemove" << endl;
            send("move", args);
            return {};
        }
    };
    message<> m_mouseleave{ this, "mouseleave",
        MIN_FUNCTION {
            send("leave", args);
            return {};
        }
    };
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
    message<> m_mousedrag{ this, "mousedrag",
        MIN_FUNCTION {
            //cout << "mousedrag" << endl;
            send("drag", args);
            return {};
        }
    };
    message<> m_mousedragdelta{ this, "mousedragdelta",
        MIN_FUNCTION {
            cout << "mousedragdelta" << endl;
            send("drag", args);
            return {};
        }
    };
    message<> m_hello{ this, "hello",
        MIN_FUNCTION {
            cout << "hello" << endl;
            send("drag", args);
            return {};
        }
    };

    // ---------------------- image

    pen m_image{ this,100.0,100.0, min_path("locator").get_path(), MIN_FUNCTION{
        target t { args };

        x_prev = 0.0; 
        y_prev = 0.0;
        float ink = 1.0;

        for (auto i = m_events_slice.size() ; i > 0; --i) {

            const auto& e {m_events_slice[i-1]};
            const auto& phase { m_event_phases_slice[i-1]};
            const auto& e_color { m_event_colors_slice[i-1]};

            if ((phase == "up") || (phase == "drag")) {
                is_using_pen = (e.type() == event::input_type::pen);
            } else if (phase == "down" && is_fading) {
                ink -= fading_speed;
            }

            bool permanent = (phase == "drag") || (phase == "down") || (phase == "up");
            number brightness = permanent ? 1.0 : 0.8;

            number radius;
            if (e.type() == event::input_type::touch)
                radius = 10;
            else if (e.type() == event::input_type::pen)
                radius = 2;
            else // input_type::mouse
                radius = 4;

            color c { e_color.red(), e_color.green(), e_color.blue(), std::max(0.08f, ink)};
            m_anchor_x = e.x();
            m_anchor_y = e.y();

            if ((i == m_events_slice.size()) || (phase == "up") ){
                start_x = m_anchor_x;
                start_y = m_anchor_y;
            }
            else {
                start_x = x_prev;
                start_y = y_prev;
            }

            number line_width_now = is_using_pen ? 0.1+4.0*e.pen_pressure() : 1.0;

            if (permanent) {
                line<> {
                    t,
                    color{ c },
                    position{ start_x, start_y },
                    destination{ m_anchor_x, m_anchor_y },
                    line_width{ line_width_now }
                };
            } else {
                ellipse<fill> {
                    t,
                    color{ c },
                    position{ m_anchor_x - radius, m_anchor_y - radius },
                    size{ radius * 2, radius * 2 }
                };
                m_events_slice.erase(m_events_slice.begin() + (i - 1));
                m_event_phases_slice.erase(m_event_phases_slice.begin() + (i - 1));
                m_event_colors_slice.erase(m_event_colors_slice.begin() + (i - 1));
            }
            x_prev = m_anchor_x;
            y_prev = m_anchor_y;
        }

        if (m_events_slice.size() >= page_refresh) {
            m_event_phases_slice[m_events_slice.size() - 1] = "up";

            m_events.push_back(m_events_slice);
            m_event_phases.push_back(m_event_phases_slice);
            m_event_colors.push_back(m_event_colors_slice); 

            m_image.write_and_lock("export24aa.png", 72);

            m_events_slice.clear();
            m_event_phases_slice.clear();
            m_event_colors_slice.clear();

            m_events_slice.push_back(m_events[m_events.size() - 1][m_events[m_events.size() - 1].size() - 1]);
            m_event_phases_slice.push_back("down");
            m_event_colors_slice.push_back(m_event_colors[m_event_colors.size() - 1][m_event_colors[m_event_colors.size() - 1].size() - 1]);

            m_image.draw_history("export24aa.png");
        }
        return{ {ink} };
    } };

    message<> m_paint{ this, "paint",
        MIN_FUNCTION {
            target t { args };
            //m_paint_target = std::make_unique<target>(args);
            if (show_canvas) {
                for (int i = 0; i < t.height(); i++) {
                    for (int j = 0; j < t.width(); j++) {
                        float v = canvas[i][j];
						rect<fill> {
							t,
							color{ v, v, v, 1.0},
							position{ j, i },
							size{ 1.0, 1.0 }
						};
                    }
                }
            }
            else {
                rect<fill> {	// background
                    t,
                    color{ 0.2, 0.2, 0.2, 1.0 }
                };
            }

            m_image.redraw(t.width(), t.height());
            m_image.draw(t, 0., 0., t.width(), t.height());

            if (is_logging) {
                text{			// text display
                    t, color {color::predefined::white},
                    position {10.0, t.height() - m_fontsize * 0.5 - 3.0},
                    fontface {m_fontname},
                    fontsize {m_fontsize},
                    content {m_text}
                };
			}
            return {};
        }
    };


    void update_text() {
        m_text = "logging - pages: " + std::to_string(m_pages.size())
            + " points: " + std::to_string(m_events_slice.size())
                          + " path: " + patch_path;
    }

};


MIN_EXTERNAL(nn_stylus);
