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
#include <algorithm>

using namespace c74::min;
using namespace c74::min::ui;

class min_multitouch : public object<min_multitouch>, public ui_operator<474, 245> {
private:
    vector<event> m_events;
    vector<symbol> m_event_phases;

    vector<vector<event>> m_pages;
    vector<vector<symbol>> m_pages_phases;

    number osc_1 {};

    number m_anchor_x {};
    number m_anchor_y {};
    number x_prev {};
    number y_prev {};
    number start_x {};
    number start_y {};

    float fading_speed = 0.3;

    string	m_text;
    attribute<symbol>  m_fontname{ this, "fontname", "lato-light" };
    attribute<number>  m_fontsize{ this, "fontsize", 8.0 };

    number pen_pressure {};
    boolean is_using_pen = false;
    boolean is_touching = false;
    boolean model_loaded = false;


    boolean is_logging = true;
    int log_page_count = 0;
    int log_point_count = 0;
    string patch_path = min_devkit_path();

    c74::min::path m_path;
    torch::jit::script::Module module;

    at::Tensor output_tensor = torch::zeros({1,8});

    numbers stroke_info = { 0.0, 0.0, 0.0, 0.0 };

public:
    MIN_DESCRIPTION{ "Pen sketching interface" };
    MIN_TAGS{ "ui, multitouch" }; // if multitouch tag is defined then multitouch is supported but mousedragdelta is not supported
    MIN_AUTHOR{ "Cycling '74" };
    MIN_RELATED{ "mousestate" };

    inlet<>     m_input{ this, "(anything) ignored" };
    inlet<>     m_osc{ this, "(float) ignored" };
    outlet<>    m_outlet_main{ this, "primary stuff: type phase x y modifiers" };
    outlet<>    m_outlet_pen{ this, "(float) pen pressure between 0. and 1." };
    outlet<>    m_outlet_index{ this, "int with index of the touch, not useful for mouse or pen" };
    outlet<>    m_outlet_module{ this, "outputs from the module inference" };

    void inference() {
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> test_inputs;
        test_inputs.push_back(torch::tensor({ { stroke_info[0], stroke_info[1], stroke_info[2], osc_1}}));

        // Execute the model and turn its output into a tensor.
        output_tensor = module.forward(test_inputs).toTensor();
    }


    min_multitouch(const atoms& args = {})
        : ui_operator::ui_operator{ this, args } {
        if (torch::hasCUDA()) {
            cout << "Torch has CUDA" << endl;
        }
        else {
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

            m_timer.delay(50);
            return {};
        }
    };
    ~min_multitouch() {
		m_timer.stop();
        cout << "destructed" << endl;
	}


    void send(symbol message_name, const event& e) {
        
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
        stroke_info[0] = scale(e.x(), d, e.target().width()-d, -3.0, 3.0);
        stroke_info[1] = scale(e.y(), 0.0, e.target().height(), -3.0, 3.0);
        stroke_info[2] = scale(e.pen_pressure(), 0.0, 1.0, -3.0, 3.0, 2.0);

        m_outlet_main.send(message_name, event_type, stroke_info[0], stroke_info[1], e.is_command_key_down(), e.is_shift_key_down());

        m_events.push_back(e);
        m_event_phases.push_back(message_name);
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
            if (args[0] == "none"){
                model_loaded = false;
                cout << "waiting to load model" << endl;
            } else {
                auto model_path = std::string(args[0]);
                
                m_path = path(model_path);
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
                test_inputs.push_back(torch::tensor({ { -0.8681, 3.5148, -1.6512, -5.0145} }));
                output_tensor = module.forward(test_inputs).toTensor();
                model_loaded = true;
                //m_outlet_module.send(output_tensor);
			}
            
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
					const auto& e {page[j]};
					const auto& phase { m_event_phases[j]};
                    src_content += std::to_string(i) + "," + std::to_string(static_cast<int>(e.x())) + "," + std::to_string(static_cast<int>(e.y())) + "," + std::to_string(e.pen_pressure()).substr(0, 5) + "," + phase.c_str() + "\n";

				}
			}

            for (int i = 0; i < m_events.size(); i++) {
				const auto& e {m_events[i]};
				const auto& phase { m_event_phases[i]};

                src_content += std::to_string(m_pages.size()) + "," + std::to_string(static_cast<int>(e.x())) + "," + std::to_string(static_cast<int>(e.y())) + "," + std::to_string(e.pen_pressure()).substr(0, 5) + "," + phase.c_str() + "\n";
                
            }
            string results = create_log_and_save(patch_path, src_content);
            cout << "saved log to: " << results << endl;
            return {};
        }
    };
    message<> m_clear{ this, "clear",
        MIN_FUNCTION {
            vector<event> new_e = m_events;
            vector<symbol> new_e_phase = m_event_phases;
            m_pages.push_back(new_e);
            m_pages_phases.push_back(new_e_phase);
            m_events.clear();
            m_event_phases.clear();
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
    message<> m_paint{ this, "paint",
        MIN_FUNCTION {
            target t { args };

            rect<fill> {	// background
                t,
                color { 0.2, 0.2, 0.2, 1.0 }
            };
            x_prev = 0.0;
            y_prev = 0.0;
            float ink = 1.0;
            for (auto i = m_events.size() ; i > 0; --i) {
                const auto& e {m_events[i-1]};

                const auto& phase { m_event_phases[i-1]};


                if ((phase == "up") || (phase == "drag")) {
                    is_using_pen = (e.type() == event::input_type::pen);
                } else if (phase == "down") {
                    ink -= fading_speed;
                }
                    //&& (e.type() == event::input_type::pen);

                bool permanent = (phase == "drag") || (phase == "down") || (phase == "up");
                number brightness = permanent ? 0.8 : 1.0;

                number radius;
                if (e.type() == event::input_type::touch)
                    radius = 30;
                else if (e.type() == event::input_type::pen)
                    radius = 2;
                else // input_type::mouse
                    radius = 4;

                color c {brightness, brightness, brightness, std::max(0.08f, ink)};

                m_anchor_x = e.x();
                m_anchor_y = e.y();

                if ((i == m_events.size()) || (phase == "up") ){
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
                }
                x_prev = m_anchor_x;
                y_prev = m_anchor_y;
            }

            for (auto i = m_events.size(); i > 0; --i) {
                const auto& phase {m_event_phases[i - 1]};
                bool permanent = (phase == "drag") || (phase == "down") || (phase == "up");
                //bool permanent = (phase == "drag") || (phase == "down");
                if (!permanent) {
                    m_events.erase(m_events.begin() + (i - 1));
                    m_event_phases.erase(m_event_phases.begin() + (i - 1));
                }
            }

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
            + " points: " + std::to_string(m_events.size())
                          + " path: " + patch_path;
    }

};


MIN_EXTERNAL(min_multitouch);
