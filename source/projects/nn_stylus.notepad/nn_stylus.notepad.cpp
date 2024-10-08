/// @file
///	@ingroup 	minexamples
///	@copyright	Copyright 2020 The Min-DevKit Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.
//#include <windows.h>
//#include <winuser.h>



#include "c74_min.h"

#include "utils.h"
#include "pen.h"
#include "min_path.h"
#include <algorithm>

using namespace c74::min;
using namespace c74::min::ui;

class nn_notepad : public object<nn_notepad>, public ui_operator<948, 490> {
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
    attribute<bool>  is_fading{ this, "fade_history", true };
    attribute<color>  ink_color{ this, "ink_color", color{1.0,1.0,1.0,1.0} };
    //attribute<color>  m_background_color{ this, "background_color", color{0.1,0.1,0.1,1.0} };
    attribute<number> page_refresh{ this, "page_refresh", 1000 };

    number pen_pressure {};
    boolean is_using_pen = false;
    boolean is_touching = false;
    boolean model_loaded = false;

    float m_scale = 3.0f;
    number m_scale_num = 3.0f;

    boolean is_logging = true;
    attribute<bool>  display_log_path{ this, "display_log_path", true };
    attribute<symbol> note_name{ this, "note_name", "001" };
    int log_page_count = 0;
    int log_point_count = 0;
    string patch_path = min_devkit_path();

    min_path m_path;

    numbers stroke_info = { 0.0, 0.0, 0.0, 0.0, 0.0 };



public:
    MIN_DESCRIPTION{ "Pen sketching interface (no nn)" };
    MIN_TAGS{ "ui, multitouch, math" }; // if multitouch tag is defined then multitouch is supported but mousedragdelta is not supported
    MIN_AUTHOR{ "Cycling '74" };
    MIN_RELATED{ "mousestate" };

    inlet<>     m_input{ this, "(anything) ignored" };
    inlet<>     m_osc{ this, "(float) ignored" };

    outlet<>    m_outlet_main{ this, "primary stuff: type phase x y modifiers" };
    outlet<>    m_outlet_pen{ this, "(float) pen pressure between 0. and 1." };
    outlet<>    m_outlet_index{ this, "int with index of the touch, not useful for mouse or pen" };
    outlet<>    m_outlet_rect{ this, "rect width and height of the loaded background" };

    nn_notepad(const atoms& args = {})
        : ui_operator::ui_operator{ this, args } {
        m_timer.delay(1000);
    }
    timer<timer_options::defer_delivery> m_timer{ this,
        MIN_FUNCTION {
            
            if (is_logging && display_log_path) {
                update_text();
			}

            m_timer.delay(25);
            return {};
        }
    };
    ~nn_notepad() {
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
        stroke_info[0] = scale(e.x(), d, e.target().width()-d, -m_scale_num, m_scale_num);
        stroke_info[1] = scale(e.y(), 0.0, e.target().height(), -m_scale_num, m_scale_num);
        stroke_info[2] = scale(e.pen_pressure(), 0.0, 1.0, -3.0, 3.0, 2.0);

        m_outlet_main.send(message_name, event_type, stroke_info[0], stroke_info[1], e.is_command_key_down(), e.is_shift_key_down());

        m_events_slice.push_back(e);
        m_event_phases_slice.push_back(message_name);
        m_event_colors_slice.push_back(color{ ink_color.get().red(), ink_color.get().green(), ink_color.get().blue(), ink_color.get().alpha() });
        redraw();
    }

    // the actual attribute for the message
    //attribute<color>  m_background_fill{ this, "background_fill_colour", color{0.1,0.1,0.1,1.0}, description {
    //        "Background fill"
    //        "Colour of the background."
    //    },
    //    setter { MIN_FUNCTION{
    //        //redraw();
    //        return {args[0]};
    //        }} 
    //};

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
                m_outlet_rect.send(m_image.get_width(), m_image.get_height());
            }
            redraw();
            return {args[0]};
        }}
	};
    message<> m_save_log{ this, "log",
        MIN_FUNCTION {

            string src_content = "";
            
            for (int i = 0; i < m_pages.size(); i++) {
				const auto& page {m_pages[i]};
				for (int j = 0; j < page.size(); j++) {
                    const auto& event_slices{ page[j] };
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

            atoms results = create_log_and_save(std::to_string(note_name), patch_path, src_content);
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

            rect<fill> {	// background
                t,
                color{ 0.1,0.1,0.1,1.0 },
            };

            m_image.redraw(t.width(), t.height());
            m_image.draw(t, 0., 0., t.width(), t.height());

            if (is_logging && display_log_path) {
                text{			// text display
                    t, ink_color.get(),
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


MIN_EXTERNAL(nn_notepad);
