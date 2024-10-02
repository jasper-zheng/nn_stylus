


#pragma once


using namespace c74::min::ui;

class pen {
public:
    pen(object_base* an_owner, const double width, const double height, short this_path, const function& a_function = nullptr)
        : m_width{ width }
        , m_height{ height }
        , m_path{ this_path }
        , m_draw_callback{ a_function } {
    }

    ~pen() {
        if (m_surface) {
            c74::max::jgraphics_surface_destroy(m_surface);
            m_surface = nullptr;
        }
        if (m_history_surface) {
            c74::max::jgraphics_surface_destroy(m_history_surface);
            m_history_surface = nullptr;
        }
        if (m_background_surface) {
            c74::max::jgraphics_surface_destroy(m_background_surface);
            m_background_surface = nullptr;
        }

    }

    void draw_history(const char* filename) {
        m_history_surface = c74::max::jgraphics_image_surface_create_from_file(filename, m_path);
	}
    void clear_history() {
        m_history_surface = c74::max::jgraphics_image_surface_create(c74::max::JGRAPHICS_FORMAT_ARGB32, m_width, m_height);
    }

    void redraw(const int width, const int height) {
        auto old_surface = m_surface;
        m_surface = c74::max::jgraphics_image_surface_create(c74::max::JGRAPHICS_FORMAT_ARGB32, width, height);
        m_width = width;
        m_height = height;
        c74::max::t_jgraphics* ctx = jgraphics_create(m_surface);

        //-----------------------------------
        //color c{ 1.0, 1.0, 1.0, 1.0};
        //c74::max::jgraphics_set_source_jrgba(ctx, c);
        //c74::max::jgraphics_rectangle_rounded(ctx, 10, 10, 100, 100, 0, 0);
        //max::jgraphics_fill(ctx);
        // 
        //rect<fill> {
        //    t,
        //        color{ 1.0, 1.0, 1.0, 0.8 },
        //        position{ 0.0, 0.0 },
        //        size{ 160.0, 120.0 }
        //};
        //-----------------------------------

        atoms a{ { ctx, m_width, m_height} };
        
        target t{ a };

        
        // here
        if (m_has_background) {
			color c{ 1.0, 1.0, 1.0, bg_alpha };
			c74::max::jgraphics_set_source_jrgba(ctx, c);
			c74::max::jgraphics_image_surface_draw(ctx, m_background_surface, { 0.0, 0.0, bg_width, bg_height }, { 0.0, 0.0, m_width, m_height });
		}
        if (m_history_surface) {
            //color c{ 1.0, 1.0, 1.0, std::max(0.7, m_history_ink) };
            color c{ 1.0, 1.0, 1.0, 1.0 };
            c74::max::jgraphics_set_source_jrgba(ctx, c);
            c74::max::jgraphics_image_surface_draw(ctx, m_history_surface, { 0.0, 0.0, m_width, m_height }, { 0.0, 0.0, m_width, m_height });
        }

        m_history_ink = m_draw_callback(a, 0)[0];

        c74::max::jgraphics_destroy(ctx);

        if (old_surface) {
            c74::max::jgraphics_surface_destroy(old_surface);
		}
    }

    void draw(ui::target& t, const double x, const double y, const double width, const double height) {
        c74::max::jgraphics_image_surface_draw(t, m_surface, { 0.0, 0.0, m_width, m_height }, { x, y, width, height });
    }

    void write_and_lock(symbol file_name, long dpi) {
        c74::max::jgraphics_image_surface_writepng(m_surface, file_name, m_path, dpi);
	}

    void set_background(const char* filename) {
		m_background_surface = c74::max::jgraphics_image_surface_create_from_file(filename, m_path);
        bg_height = static_cast<int>(c74::max::jgraphics_image_surface_get_height(m_background_surface));
        bg_width = static_cast<int>(c74::max::jgraphics_image_surface_get_width(m_background_surface));
		m_has_background = true;
	}

    void clear_background() {
        m_background_surface = c74::max::jgraphics_image_surface_create(c74::max::JGRAPHICS_FORMAT_ARGB32, m_width, m_height);
        m_has_background = false;
    }

private:
    double					m_width;
    double					m_height;
    function        	    m_draw_callback;
    short				    m_path;
    bool					m_has_background{ false };
    c74::max::t_jsurface*   m_surface{ nullptr };
    c74::max::t_jsurface*   m_background_surface{ nullptr };
    double  			    bg_width{ 0.0 };
    double  			    bg_height{ 0.0 };
    double  			    bg_alpha{ 1.0 };
    c74::max::t_jsurface*   m_history_surface{ nullptr };
    double  			    m_history_ink{ 0.0 };

};