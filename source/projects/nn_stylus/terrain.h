


#pragma once


using namespace c74::min::ui;

class terrain {
public:
    terrain(object_base* an_owner, const double width, const double height, short this_path, const function& a_function = nullptr)
        : m_width{ width }
        , m_height{ height }
        , m_path{ this_path }
        , m_draw_callback{ a_function } {
    }

    ~terrain() {
        if (m_surface) {
            c74::max::jgraphics_surface_destroy(m_surface);
            m_surface = nullptr;
        }
    }


    void redraw(const int width, const int height, symbol file_name, long dpi) {
        auto old_surface = m_surface;
        m_surface = c74::max::jgraphics_image_surface_create(c74::max::JGRAPHICS_FORMAT_ARGB32, width, height);
        m_width = width;
        m_height = height;
        c74::max::t_jgraphics* ctx = jgraphics_create(m_surface);

        atoms a{ { ctx, m_width, m_height} };

        target t{ a };

        m_draw_callback(a, 0);
        c74::max::jgraphics_image_surface_draw(ctx, m_surface, { 0.0, 0.0, m_width, m_height }, { 0.0, 0.0, m_width, m_height });


        c74::max::jgraphics_image_surface_writepng(m_surface, file_name, m_path, dpi);
        visual_filename = file_name;


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


private:
    double					m_width;
    double					m_height;
    function        	    m_draw_callback;
    short				    m_path;
    c74::max::t_jsurface* m_surface{ nullptr };
    symbol				    visual_filename;     
};