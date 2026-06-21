cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
endif()

project(nn_terrain)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()
    message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
endif()

add_subdirectory(backend) # DEEP LEARNING BACKEND

# if ((NOT ("${PUREDATA_INCLUDE_DIR}" STREQUAL "")) OR (UNIX AND NOT APPLE))
#     add_subdirectory(frontend/puredata/nn_terrain) # PURE DATA EXTERNAL
# endif()

if(APPLE OR MSVC)
    add_subdirectory(frontend/maxmsp/nn.terrain_tilde) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mc.nn.terrain_tilde) # MAX MSP EXTERNAL (multichannel, batch on channels)
    add_subdirectory(frontend/maxmsp/mcs.nn.terrain_tilde) # MAX MSP EXTERNAL (multichannel, batch on inlets)
    add_subdirectory(frontend/maxmsp/nn.terrain.encode) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/nn.terrain.record) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/nn.terrain.gui) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/nn.terrain.mc.gui)
endif()