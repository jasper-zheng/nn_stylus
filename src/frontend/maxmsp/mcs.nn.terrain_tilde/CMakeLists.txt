# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

set ( MSVC_COMPILER_NAME "MSVC" )
if (${CMAKE_CXX_COMPILER_ID} STREQUAL ${MSVC_COMPILER_NAME})
    string (SUBSTRING ${CMAKE_CXX_COMPILER_VERSION} 0 4 MSVC_VERSION_SHORT)
    string (SUBSTRING ${CMAKE_CXX_COMPILER_VERSION} 0 2 MSVC_VERSION_MAJOR)
    string (SUBSTRING ${CMAKE_CXX_COMPILER_VERSION} 3 1 MSVC_VERSION_MINOR)

    if (${MSVC_VERSION_MAJOR} VERSION_LESS 19 OR ${MSVC_VERSION_MAJOR} MATCHES 19 AND ${MSVC_VERSION_MINOR} VERSION_LESS 1)
           # message(STATUS "Visual Studio ${MSVC_VERSION_SHORT} detected. Visual Studio 17 (19.1) or greater is required for UI objects.")
         message(STATUS "Visual Studio 17 or greater is required for UI objects.")
          message(STATUS "SKIPPING!")
          return ()
    endif ()
endif ()

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()

#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}"
)
find_package(Torch REQUIRED)

set(
SOURCE_FILES
	mcs.nn.terrain_tilde.cpp
    	"../shared/circular_buffer.h"
	"../shared/utils.h"
	"../shared/min_path.h"
	"../shared/min_dictionary.h"
)

if (UNIX)
	execute_process(
			COMMAND git describe --tags
			WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
			OUTPUT_VARIABLE VERSION
			OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	message(STATUS ${VERSION})
	add_definitions(-DVERSION="${VERSION}")
endif()

add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
    	"../shared/circular_buffer.h"
	"../shared/utils.h"
	"../shared/min_path.h"
	"../shared/min_dictionary.h"
)
target_link_libraries(${PROJECT_NAME} PRIVATE backend)

if (APPLE)
	target_link_libraries(${PROJECT_NAME} PUBLIC "$<LINK_LIBRARY:FRAMEWORK,CoreFoundation>")
	set_target_properties(${PROJECT_NAME} PROPERTIES
	BUILD_WITH_INSTALL_RPATH FALSE
	LINK_FLAGS "-Wl,-rpath,@loader_path/"
	)
	# I can't figure out how to set the Bundle Identifier nicely in for max-sdk-base, so I'm just gonna do this thing here...
	set(AUTHOR_DOMAIN "com.jasperzheng")
	set(BUNDLE_IDENTIFIER "mcs-nn-terrain-")
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

if (APPLE) # COPY TORCH DYLIB IN THE LOADER FOLDER
add_custom_command( 
		TARGET ${PROJECT_NAME} 
		POST_BUILD 
		COMMAND cp "${TORCH_INSTALL_PREFIX}/lib/*.dylib" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/" 
		COMMENT "Copy Torch Libraries" 
	)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
	add_custom_command( 
		TARGET ${PROJECT_NAME} 
		POST_BUILD 
		COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
		COMMENT "Codesign external" 
	)
endif()
endif()

if (MSVC) # COPY TORCH DLL IN THE LOADER FOLDER
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TORCH_INSTALL_PREFIX}/lib/" ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
)
endif()

# This external uses C++20 features (e.g. std::binary_semaphore). min-posttarget
# pins CXX_STANDARD to 17, so override it here for every platform (previously this
# was MSVC-only, which broke macOS builds on newer toolchains).
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
