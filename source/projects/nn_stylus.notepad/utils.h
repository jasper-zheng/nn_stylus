
#include <filesystem>
#include <chrono>
#include <ctime>

#include "c74_min_api.h"


using namespace c74::min;


atoms create_log_and_save(const std::string& save_name, const std::string& base_path, const std::string& input) {

    std::filesystem::create_directory(base_path + "/logs");

    // Get the current timestamp
    std::time_t timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    // Convert the timestamp to a string
    string timestamp_str = std::ctime(&timestamp);

    // Remove newline character from the timestamp string
    timestamp_str.erase(timestamp_str.find_last_not_of("\n") + 1);


    for (char& c : timestamp_str) {
        if (c == ':') {
            c = '-';
        }
    }
    // Use the timestamp as the file name
    string file_name = save_name + " note log - " + timestamp_str;
    string src_path_str = base_path + "/logs/" + file_name + ".txt";

    string src_content;
    {
        std::ifstream in{ src_path_str };
        src_content = string{ std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>() };
        //src_content += input + " - " + timestamp_str;
    }
    {
        std::ofstream out{ src_path_str };
        out << input;
        out.close();
    }
    
    return { {src_path_str, file_name} };
}

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
    CFBundleRef this_bundle = CFBundleGetBundleWithIdentifier(CFSTR("com.cycling74.min-project"));
    CFURLRef    this_url = CFBundleCopyExecutableURL(this_bundle);
    char        this_path[4096];

    CFURLGetFileSystemRepresentation(this_url, true, reinterpret_cast<UInt8*>(this_path), 4096);

    string this_path_str{ this_path };

    CFRelease(this_url);
    // remember: we don't want to release the bundle because Max is still using it!

    // we now have a path like this:
    // /Users/tim/Materials/min-devkit/externals/min.project.mxo/Contents/MacOS/min.project"
    // so we need to chop off 5 slashes from the end

    auto iter = this_path_str.find("/externals/min.project.mxo/Contents/MacOS/min.project");
    this_path_str.erase(iter, strlen("/externals/min.project.mxo/Contents/MacOS/min.project"));
    return this_path_str;
#endif    // MAC_VERSION
}