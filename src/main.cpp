#include <exception>
#include <iostream>
#include <systemc>

#include "simulator/run.hpp"

int sc_main(int argc, char* argv[]) {
    try {
        return simulator::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
