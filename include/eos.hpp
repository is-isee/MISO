#pragma once
#include "config.hpp"

template <typename Real>
struct EOS {
    Real gm;

    EOS(const Config& config) : 
        gm(config.yaml_obj["eos"]["gm"].as<Real>()) {}
};