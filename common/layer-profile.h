#pragma once

#include "ggml-backend.h"

#include <memory>
#include <string>

struct common_params;

struct common_layer_profile_config {
    std::string path;
    std::string detail = "summary";
    std::string sync = "layer";
    int warmup = 0;
    int max_tokens = -1;
};

bool common_layer_profile_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

struct common_layer_profile_user_data {
    struct impl;
    std::unique_ptr<impl> pimpl;

    common_layer_profile_user_data();
    ~common_layer_profile_user_data();

    common_layer_profile_user_data(const common_layer_profile_user_data &) = delete;
    common_layer_profile_user_data & operator=(const common_layer_profile_user_data &) = delete;

    common_layer_profile_user_data(common_params & params, const common_layer_profile_config & config);

    bool ok() const;
};
