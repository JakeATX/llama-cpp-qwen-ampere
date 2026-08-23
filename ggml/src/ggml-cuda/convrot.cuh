#pragma once

#include "common.cuh"

// Rotates f32 rows in groups of QK8_CR with the normalized Kronecker-Hadamard
// "ConvRot" matrix H_QK8_CR = kron(H4, ...) / sqrt(QK8_CR), matching the
// rotation baked into Q8_CR weights by llama-quantize. Applying it to the
// activations of a mul_mat cancels the weight rotation exactly.
void ggml_cuda_convrot_rotate(ggml_backend_cuda_context & ctx, const ggml_tensor * src, float * dst);
