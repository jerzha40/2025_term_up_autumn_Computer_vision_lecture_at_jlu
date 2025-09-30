// einsum.cpp — 最小可跑：仅支持 "ij,jk->ik"（float32）
// 依赖：cuTENSOR 2.3+，CUDA 运行时，perco::Tensor（CPU 连续内存 + strides）

#include <cutensor.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream> // 仅用于错误字符串
#include "perco/core/tensor.h"

namespace perco
{

// ---- 小工具：错误检查 ----
#define CHECK_CUDA(expr)                                           \
    do                                                             \
    {                                                              \
        cudaError_t __e = (expr);                                  \
        if (__e != cudaSuccess)                                    \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(__e));     \
    } while (0)

#define CHECK_CUTENSOR(expr)                                           \
    do                                                                 \
    {                                                                  \
        cutensorStatus_t __s = (expr);                                 \
        if (__s != CUTENSOR_STATUS_SUCCESS)                            \
            throw std::runtime_error(std::string("cuTENSOR error: ") + \
                                     cutensorGetErrorString(__s));     \
    } while (0)

    // 仅支持两个输入、"ij,jk->ik"
    Tensor einsum(const std::string &spec, const Tensor &A, const Tensor &B)
    {
        // 0) 校验与输出形状
        if (spec != "ij,jk->ik")
            throw std::invalid_argument("perco::einsum: only supports \"ij,jk->ik\" for now");
        const auto &sA = A.shape();
        const auto &sB = B.shape();
        if (sA.size() != 2 || sB.size() != 2)
            throw std::invalid_argument("perco::einsum: A and B must be 2D for \"ij,jk->ik\"");
        if (sA[1] != sB[0])
            throw std::invalid_argument("perco::einsum: A.shape[1] must equal B.shape[0]");

        const int64_t M = static_cast<int64_t>(sA[0]);
        const int64_t K = static_cast<int64_t>(sA[1]);
        const int64_t N = static_cast<int64_t>(sB[1]);
        (void)K;

        // 模式标签（int32）
        std::vector<int32_t> modesA = {'i', 'j'};
        std::vector<int32_t> modesB = {'j', 'k'};
        std::vector<int32_t> modesC = {'i', 'k'};

        // 1) 输出 Tensor（CPU 端）
        Tensor C({static_cast<uint64_t>(M), static_cast<uint64_t>(N)});

        // 2) cuTENSOR 句柄
        cutensorHandle_t handle = nullptr;
        CHECK_CUTENSOR(cutensorCreate(&handle));

        // 3) extents/strides（单位=元素）
        std::vector<int64_t> extA = {static_cast<int64_t>(sA[0]), static_cast<int64_t>(sA[1])};
        std::vector<int64_t> extB = {static_cast<int64_t>(sB[0]), static_cast<int64_t>(sB[1])};
        std::vector<int64_t> extC = {M, N};

        auto vStrA = A.strides();
        auto vStrB = B.strides();
        auto vStrC = C.strides();
        std::vector<int64_t> strA(vStrA.begin(), vStrA.end());
        std::vector<int64_t> strB(vStrB.begin(), vStrB.end());
        std::vector<int64_t> strC(vStrC.begin(), vStrC.end());

        cutensorTensorDescriptor_t descA = nullptr;
        cutensorTensorDescriptor_t descB = nullptr;
        cutensorTensorDescriptor_t descC = nullptr;

        const uint32_t alignment = 256; // 与 cudaMalloc 对齐一致，简单稳妥
        CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle, &descA,
                                                      2, extA.data(), strA.data(), CUTENSOR_R_32F, alignment));
        CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle, &descB,
                                                      2, extB.data(), strB.data(), CUTENSOR_R_32F, alignment));
        CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle, &descC,
                                                      2, extC.data(), strC.data(), CUTENSOR_R_32F, alignment));

        // 4) Contraction 描述（A×B→C；D 用 C 占位）
        cutensorOperationDescriptor_t op = nullptr;
        CHECK_CUTENSOR(cutensorCreateContraction(handle, &op,
                                                 descA, modesA.data(), CUTENSOR_OP_IDENTITY,
                                                 descB, modesB.data(), CUTENSOR_OP_IDENTITY,
                                                 descC, modesC.data(), CUTENSOR_OP_IDENTITY,
                                                 descC, modesC.data(),
                                                 CUTENSOR_COMPUTE_DESC_32F));

        // 5) 计划与工作区
        cutensorPlanPreference_t pref = nullptr;
        CHECK_CUTENSOR(cutensorCreatePlanPreference(handle, &pref,
                                                    CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

        uint64_t workspaceEstimate = 0;
        CHECK_CUTENSOR(cutensorEstimateWorkspaceSize(handle, op, pref,
                                                     CUTENSOR_WORKSPACE_DEFAULT, &workspaceEstimate));

        cutensorPlan_t plan = nullptr;
        CHECK_CUTENSOR(cutensorCreatePlan(handle, &plan, op, pref, workspaceEstimate));

        uint64_t workspaceRequired = 0;
        CHECK_CUTENSOR(cutensorPlanGetAttribute(handle, plan,
                                                CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceRequired, sizeof(workspaceRequired)));

        void *workspace = nullptr;
        if (workspaceRequired)
        {
            // 尝试分配；失败则回退为 0 工作区
            if (cudaMalloc(&workspace, workspaceRequired) != cudaSuccess)
            {
                workspace = nullptr;
                workspaceRequired = 0;
            }
        }

        // 6) 设备内存（把 CPU Tensor 的数据搬到 GPU）
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        size_t bytesA = sizeof(float) * static_cast<size_t>(A.numel());
        size_t bytesB = sizeof(float) * static_cast<size_t>(B.numel());
        size_t bytesC = sizeof(float) * static_cast<size_t>(C.numel());
        CHECK_CUDA(cudaMalloc(&dA, bytesA));
        CHECK_CUDA(cudaMalloc(&dB, bytesB));
        CHECK_CUDA(cudaMalloc(&dC, bytesC));
        CHECK_CUDA(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dC, 0, bytesC));

        // 7) 执行
        float alpha = 1.f, beta = 0.f;
        cudaStream_t stream = nullptr; // 默认流
        cutensorStatus_t st = cutensorContract(handle, plan,
                                               &alpha, dA, dB,
                                               &beta, dC, dC,
                                               workspace, workspaceRequired, stream);

        // 同步 & 拷回结果
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(C.data(), dC, bytesC, cudaMemcpyDeviceToHost));

        // 8) 释放
        if (workspace)
            cudaFree(workspace);
        if (dC)
            cudaFree(dC);
        if (dB)
            cudaFree(dB);
        if (dA)
            cudaFree(dA);

        if (plan)
            cutensorDestroyPlan(plan);
        if (pref)
            cutensorDestroyPlanPreference(pref);
        if (op)
            cutensorDestroyOperationDescriptor(op);
        if (descC)
            cutensorDestroyTensorDescriptor(descC);
        if (descB)
            cutensorDestroyTensorDescriptor(descB);
        if (descA)
            cutensorDestroyTensorDescriptor(descA);
        if (handle)
            cutensorDestroy(handle);

        if (st != CUTENSOR_STATUS_SUCCESS)
            throw std::runtime_error("cuTENSOR contract failed");

        return C;
    }

} // namespace perco
