#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Side = 2;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Format8UNorm = 1;
constexpr std::uint32_t Format8_8UNorm = 14;
constexpr std::uint32_t Format1616UNorm = 23;
constexpr std::uint32_t Format9995Float = 132;
constexpr std::uint32_t Format8Srgb = 128;
constexpr std::uint32_t Format8_8Srgb = 129;
constexpr std::uint32_t Format8888Srgb = 130;
constexpr std::uint32_t Format8888UInt = 60;
constexpr std::uint32_t Format32Float = 22;
constexpr std::uint32_t Format16161616Float = 71;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Type2DArray = 13;
constexpr std::uint32_t TypeCube = 11;
constexpr std::uint32_t Layers = 4;
constexpr std::uint32_t LessEqual = 3;
constexpr std::uint32_t Greater = 4;
constexpr std::uint32_t NotEqual = 5;
constexpr std::uint32_t GreaterEqual = 6;
constexpr std::uint32_t ClampWrap = 0;
constexpr std::uint32_t ClampMirror = 1;
constexpr std::uint32_t ClampEdge = 2;
constexpr std::uint32_t ClampHalfBorder = 4;
constexpr std::uint32_t ClampBorder = 6;
constexpr std::uint32_t BorderBlack = 0;
constexpr std::uint32_t BorderWhite = 2;
constexpr std::uint32_t BorderTable = 3;
constexpr std::uint32_t FilterPoint = 0;
constexpr std::uint32_t FilterBilinear = 1;
constexpr std::uint32_t FilterAnisoPoint = 2;
constexpr std::uint32_t FilterAnisoBilinear = 3;
constexpr std::uint32_t ReductionMin = 1;
constexpr std::array<std::array<std::uint8_t, 4>, Layers> Red{{{0, 64, 128, 255}, {230, 240, 250, 255}, {5, 15, 25, 35}, {200, 220, 210, 245}}};
constexpr std::array<float, 4> HalfRed{0.25f, std::numeric_limits<float>::quiet_NaN(), 0.5f, 0.0f};
constexpr std::array<std::uint16_t, 4> HalfRedBits{0x3400u, 0x7e00u, 0x3800u, 0x0000u};

alignas(256) std::array<float, Threads * 3> Input{};
alignas(256) std::array<float, Threads * 4> InputArray{};
alignas(256) std::array<float, Threads * 3> InputNaN{};
alignas(256) std::array<float, Threads * 4> CubeInput{};
alignas(256) std::array<float, Threads * 4> Output{};
std::uint32_t Swizzle = 0xfacu;
alignas(256) std::array<std::uint32_t, Threads * 3> Extra{};
alignas(4096) std::array<std::uint8_t, 65536> Texels{};
alignas(4096) std::array<std::uint8_t, 4096> HalfTexels{};

alignas(256) constexpr std::array<std::uint32_t, 13> Code{
    0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0bc0108, 0x00820401, 0xbf8c3f70,
    0x34160082, 0xe0701000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 13> ArrayCode{
    0x1614008c, 0xe03c1000, 0x8000010a, 0xbf8c3f70, 0xf0bc0128, 0x00820401, 0xbf8c3f70,
    0x34160082, 0xe0701000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 13> LayerCode{
    0x16140090, 0xe0381000, 0x8000010a, 0xbf8c3f70, 0xf0bc0128, 0x00820501, 0xbf8c3f70,
    0x34160082, 0xe0701000, 0x8001050b, 0xbf810000, 0xbf810000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 13> CubeCode{
    0x16140090, 0xe0381000, 0x8000010a, 0xbf8c3f70, 0xf0bc0118, 0x00820401, 0xbf8c3f70,
    0x34160082, 0xe0701000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 16> OffsetCode{
    0x1614008c, 0xe03c1000, 0x8000030a, 0xe03c1000, 0x8005060a, 0xbf8c3f70, 0x7e020306, 0x7e040307,
    0x7e0c0308, 0xf0f80108, 0x00820901, 0xbf8c3f70, 0x34160082, 0xe0701000, 0x8001090b, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 17> QuadCode{
    0x1614008c, 0xe03c1000, 0x8000010a, 0x7e0802f3, 0x7e0a02f3, 0x7e0c02f3, 0x7e0e02f3, 0xbf8c3f70, 0xf0bc0f08, 0x00820401, 0xbf8c3f70, 0x34160084, 0xe0781000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 17> PairCode{
    0x1614008c, 0xe03c1000, 0x8000010a, 0x7e0802f3, 0x7e0a02f3, 0x7e0c02f3, 0x7e0e02f3, 0xbf8c3f70, 0xf0bc0608, 0x00820401, 0xbf8c3f70, 0x34160084, 0xe0781000, 0x8001040b, 0xbf810000, 0xbf810000, 0xbf810000,
};

struct Sampler {
    std::uint32_t clamp;
    std::uint32_t filter;
    std::uint32_t border = BorderBlack;
    std::uint32_t reduction = 0;
    std::uint32_t lodBias = 0;
    bool disableCubeWrap = false;
    std::uint32_t function = LessEqual;
    bool truncCoord = false;
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(std::uint32_t format, bool cube = false, bool layered = false) {
    const void* base = format == Format16161616Float ? static_cast<const void*>(HalfTexels.data()) : static_cast<const void*>(Texels.data());
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        Swizzle | ((cube ? TypeCube : layered ? Type2DArray : Type2D) << 28u),
        cube ? 5u : layered ? Layers - 1u : 0u, 0u, 0u, 0u,
    };
}

std::array<std::uint32_t, 4> SamplerDescriptor(const Sampler& sampler) {
    return {
        sampler.clamp | (sampler.clamp << 3u) | (ClampEdge << 6u) | (sampler.function << 12u) | (sampler.truncCoord ? (1u << 27u) : 0u) | (sampler.reduction << 29u) | (sampler.disableCubeWrap ? (1u << 28u) : 0u),
        0u,
        sampler.lodBias | (sampler.filter << 20u) | (sampler.filter << 22u),
        sampler.border << 30u,
    };
}

void FillTexels() {
    Texels.fill(0);
    const auto descriptor = TextureDescriptor(Format1616UNorm, false, true);
    const auto surface = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(descriptor));
    Require(surface.guestBytes <= Texels.size(), "array descriptor did not expose its layers within texture storage");
    const auto& mip = surface.mips.at(0);
    for (std::uint32_t layer = 0; layer < Layers; ++layer) {
        for (std::uint32_t y = 0; y < Side; ++y) {
            for (std::uint32_t x = 0; x < Side; ++x) {
                auto* texel = Texels.data() + surface.GuestLayerOffset(layer) + mip.tiledOffset + y * mip.pitchBytes + x * 4u;
                texel[0] = Red[layer][y * Side + x];
                texel[1] = Red[layer][y * Side + x];
                texel[2] = 0x22;
                texel[3] = 0xff;
            }
        }
    }
    HalfTexels.fill(0);
    const auto half = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(TextureDescriptor(Format16161616Float)));
    const auto& halfMip = half.mips.at(0);
    for (std::uint32_t y = 0; y < Side; ++y) {
        for (std::uint32_t x = 0; x < Side; ++x) {
            const auto offset = halfMip.tiledOffset + y * halfMip.pitchBytes + x * 8u;
            Require(offset + 8u <= HalfTexels.size(), "the half float test surface does not fit its buffer");
            const std::array<std::uint16_t, 4> channels{HalfRedBits[y * Side + x], 0u, 0u, 0x3c00u};
            std::memcpy(HalfTexels.data() + offset, channels.data(), sizeof(channels));
        }
    }
}

void FillCubeTexels() {
    Texels.fill(0);
    const auto descriptor = TextureDescriptor(Format1616UNorm, true);
    const auto surface = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(descriptor));
    Require(surface.layers == 6u && surface.guestBytes <= Texels.size(), "cube descriptor did not expose six faces within texture storage");
    const auto& mip = surface.mips.at(0);
    for (std::uint32_t face = 0; face < 6u; ++face) {
        for (std::uint32_t y = 0; y < Side; ++y) {
            for (std::uint32_t x = 0; x < Side; ++x) {
                auto* texel = Texels.data() + surface.GuestLayerOffset(face) + mip.tiledOffset + y * mip.pitchBytes + x * 4u;
                texel[0] = static_cast<std::uint8_t>(face == 0u ? 255u : 0u);
                texel[1] = texel[0];
                texel[3] = 0xffu;
            }
        }
    }
}

void FillInput() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Input[tid * 3u + 0u] = -0.6f + 0.07f * static_cast<float>(tid);
        Input[tid * 3u + 1u] = 1.55f - 0.065f * static_cast<float>(tid);
        Input[tid * 3u + 2u] = -0.25f + 0.05f * static_cast<float>((tid * 7u) % 32u);
    }
    Input[0] = 0.0f; Input[1] = 0.0f; Input[2] = 0.0f;
    Input[3] = 0.999f; Input[4] = 0.999f; Input[5] = 1.2f;
    Input[6] = 0.6f; Input[7] = -0.1f; Input[8] = 0.5f;
    Input[9] = 0.3f; Input[10] = 0.5f; Input[11] = 1.1f;
    Input[12] = 0.9f; Input[13] = 1.05f; Input[14] = -0.05f;
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto x = static_cast<std::int32_t>(tid % 7u) - 3;
        const auto y = static_cast<std::int32_t>((tid / 7u) % 5u) - 2;
        Extra[tid * 3u + 0u] = (static_cast<std::uint32_t>(x) & 0x3fu) | ((static_cast<std::uint32_t>(y) & 0x3fu) << 8u) | (tid % 3u == 0u ? 0x3f0000u : 0u);
        Extra[tid * 3u + 1u] = std::bit_cast<std::uint32_t>(-2.5f + 0.25f * static_cast<float>(tid % 21u));
        Extra[tid * 3u + 2u] = std::bit_cast<std::uint32_t>(0.5f * static_cast<float>(tid % 5u));
    }
    Extra[0] = 0x00001f20u;
}

float LayerCoordinate(std::uint32_t tid) {
    constexpr std::array<float, 12> values{-0.4f, 0.0f, 0.49999997f, 0.5f, 0.99f, 1.0f, 1.5f, 2.5f, 2.49f, 3.5f, 3.7f, 7.0f};
    return values[tid % values.size()];
}

void FillInputArray() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        for (std::uint32_t i = 0; i < 3u; ++i) InputArray[tid * 4u + i] = Input[tid * 3u + i];
        InputArray[tid * 4u + 3u] = LayerCoordinate(tid);
    }
}

float NaNReference(std::uint32_t tid) {
    constexpr std::array<float, 4> references{0.25f, std::numeric_limits<float>::quiet_NaN(), 0.5f, -0.0f};
    return references[(tid / 4u) % references.size()];
}

void FillInputNaN() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto texel = tid % 4u;
        InputNaN[tid * 3u + 0u] = NaNReference(tid);
        InputNaN[tid * 3u + 1u] = (static_cast<float>(texel % Side) + 0.5f) / static_cast<float>(Side);
        InputNaN[tid * 3u + 2u] = (static_cast<float>(texel / Side) + 0.5f) / static_cast<float>(Side);
    }
}

float Passes(float reference, float red, std::uint32_t compare) {
    switch (compare) {
        case LessEqual: return reference <= red ? 1.0f : 0.0f;
        case Greater: return reference > red ? 1.0f : 0.0f;
        case NotEqual: return reference != red ? 1.0f : 0.0f;
        default: throw std::runtime_error("the test reference does not model this compare function");
    }
}

std::uint32_t ExpectedLayer(std::uint32_t tid) {
    const auto rounded = static_cast<int>(std::nearbyint(LayerCoordinate(tid)));
    return static_cast<std::uint32_t>(std::clamp(rounded, 0, static_cast<int>(Layers) - 1));
}

std::int32_t OffsetComponent(std::uint32_t tid, std::uint32_t component) {
    const auto field = (Extra[tid * 3u] >> (component * 8u)) & 0x3fu;
    return static_cast<std::int32_t>(field ^ 0x20u) - 0x20;
}

float ReferenceTexel(int x, int y, float reference, const Sampler& sampler, std::uint32_t layer = 0) {
    const std::uint32_t clamp = sampler.clamp;
    const int size = static_cast<int>(Side);
    if (clamp == ClampBorder && (x < 0 || x >= size || y < 0 || y >= size)) {
        const float border = sampler.border == BorderWhite ? 1.0f : 0.0f;
        return Passes(reference, border, sampler.function);
    }
    const auto address = [&](int value) {
        if (clamp == ClampEdge) return std::clamp(value, 0, static_cast<int>(Side) - 1);
        const int size = static_cast<int>(Side);
        return ((value % size) + size) % size;
    };
    const float red = static_cast<float>(Red[layer][address(y) * Side + address(x)]) / 255.0f;
    return Passes(reference, red, sampler.function);
}

float Expected(std::uint32_t tid, const Sampler& sampler, bool offsets, std::uint32_t layer = 0) {
    const float u = Input[tid * 3u + 1u] * static_cast<float>(Side);
    const float v = Input[tid * 3u + 2u] * static_cast<float>(Side);
    const float reference = std::clamp(Input[tid * 3u + 0u], 0.0f, 1.0f);
    const int offsetX = offsets ? OffsetComponent(tid, 0u) : 0;
    const int offsetY = offsets ? OffsetComponent(tid, 1u) : 0;
    if (sampler.filter == FilterPoint) return ReferenceTexel(static_cast<int>(std::floor(u)) + offsetX, static_cast<int>(std::floor(v)) + offsetY, reference, sampler, layer);
    const float cu = u - 0.5f;
    const float cv = v - 0.5f;
    const int x = static_cast<int>(std::floor(cu)) + offsetX;
    const int y = static_cast<int>(std::floor(cv)) + offsetY;
    const float a = cu - std::floor(cu);
    const float b = cv - std::floor(cv);
    const float top = ReferenceTexel(x, y, reference, sampler, layer) * (1.0f - a) + ReferenceTexel(x + 1, y, reference, sampler, layer) * a;
    const float bottom = ReferenceTexel(x, y + 1, reference, sampler, layer) * (1.0f - a) + ReferenceTexel(x + 1, y + 1, reference, sampler, layer) * a;
    return top * (1.0f - b) + bottom * b;
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device, std::uint32_t format, const Sampler& sampler, std::span<const std::uint32_t> code = Code, bool useCache = false, bool nativeSampleOffsets = true, bool cube = false, bool layered = false, std::span<const float> inputs = {}) {
    std::vector<std::uint32_t> userData(24, 0u);
    const auto input = !inputs.empty() ? BufferDescriptor(inputs.data(), static_cast<std::uint32_t>(inputs.size_bytes())) : cube ? BufferDescriptor(CubeInput.data(), static_cast<std::uint32_t>(sizeof(CubeInput))) : BufferDescriptor(Input.data(), static_cast<std::uint32_t>(sizeof(Input)));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(sizeof(Output)));
    const auto texture = TextureDescriptor(format, cube, layered);
    const auto samplerWords = SamplerDescriptor(sampler);
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    std::copy(texture.begin(), texture.end(), userData.begin() + 8);
    std::copy(samplerWords.begin(), samplerWords.end(), userData.begin() + 16);
    const auto extra = BufferDescriptor(Extra.data(), static_cast<std::uint32_t>(sizeof(Extra)));
    std::copy(extra.begin(), extra.end(), userData.begin() + 20);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = useCache;
    request.target.nonConstantImageOffsets = request.target.nonConstantImageOffsets && nativeSampleOffsets;
    return ShaderRecompiler::Recompile(request);
}

std::uint64_t Run(AgcDriver::VulkanDevice& device, const Sampler& sampler, const char* name, bool offsets = false, bool useCache = false, bool array = false, std::uint64_t expectedVariant = 0) {
    Output.fill(-1.0f);
    const std::span<const std::uint32_t> code = offsets ? std::span<const std::uint32_t>(OffsetCode) : array ? std::span<const std::uint32_t>(ArrayCode) : std::span<const std::uint32_t>(Code);
    const auto result = Compile(device, Format1616UNorm, sampler, code, useCache);
    if (expectedVariant != 0) Require(result.cacheHit && result.variantId == expectedVariant, std::string(name) + " did not reuse the compiled comparison artifact");
    for (const auto& binding : result.bindings) {
        Require(binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers, "emulated comparison retained a sampler binding");
        Require(std::all_of(binding.imageSamplers.begin(), binding.imageSamplers.end(), [](auto mask) { return mask == 0u; }), "emulated comparison retained an unused sampler association");
    }
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const float expected = Expected(tid, sampler, offsets);
        const float tolerance = sampler.filter == FilterPoint ? 0.0f : 1e-4f;
        Require(std::fabs(Output[tid] - expected) <= tolerance, std::string(name) + ": thread " + std::to_string(tid) + " compared to " + std::to_string(Output[tid]) + ", expected " + std::to_string(expected));
    }
    return result.variantId;
}

void RunLayers(AgcDriver::VulkanDevice& device, const Sampler& sampler, const char* name) {
    Output.fill(-1.0f);
    const auto result = Compile(device, Format1616UNorm, sampler, LayerCode, false, true, false, true, InputArray);
    for (const auto& binding : result.bindings) {
        Require(binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers, "array emulated comparison retained a sampler binding");
        Require(std::all_of(binding.imageSamplers.begin(), binding.imageSamplers.end(), [](auto mask) { return mask == 0u; }), "array emulated comparison retained an unused sampler association");
    }
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(LayerCode.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const float expected = Expected(tid, sampler, false, ExpectedLayer(tid));
        const float tolerance = sampler.filter == FilterPoint ? 0.0f : 1e-4f;
        Require(std::fabs(Output[tid] - expected) <= tolerance, std::string(name) + ": thread " + std::to_string(tid) + " compared to " + std::to_string(Output[tid]) + ", expected " + std::to_string(expected));
    }
}

void RunNaN(AgcDriver::VulkanDevice& device, std::uint32_t compare, const char* name) {
    Output.fill(-1.0f);
    const Sampler sampler{ClampEdge, FilterPoint, BorderBlack, 0u, 0u, false, compare};
    const auto result = Compile(device, Format16161616Float, sampler, Code, false, true, false, false, InputNaN);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(Code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const float expected = Passes(NaNReference(tid), HalfRed[tid % 4u], compare);
        Require(Output[tid] == expected, std::string(name) + ": thread " + std::to_string(tid) + " compared to " + std::to_string(Output[tid]) + ", expected " + std::to_string(expected));
    }
}

constexpr std::uint32_t Select(std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t w) {
    return x | (y << 3u) | (z << 6u) | (w << 9u);
}

void RunChannels(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::array<std::uint32_t, 4> channels, std::uint32_t registers, std::uint32_t firstChannel, const char* name) {
    const Sampler sampler{ClampEdge, FilterBilinear};
    Swizzle = Select(channels[0], channels[1], channels[2], channels[3]);
    Output.fill(-2.0f);
    const auto result = Compile(device, Format1616UNorm, sampler, code);
    Swizzle = 0xfacu;
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const float compared = Expected(tid, sampler, false);
        for (std::uint32_t reg = 0; reg < 4u; ++reg) {
            float expected = -1.0f;
            if (reg < registers) {
                const auto select = channels[firstChannel + reg];
                expected = select == 0u ? 0.0f : select == 1u ? 1.0f : compared;
            }
            const float actual = Output[tid * 4u + reg];
            Require(std::fabs(actual - expected) <= 1e-4f, std::string(name) + ": thread " + std::to_string(tid) + " register " + std::to_string(reg) + " returned " + std::to_string(actual) + ", expected " + std::to_string(expected));
        }
    }
}

void RunCube(AgcDriver::VulkanDevice& device, const Sampler& sampler, const char* name) {
    FillCubeTexels();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        CubeInput[tid * 4u + 0u] = 0.5f;
        CubeInput[tid * 4u + 1u] = 1.0f;
        CubeInput[tid * 4u + 2u] = 0.0f;
        CubeInput[tid * 4u + 3u] = 0.0f;
    }
    Output.fill(-1.0f);
    const auto result = Compile(device, Format1616UNorm, sampler, CubeCode, false, true, true);
    for (const auto& binding : result.bindings) {
        Require(binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers, "cube emulated comparison retained a sampler binding");
        Require(std::all_of(binding.imageSamplers.begin(), binding.imageSamplers.end(), [](auto mask) { return mask == 0u; }), "cube emulated comparison retained an unused sampler association");
    }
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(CubeCode.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(Output[tid] == 1.0f, std::string(name) + ": +X cube face comparison returned " + std::to_string(Output[tid]) + ", expected 1");
    }
}

bool BindsDepthCompare(const ShaderRecompiler::RecompileResult& result) {
    return std::any_of(result.bindings.begin(), result.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) {
        return std::any_of(binding.imageDepthCompare.begin(), binding.imageDepthCompare.end(), [](bool compare) { return compare; });
    });
}

void RunSrgbCubeZero(AgcDriver::VulkanDevice& device, std::uint32_t format, const Sampler& sampler, const char* name) {
    FillCubeTexels();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        CubeInput[tid * 4u + 0u] = 0.5f;
        CubeInput[tid * 4u + 1u] = 1.0f;
        CubeInput[tid * 4u + 2u] = 0.0f;
        CubeInput[tid * 4u + 3u] = 0.0f;
    }
    Output.fill(-1.0f);
    const auto result = Compile(device, format, sampler, CubeCode, false, true, true);
    Require(!BindsDepthCompare(result), std::string(name) + " kept a depth-compare binding");
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(CubeCode.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(Output[tid] == 0.0f, std::string(name) + ": thread " + std::to_string(tid) + " compared to " + std::to_string(Output[tid]) + ", expected 0");
    }
}

void RunSrgbZero(AgcDriver::VulkanDevice& device, std::uint32_t format, const Sampler& sampler, const char* name, bool useCache, std::uint64_t expectedVariant) {
    Output.fill(-1.0f);
    const auto result = Compile(device, format, sampler, Code, useCache);
    if (useCache) Require(result.cacheHit && result.variantId == expectedVariant, std::string(name) + " did not reuse the compiled comparison artifact");
    Require(!BindsDepthCompare(result), std::string(name) + " kept a depth-compare binding");
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(Code.data()));
    device.WaitIdle();
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(Output[tid] == 0.0f, std::string(name) + ": thread " + std::to_string(tid) + " compared to " + std::to_string(Output[tid]) + ", expected 0");
    }
}

void Reject(AgcDriver::VulkanDevice& device, std::uint32_t format, const Sampler& sampler, std::string_view reason, std::span<const std::uint32_t> code = Code, bool nativeSampleOffsets = true, bool cube = false) {
    try {
        static_cast<void>(Compile(device, format, sampler, code, false, nativeSampleOffsets, cube));
    } catch (const std::exception& error) {
        Require(std::string_view(error.what()).find(reason) != std::string_view::npos, std::string("unexpected rejection: ") + error.what());
        return;
    }
    Require(false, std::string("expected rejection: ") + std::string(reason));
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto* forced = std::getenv("APS5_SRGB_SHADER_DECODE");
        if (forced != nullptr && std::string_view(forced) == "1") Require((device->Target().srgbDecodeFormats & 3u) == 3u, "forced R8 and RG8 sRGB shader decoding is unavailable");
        FillTexels();
        FillInput();
        FillInputArray();
        FillInputNaN();
        RunCube(*device, {ClampEdge, FilterPoint}, "point cube, clamp to edge");
        Reject(*device, Format1616UNorm, {ClampEdge, FilterBilinear}, "bilinear cube comparison requires DISABLE_CUBE_WRAP", CubeCode, true, true);
        for (std::uint32_t function = 0; function < 8u; ++function) {
            const auto name = "bilinear 8_8_8_8 srgb cube function " + std::to_string(function);
            RunSrgbCubeZero(*device, Format8888Srgb, {ClampEdge, FilterBilinear, BorderBlack, 0u, 0u, false, function}, name.c_str());
        }
        RunSrgbCubeZero(*device, Format8Srgb, {ClampEdge, FilterBilinear}, "bilinear 8 srgb cube without DISABLE_CUBE_WRAP");
        RunSrgbCubeZero(*device, Format8_8Srgb, {ClampEdge, FilterBilinear}, "bilinear 8_8 srgb cube without DISABLE_CUBE_WRAP");
        RunCube(*device, {ClampEdge, FilterBilinear, BorderBlack, 0u, 0u, true}, "bilinear cube with DISABLE_CUBE_WRAP");
        Reject(*device, Format9995Float, {ClampEdge, FilterPoint}, "implemented only for float, unorm and snorm formats");
        for (std::uint32_t function = 0; function < 8u; ++function) {
            const auto name = "bilinear 8_8_8_8 unorm cube function " + std::to_string(function);
            RunSrgbCubeZero(*device, Format8888UNorm, {ClampEdge, FilterBilinear, BorderBlack, 0u, 0u, false, function}, name.c_str());
        }
        RunSrgbCubeZero(*device, Format8888UNorm, {ClampEdge, FilterAnisoBilinear, BorderBlack, 0u, 0u, false, GreaterEqual}, "anisotropic bilinear 8_8_8_8 unorm cube without DISABLE_CUBE_WRAP");
        Reject(*device, Format1616UNorm, {ClampEdge, FilterAnisoBilinear}, "point or bilinear");
        Reject(*device, Format1616UNorm, {ClampEdge, FilterAnisoPoint}, "point or bilinear");
        FillTexels();
        Require(BindsDepthCompare(Compile(*device, Format32Float, {ClampEdge, FilterBilinear})), "an R32 float texture left the native comparison path");
        Require(!BindsDepthCompare(Compile(*device, Format8888UNorm, {ClampEdge, FilterBilinear})), "a color texture kept a depth-compare binding");
        for (const bool useCache : {false, true}) {
            const auto variant = useCache ? Run(*device, {ClampEdge, FilterPoint}, "cached UNORM before sRGB", false, true) : 0u;
            if (useCache) Require(variant != 0, "the cached comparison has no compiled artifact identity");
            for (const auto format : {Format8Srgb, Format8_8Srgb, Format8888Srgb}) {
                for (std::uint32_t function = 0; function < 8u; ++function) {
                    for (const auto filter : {FilterPoint, FilterBilinear}) {
                        const auto name = std::string(useCache ? "cached" : "uncached") + " sRGB format " + std::to_string(format) + " function " + std::to_string(function) + (filter == FilterPoint ? " point" : " bilinear");
                        RunSrgbZero(*device, format, {ClampEdge, filter, BorderBlack, 0u, 0u, false, function}, name.c_str(), useCache, variant);
                    }
                }
            }
            if (useCache) Run(*device, {ClampEdge, FilterPoint}, "cached UNORM after sRGB", false, true, false, variant);
        }
        for (const auto format : {Format8UNorm, Format8_8UNorm, Format8888UNorm}) {
            for (std::uint32_t function = 0; function < 8u; ++function) {
                for (const auto filter : {FilterPoint, FilterBilinear, FilterAnisoBilinear}) {
                    if (format != Format8888UNorm && function != LessEqual) continue;
                    const auto name = "8-bit unorm format " + std::to_string(format) + " function " + std::to_string(function) + " filter " + std::to_string(filter);
                    RunSrgbZero(*device, format, {ClampEdge, filter, BorderBlack, 0u, 0u, false, function}, name.c_str(), false, 0u);
                }
            }
        }
        Run(*device, {ClampEdge, FilterPoint}, "point, clamp to edge");
        Run(*device, {ClampWrap, FilterPoint}, "point, wrap");
        Run(*device, {ClampWrap, FilterPoint}, "point, wrap with negative offsets", true);
        Run(*device, {ClampEdge, FilterBilinear}, "bilinear, 2D array instruction on a 2D texture", false, false, true);
        Run(*device, {ClampEdge, FilterBilinear}, "bilinear, clamp to edge");
        Run(*device, {ClampWrap, FilterBilinear}, "bilinear, wrap");
        Run(*device, {ClampBorder, FilterPoint, BorderWhite}, "point, white border");
        Run(*device, {ClampBorder, FilterPoint, BorderBlack}, "point, black border");
        Run(*device, {ClampBorder, FilterBilinear, BorderWhite}, "bilinear, white border");
        Run(*device, {ClampBorder, FilterBilinear, BorderBlack}, "bilinear, black border");
        Run(*device, {ClampEdge, FilterPoint, BorderBlack, 0u, 0x0140u}, "point, clamp to edge, sampler LOD bias");
        Run(*device, {ClampEdge, FilterPoint}, "point, clamp to edge, offsets, bias and LOD clamp", true);
        Run(*device, {ClampWrap, FilterBilinear}, "bilinear, wrap, offsets, bias and LOD clamp", true);
        Run(*device, {ClampBorder, FilterBilinear, BorderWhite, 0u, 0x3f00u}, "bilinear, white border, offsets, bias and LOD clamp", true);
        Run(*device, {ClampEdge, FilterPoint, BorderBlack, 0u, 0u, false, Greater}, "point, GREATER");
        Run(*device, {ClampBorder, FilterBilinear, BorderWhite, 0u, 0u, false, Greater}, "bilinear, white border, GREATER");
        RunLayers(*device, {ClampEdge, FilterPoint}, "2D array, point, clamp to edge");
        RunLayers(*device, {ClampEdge, FilterBilinear}, "2D array, bilinear, clamp to edge");
        RunLayers(*device, {ClampWrap, FilterBilinear}, "2D array, bilinear, wrap");
        RunLayers(*device, {ClampBorder, FilterBilinear, BorderWhite}, "2D array, bilinear, white border");
        RunLayers(*device, {ClampEdge, FilterPoint, BorderBlack, 0u, 0u, false, Greater}, "2D array, point, GREATER");
        RunNaN(*device, NotEqual, "half float, NOT_EQUAL with NaN and signed zero");
        RunNaN(*device, LessEqual, "half float, LESSEQUAL with NaN and signed zero");
        for (unsigned iteration = 0; iteration < 2; ++iteration) {
            Run(*device, {ClampEdge, FilterPoint}, "cached point comparison", false, true);
            Run(*device, {ClampWrap, FilterBilinear}, "cached bilinear comparison with offsets", true, true);
        }
        Reject(*device, Format32Float, {ClampEdge, FilterPoint}, "native comparison with a nonconstant texel offset requires", OffsetCode, false);
        RunChannels(*device, QuadCode, {4, 5, 6, 7}, 4, 0, "view R,G,B,A");
        RunChannels(*device, QuadCode, {0, 0, 0, 0}, 4, 0, "view 0,0,0,0");
        RunChannels(*device, QuadCode, {1, 1, 1, 1}, 4, 0, "view 1,1,1,1");
        RunChannels(*device, QuadCode, {0, 4, 1, 5}, 4, 0, "view 0,R,1,G");
        RunChannels(*device, QuadCode, {4, 0, 0, 1}, 4, 0, "view R,0,0,1");
        RunChannels(*device, QuadCode, {5, 4, 5, 6}, 4, 0, "view G,R,B,A");
        RunChannels(*device, QuadCode, {7, 4, 4, 4}, 4, 0, "view A,R,R,R");
        RunChannels(*device, PairCode, {0, 4, 4, 4}, 2, 1, "view 0,R,R,R, dmask 6");
        Swizzle = Select(4, 2, 4, 4);
        Reject(*device, Format8888UNorm, {ClampEdge, FilterPoint}, "every view channel is a constant or a color channel");
        Swizzle = Select(5, 5, 5, 5);
        Reject(*device, Format8888UNorm, {ClampEdge, FilterPoint}, "a view channel selects red");
        Swizzle = 0xfacu;
        Reject(*device, Format8888UInt, {ClampEdge, FilterPoint}, "unsupported format");
        Reject(*device, Format8888UNorm, {ClampMirror, FilterPoint}, "wrap, clamp-to-edge or clamp-to-border");
        Reject(*device, Format8888UNorm, {ClampHalfBorder, FilterPoint}, "wrap, clamp-to-edge or clamp-to-border");
        Reject(*device, Format8888UNorm, {ClampBorder, FilterPoint, BorderTable}, "border color table");
        Reject(*device, Format8888UNorm, {ClampEdge, FilterBilinear, BorderBlack, ReductionMin}, "min or max reduction");
        Reject(*device, Format8888UNorm, {ClampEdge, FilterPoint, BorderBlack, 0u, 0u, false, LessEqual, true}, "TRUNC_COORD");
        std::puts("emulated color compare tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
