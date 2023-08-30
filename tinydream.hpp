#pragma once
/*
* Tiny Dream - Header-Only, Embedded Stable Diffusion Inference Library.
* 
* Copyright (C) 2023 PixLab| Symisc Systems. https://pixlab.io/tiny-dream
* Version 1.7.5
*
* Symisc Systems employs a dual licensing model that offers customers
* a choice of either our open source license (GNU Affero AGPLv3) 
* or a commercial license.
*
* For information on licensing, redistribution of the Tiny Dream, 
* and for a DISCLAIMER OF ALL WARRANTIES please visit:
*     https://pixlab.io/tiny-dream#license
* or contact:
*     licensing@symisc.net
*     support@pixlab.io
*/
/*
 * This file is part of Tiny Dream - Open Source Release (GNU Affero AGPLv3)
 *
 * Tiny Dream is free software : you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Tiny Dream is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Tiny Dream. If not, see <http://www.gnu.org/licenses/>.
 */
 /*
  * The TINY_DREAM_VERSION_STR C preprocessor macro evaluates to a string literal
  * that is the Tiny Dream version in the format "X.Y.Z" where X is the major
  * version number and Y is the minor version number and Z is the release
  * number.
  */
#define TINY_DREAM_VERSION_STR "1.7.5"
  /*
  * The TINY_DREAM_VERSION_NUMBER C preprocessor macro resolves to an integer
  * with the value (X*1000000 + Y*1000 + Z) where X, Y, and Z are the same
  * numbers used in [TINY_DREAM_VERSION_STR].
  */
#define TINY_DREAM_VERSION_NUMBER 1007005
/* $SymiscID: tinydream.hpp v1.7.5 WIN10/VS2019 2023-08-07 05:34 stable <devel@symisc.net> $ */
#include<algorithm>
#include<cmath>
#include<fstream>
#include<iostream>
#include<map>
#include<numeric>
#include<random>
#include<regex>
#include<stack>
#include<string>
#include<unordered_map>
#include<vector>
#include<functional>
#include<memory>
/*
* As of this release, the current backend tensor library is Tencent NCNN
* with an urgent, on-going transition to a less bloated tensor library
* such as SOD (https://sod.pixlab.io) or GGML (https://github.com/ggerganov/ggml).
* 
* Refer to the Roadmap page at: https://pixlab.io/tiny-dream#roadmap for the ongoing
* progress.
*/
#define TINY_DREAM_INFERENCE_ENGINE "ncnn" //sod or ggml in the planned transition
#include<ncnn/benchmark.h>
#include<ncnn/net.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
class tinyDream {
private:
    std::unordered_map<std::string, int> idxTokens;
    std::string assetsPath{ "./assets/" };
    std::string outputPath{ "./" };
    std::string imgPrefix{ "tinydream_" };
    std::function<void(const char* /* zLogMsg */, int /* msg length */ , void* /* pCookie */)> xLog{nullptr};
    void* xLogUserData{ nullptr };
    inline constexpr void _log(const std::string& msg) const {
        if (xLog) {
            xLog(msg.c_str(), static_cast<int>(msg.size()), xLogUserData);
        }
    }
    inline std::string joinPaths(const char* zModel) {
        // Windows handle the '/' directory separator without issue.
        return assetsPath + zModel;
    }
    std::string randomString(int length = 12)
    {
        static const char chars[] = "_0123456789abcdefghijklmnopqrstuvwxyz";
        thread_local static std::mt19937 rg{ std::random_device{}() };
        thread_local static std::uniform_int_distribution<std::string::size_type> pick(0, sizeof(chars) - 2);
        std::string out;
        out.reserve(length);
        while (length--)
            out += chars[pick(rg)];
        return std::move(out);
    }
    inline ncnn::Mat rand64(int seed)
    {
        std::vector<float> arr;
        {
            thread_local std::mt19937 gen{ static_cast<unsigned long>(seed) };
            thread_local std::normal_distribution<float> d{ 0.0f, 1.0f };
            arr.resize(64 * 64 * 4);
            std::for_each(arr.begin(), arr.end(), [&](float& x)
            {
                x = d(gen);
            });
        }
        ncnn::Mat out(64, 64, 4, reinterpret_cast<void*>(arr.data()));
        return out.clone();
    }
    inline std::vector<std::string> split(std::string str)
    {
        std::vector<std::string> out;
        std::string::size_type pos;
        str += " ";
        int size = str.size();
        for (int i = 0; i < size; i++)
        {
            pos = std::min(str.find(" ", i), str.find(",", i));
            if (pos < str.size())
            {
                std::string s = str.substr(i, pos - i);
                std::string pat = std::string(1, str[pos]);
                if (s.length() > 0)
                    out.push_back(s + "</w>");
                if (pat != " ")
                    out.push_back(pat + "</w>");
                i = pos;
            }
        }
        return out;
    }
    inline ncnn::Mat CFGDenoiserCompVisDenoiser(ncnn::Net& net, float const* log_sigmas, ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond);
    ncnn::Mat getTextEmbedding(ncnn::Net& net, std::string prompt);
    bool loadTokens();
    std::pair<ncnn::Mat, ncnn::Mat> promptEncoder(const std::string& positivePrompt, const std::string& negativePrompt);
    ncnn::Mat diffusionSolver(ncnn::Mat& c, ncnn::Mat& uc, int seed, int step);
    ncnn::Mat decoderSolver(ncnn::Mat& sample);
    inline ncnn::Mat upScale4xEsprGan(ncnn::Mat& input);
    std::vector<std::pair<std::string, float>> parsePromptAttention(std::string& texts);
public:
    /*
    * Public Exposed Methods.
    *
    * Refer to the Tiny Dream Documentation at: https://pixlab.io/tiny-dream
    * for the official C++ API Reference Guide.
    */
    tinyDream(const std::string& assetsPath = "./assets/") {
        if (this->assetsPath != assetsPath) {
            this->assetsPath = assetsPath;
            // Windows handle the '/' directory separator without issue.
            if (this->assetsPath.back() != '/') {
                this->assetsPath.push_back('/');
            }
        }
    }
    ~tinyDream() = default;
    static const char * about() {
        return "Tiny Dream " TINY_DREAM_VERSION_STR ": Stable Diffusion Inference (" TINY_DREAM_INFERENCE_ENGINE ") in C++ - Copyright (C) PixLab | Symisc Systems(https://pixlab.io/ - https://symisc.net/). All rights reserved.";
    }
    static std::pair<std::string/*Positive Prompt */, std::string /* Negative Prompt*/> promptExample() {
        return std::make_pair("pyramid, desert, palm trees, river, sun, (landscape), (high quality)", "nsfw, nudity, gore, blood, genitals, mutilation, mutation, disfigured, deformed, mutated, war, destruction, hell, torture, apocalypse, sex, chocking, blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, extra legs, extra arms, cross-eye, body out of frame, bad art, bad anatomy, blurred, text, watermark, grainy");
    }
    void setAssetsPath(const std::string& assetsPath) {
        this->assetsPath = assetsPath;
        // Windows handle the '/' directory separator without issue.
        if (this->assetsPath.back() != '/') {
            this->assetsPath.push_back('/');
        }
    }
    void setImageOutputPath(const std::string& outputPath) {
        this->outputPath = outputPath;
        // Windows handle the '/' directory separator without issue.
        if (this->outputPath.back() != '/') {
            this->outputPath.push_back('/');
        }
    }
    void setImageOutputPrefix(const std::string& outputImgPrefix = "tinydream_") {
        this->imgPrefix = outputImgPrefix;
    }
    void setLogCallback(std::function<void(const char*/* zLogMsg */, int /*msg length*/, void*/* pCookie */)> xLogFunc, void *pUserData) {
        xLog = xLogFunc;
        xLogUserData = pUserData;
    }
    bool dream(const std::string& positivePrompt, const std::string& negativePrompt, std::string& outputImgPath, bool upScale = true, int seed = 42, int step = 30);
};
inline ncnn::Mat tinyDream::CFGDenoiserCompVisDenoiser(ncnn::Net& net, float const* log_sigmas, ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond)
{
    // Based on the excellent fork by fengwang: https://github.com/fengwang
    float c_out = -1.0 * sigma;
    float c_in = 1.0 / std::sqrt(sigma * sigma + 1);
    float log_sigma = std::log(sigma);
    std::vector<float> dists(1000);
    for (int i = 0; i < 1000; i++){
        if (log_sigma - log_sigmas[i] >= 0)
            dists[i] = 1;
        else
            dists[i] = 0;
        if (i == 0) continue;
        dists[i] += dists[i - 1];
    }
    int low_idx = std::min(int(std::max_element(dists.begin(), dists.end()) - dists.begin()), 1000 - 2);
    int high_idx = low_idx + 1;
    float low = log_sigmas[low_idx];
    float high = log_sigmas[high_idx];
    float w = (low - log_sigma) / (low - high);
    w = std::max(0.f, std::min(1.f, w));
    float t = (1 - w) * low_idx + w * high_idx;
    ncnn::Mat t_mat(1);
    t_mat[0] = t;
    ncnn::Mat c_in_mat(1);
    c_in_mat[0] = c_in;
    ncnn::Mat c_out_mat(1);
    c_out_mat[0] = c_out;
    ncnn::Mat denoised_cond;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.input("in0", input);
        ex.input("in1", t_mat);
        ex.input("in2", cond);
        ex.input("c_in", c_in_mat);
        ex.input("c_out", c_out_mat);
        ex.extract("outout", denoised_cond);
    }
    ncnn::Mat denoised_uncond;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.input("in0", input);
        ex.input("in1", t_mat);
        ex.input("in2", uncond);
        ex.input("c_in", c_in_mat);
        ex.input("c_out", c_out_mat);
        ex.extract("outout", denoised_uncond);
    }
    for (int c = 0; c < 4; c++)
    {
        float* u_ptr = denoised_uncond.channel(c);
        float* c_ptr = denoised_cond.channel(c);

        for (int hw = 0; hw < 64 * 64; hw++)
        {
            (*u_ptr) = (*u_ptr) + 7 * ((*c_ptr) - (*u_ptr));
            u_ptr++;
            c_ptr++;
        }
    }
    return denoised_uncond;
}
inline ncnn::Mat tinyDream::upScale4xEsprGan(ncnn::Mat& input)
{
    _log("[-Info-]: Starting Real-ESRGAN Image Resolution Upscaler...\n");
    auto startingTime = ncnn::get_current_time();
    ncnn::Net net;
    {
        net.opt.use_vulkan_compute = false;
        net.opt.use_winograd_convolution = false;
        net.opt.use_sgemm_convolution = false;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_packing_layout = true;
        net.load_param(joinPaths("RealESRGAN_x4plus_anime.param").c_str());
        net.load_model(joinPaths("RealESRGAN_x4plus_anime.bin").c_str());
    }
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    {
        constexpr float mean[] = { 0.0f, 0.0f, 0.0f };
        constexpr float norm[] = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
        input.substract_mean_normalize(mean, norm);
    }
    ex.input("data", input);
    ncnn::Mat highres;
    ex.extract("output", highres);
    {
        constexpr float mean[] = { 0.0f, 0.0f, 0.0f };
        constexpr float norm[] = { 255.0f, 255.0f, 255.0f };
        highres.substract_mean_normalize(mean, norm);
    }
    auto elapsed = ncnn::get_current_time() - startingTime;
    _log(std::string{ "Real-ESRGAN inference took " } + std::to_string(elapsed) + std::string{ " milliseconds to complete.\n" });
    return highres;
}
ncnn::Mat tinyDream::decoderSolver(ncnn::Mat& sample)
{
    ncnn::Net net;
    _log("[-Info-]: Starting Variational Auto-Encoder (VAE) model (AutoencoderKL FP16) to decode images from latent representations...\n");
    auto startingTime = ncnn::get_current_time();
    {
        net.opt.use_vulkan_compute = false;
        net.opt.use_winograd_convolution = false;
        net.opt.use_sgemm_convolution = false;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.use_packing_layout = true;
        net.load_param(joinPaths("AutoencoderKL-fp16.param").c_str());
        net.load_model(joinPaths("AutoencoderKL-fp16.bin").c_str());
    }
    ncnn::Mat xDdim;
    {
        constexpr float factor[4] = { 5.48998f, 5.48998f, 5.48998f, 5.48998f };
        sample.substract_mean_normalize(0, factor);
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.input("input.1", sample);
        ex.extract("815", xDdim);
        constexpr float _mean_[3] = { -1.0f, -1.0f, -1.0f };
        constexpr float _norm_[3] = { 127.5f, 127.5f, 127.5f };
        xDdim.substract_mean_normalize(_mean_, _norm_);
    }
    auto elapsed = ncnn::get_current_time() - startingTime;
    _log(std::string{ "Variational Auto-Encoder (VAE) took " } + std::to_string(elapsed) + std::string{ " milliseconds to complete.\n" });
    return xDdim;
}
ncnn::Mat tinyDream::diffusionSolver(ncnn::Mat& c, ncnn::Mat& uc, int seed, int step)
{
    ncnn::Net net;
    _log("[-Info-]: Starting UNet Model (UNetModel FP16) to denoise the encoded image latent...\n");
    auto startingTime = ncnn::get_current_time();
    {
        net.opt.use_vulkan_compute = false;
        net.opt.use_winograd_convolution = false;
        net.opt.use_sgemm_convolution = false;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        net.opt.use_packing_layout = true;
        net.load_param(joinPaths("UNetModel-fp16.param").c_str());
        net.load_model(joinPaths("UNetModel-fp16.bin").c_str());
    }
    float const log_sigmas[1000] = { -3.534698963f, -3.186542273f, -2.982215166f, -2.836785793f, -2.723614454f, -2.63086009f, -2.552189827f, -2.483832836f, -2.423344612f, -2.369071007f, -2.319822073f, -2.274721861f, -2.233105659f, -2.1944592f, -2.15836978f, -2.124504805f, -2.092598915f, -2.062425613f, -2.033797979f, -2.006558657f, -1.980568767f, -1.955715537f, -1.931894541f, -1.90902102f, -1.887015939f, -1.865811229f, -1.845347762f, -1.825569034f, -1.806429505f, -1.787884474f, -1.769894958f, -1.752426744f, -1.735446692f, -1.718925714f, -1.702836871f, -1.687156916f, -1.671862602f, -1.656933904f, -1.642351151f, -1.628097653f, -1.614156127f, -1.60051167f, -1.587151766f, -1.574060798f, -1.561229229f, -1.548643827f, -1.536295056f, -1.524172544f, -1.512266994f, -1.500569701f, -1.489071608f, -1.477766395f, -1.466645837f, -1.455702543f, -1.444930911f, -1.43432498f, -1.423877597f, -1.413584232f, -1.403439164f, -1.393437862f, -1.383575559f, -1.373847008f, -1.364248514f, -1.35477531f, -1.345424652f, -1.336191654f, -1.327073216f, -1.318066359f, -1.309167266f, -1.300373077f, -1.291680217f, -1.283086777f, -1.2745893f, -1.266185522f, -1.257872462f, -1.249648571f, -1.24151063f, -1.233456612f, -1.225485086f, -1.217592716f, -1.209778547f, -1.202040195f, -1.194375992f, -1.186783791f, -1.179262877f, -1.171809912f, -1.164424658f, -1.157105207f, -1.14985013f, -1.142657518f, -1.135526419f, -1.128455162f, -1.121442795f, -1.114487886f, -1.107589245f, -1.100745678f, -1.093955874f, -1.087218761f, -1.080533504f, -1.073898554f, -1.067313433f, -1.060776949f, -1.05428803f, -1.04784584f, -1.041449785f, -1.035098076f, -1.028790832f, -1.022526741f, -1.01630497f, -1.010124922f, -1.003985763f, -0.9978865385f, -0.9918267727f, -0.9858058691f, -0.9798227549f, -0.9738773108f, -0.9679679871f, -0.9620951414f, -0.9562574625f, -0.9504545927f, -0.9446860552f, -0.9389512539f, -0.9332492948f, -0.9275799394f, -0.9219425917f, -0.9163367748f, -0.910761714f, -0.9052170515f, -0.8997026086f, -0.8942174911f, -0.8887614012f, -0.8833341002f, -0.8779345155f, -0.8725628257f, -0.8672183752f, -0.8619008064f, -0.8566094041f, -0.851344347f, -0.8461046219f, -0.8408905864f, -0.8357009888f, -0.8305362463f, -0.8253954053f, -0.8202784657f, -0.8151849508f, -0.8101147413f, -0.8050670028f, -0.800041914f, -0.7950390577f, -0.7900577784f, -0.7850983739f, -0.7801600099f, -0.7752425075f, -0.7703458071f, -0.7654693723f, -0.7606129646f, -0.7557764053f, -0.7509595752f, -0.7461619377f, -0.7413833141f, -0.7366235256f, -0.7318821549f, -0.7271592021f, -0.7224541306f, -0.7177669406f, -0.7130974531f, -0.7084451318f, -0.7038100958f, -0.6991918087f, -0.6945903301f, -0.6900054812f, -0.6854367256f, -0.6808840632f, -0.6763471961f, -0.6718261838f, -0.6673204899f, -0.6628302932f, -0.658354938f, -0.6538946629f, -0.64944911f, -0.6450180411f, -0.6406013966f, -0.6361990571f, -0.6318103671f, -0.6274358034f, -0.6230751276f, -0.618727684f, -0.6143938303f, -0.61007303f, -0.6057654023f, -0.6014707088f, -0.597188592f, -0.5929191113f, -0.5886622667f, -0.5844176412f, -0.5801851153f, -0.5759648085f, -0.5717563629f, -0.5675594807f, -0.5633742213f, -0.5592005849f, -0.5550382137f, -0.5508873463f, -0.546747148f, -0.5426182747f, -0.5385001898f, -0.5343927145f, -0.5302959085f, -0.5262096524f, -0.5221338272f, -0.5180680752f, -0.5140126348f, -0.5099670291f, -0.5059314966f, -0.5019059181f, -0.4978898764f, -0.4938833714f, -0.4898864031f, -0.4858988523f, -0.4819207191f, -0.477951467f, -0.4739915431f, -0.4700405598f, -0.4660984278f, -0.4621651769f, -0.4582404494f, -0.4543244243f, -0.4504169226f, -0.4465178251f, -0.4426270425f, -0.438744545f, -0.4348701835f, -0.4310038984f, -0.4271455407f, -0.4232949913f, -0.4194523394f, -0.415617466f, -0.4117901921f, -0.4079704583f, -0.4041582644f, -0.4003533721f, -0.3965558708f, -0.3927654326f, -0.3889823556f, -0.3852062523f, -0.3814373016f, -0.3776751757f, -0.3739199638f, -0.3701713085f, -0.3664295971f, -0.3626944721f, -0.3589659631f, -0.3552440405f, -0.3515283465f, -0.3478190601f, -0.3441161811f, -0.3404195011f, -0.3367289305f, -0.3330446184f, -0.3293660879f, -0.3256936371f, -0.3220270574f, -0.3183663487f, -0.3147114515f, -0.3110622168f, -0.3074187338f, -0.3037807941f, -0.3001481593f, -0.296521306f, -0.2928997874f, -0.2892835438f, -0.2856727242f, -0.2820670605f, -0.2784664929f, -0.2748712003f, -0.271281004f, -0.2676956952f, -0.2641154826f, -0.2605400383f, -0.2569694519f, -0.2534037232f, -0.2498426586f, -0.2462864369f, -0.24273476f, -0.2391877174f, -0.2356451303f, -0.2321071476f, -0.2285735607f, -0.2250443399f, -0.2215195149f, -0.2179989815f, -0.214482531f, -0.210970372f, -0.2074623704f, -0.2039585114f, -0.2004585862f, -0.1969628185f, -0.1934709698f, -0.1899829209f, -0.1864988059f, -0.1830185503f, -0.1795420945f, -0.1760693043f, -0.172600165f, -0.1691347659f, -0.1656728834f, -0.1622146815f, -0.1587598771f, -0.1553086042f, -0.1518607438f, -0.1484163553f, -0.1449751854f, -0.1415374279f, -0.138102904f, -0.1346716136f, -0.1312434822f, -0.1278186142f, -0.1243967116f, -0.1209778786f, -0.1175621748f, -0.1141493395f, -0.1107395291f, -0.1073326841f, -0.1039286032f, -0.100527443f, -0.09712906927f, -0.09373350441f, -0.09034062177f, -0.08695036173f, -0.08356288075f, -0.08017785847f, -0.0767955035f, -0.07341576368f, -0.07003845274f, -0.06666365266f, -0.06329131871f, -0.05992120504f, -0.05655376986f, -0.05318845809f, -0.04982547462f, -0.04646483436f, -0.04310630262f, -0.03975001723f, -0.03639599681f, -0.03304407373f, -0.02969419584f, -0.02634644695f, -0.02300059609f, -0.01965690777f, -0.0163150914f, -0.01297534816f, -0.009637393057f, -0.006301366724f, -0.002967105946f, 0.0003651905863f, 0.00369580253f, 0.007024710067f, 0.01035177521f, 0.0136772152f, 0.01700089127f, 0.02032313682f, 0.02364357933f, 0.02696254663f, 0.03028002009f, 0.03359586f, 0.03691027686f, 0.04022336379f, 0.04353487119f, 0.04684500396f, 0.05015373603f, 0.05346116424f, 0.05676726624f, 0.060072124f, 0.06337571889f, 0.06667824835f, 0.06997924298f, 0.073279351f, 0.07657821476f, 0.07987590879f, 0.08317264169f, 0.08646827191f, 0.08976276964f, 0.0930563435f, 0.09634894878f, 0.09964046627f, 0.1029312909f, 0.1062210724f, 0.109510012f, 0.1127980649f, 0.1160853282f, 0.1193717569f, 0.1226574481f, 0.1259424686f, 0.1292266697f, 0.1325102597f, 0.1357929856f, 0.1390753537f, 0.1423569024f, 0.1456380188f, 0.1489184797f, 0.1521983445f, 0.155477792f, 0.1587566882f, 0.1620351225f, 0.1653131396f, 0.1685907096f, 0.1718678325f, 0.1751447469f, 0.1784212291f, 0.1816974431f, 0.1849732697f, 0.1882487684f, 0.1915241033f, 0.1947992444f, 0.198074162f, 0.2013489157f, 0.2046233714f, 0.2078978866f, 0.211172238f, 0.214446485f, 0.2177205831f, 0.2209947109f, 0.2242688239f, 0.2275429815f, 0.230817154f, 0.2340912819f, 0.2373655587f, 0.2406399995f, 0.2439144254f, 0.247189045f, 0.2504638135f, 0.2537388504f, 0.2570140362f, 0.2602894902f, 0.2635650337f, 0.266841054f, 0.2701171935f, 0.27339378f, 0.2766706944f, 0.2799479663f, 0.2832255363f, 0.2865035534f, 0.2897821367f, 0.2930608988f, 0.2963403165f, 0.2996201515f, 0.3029005229f, 0.306181401f, 0.3094629347f, 0.3127449751f, 0.3160274923f, 0.3193107247f, 0.322594583f, 0.3258791566f, 0.329164356f, 0.3324502707f, 0.335736841f, 0.3390242159f, 0.3423123956f, 0.3456012905f, 0.3488909006f, 0.352181375f, 0.3554728627f, 0.3587650359f, 0.3620581031f, 0.365352124f, 0.3686470091f, 0.371942848f, 0.3752396405f, 0.3785375357f, 0.3818363845f, 0.3851362169f, 0.3884370327f, 0.3917389512f, 0.3950420022f, 0.3983460069f, 0.4016513228f, 0.4049576223f, 0.408265233f, 0.4115738571f, 0.4148837626f, 0.4181949198f, 0.4215073586f, 0.4248209298f, 0.4281358421f, 0.4314520359f, 0.434769541f, 0.4380882978f, 0.441408515f, 0.444730103f, 0.448053062f, 0.4513774216f, 0.4547032118f, 0.4580304027f, 0.461359024f, 0.4646892846f, 0.4680209458f, 0.4713541865f, 0.4746888876f, 0.4780252576f, 0.4813631475f, 0.4847026467f, 0.4880437851f, 0.4913864136f, 0.4947308302f, 0.4980769157f, 0.5014246702f, 0.5047741532f, 0.5081253052f, 0.5114781857f, 0.5148329139f, 0.5181894302f, 0.5215476751f, 0.5249077678f, 0.5282697678f, 0.5316335559f, 0.5349991322f, 0.5383667946f, 0.5417361856f, 0.5451076627f, 0.5484809279f, 0.5518562794f, 0.5552335382f, 0.5586128235f, 0.5619941354f, 0.5653774738f, 0.5687628388f, 0.5721503496f, 0.5755399466f, 0.5789316297f, 0.5823253393f, 0.585721314f, 0.5891193748f, 0.5925196409f, 0.5959220529f, 0.5993267298f, 0.6027336717f, 0.6061428785f, 0.6095542312f, 0.6129679084f, 0.6163839698f, 0.6198022962f, 0.6232229471f, 0.6266459823f, 0.6300714016f, 0.6334991455f, 0.6369293332f, 0.6403619647f, 0.6437969804f, 0.6472345591f, 0.6506744623f, 0.6541169882f, 0.6575619578f, 0.6610094905f, 0.6644595861f, 0.6679121852f, 0.6713674068f, 0.6748251915f, 0.6782855988f, 0.6817486286f, 0.6852144003f, 0.688682735f, 0.6921537519f, 0.6956274509f, 0.6991039515f, 0.7025832534f, 0.7060650587f, 0.7095498443f, 0.713037312f, 0.7165275812f, 0.7200207114f, 0.7235167027f, 0.7270154953f, 0.730517149f, 0.7340217829f, 0.7375292182f, 0.7410396338f, 0.7445529699f, 0.7480692267f, 0.7515884042f, 0.7551106215f, 0.7586359382f, 0.7621641755f, 0.7656953931f, 0.7692299485f, 0.7727673054f, 0.7763077617f, 0.779851377f, 0.7833981514f, 0.786947906f, 0.7905010581f, 0.7940571904f, 0.7976165414f, 0.8011791706f, 0.8047449589f, 0.8083140254f, 0.8118864298f, 0.8154619932f, 0.8190407753f, 0.8226229548f, 0.8262084126f, 0.8297972679f, 0.833389461f, 0.836984992f, 0.8405839205f, 0.8441862464f, 0.8477919698f, 0.8514010906f, 0.8550137877f, 0.8586298227f, 0.8622494936f, 0.8658725023f, 0.8694992065f, 0.8731292486f, 0.8767629862f, 0.8804001808f, 0.8840410113f, 0.8876854181f, 0.8913334608f, 0.894985199f, 0.8986404538f, 0.9022994041f, 0.9059621096f, 0.9096283317f, 0.9132984877f, 0.9169722795f, 0.9206498265f, 0.924331069f, 0.9280161858f, 0.9317050576f, 0.9353976846f, 0.9390941858f, 0.9427945018f, 0.9464985728f, 0.9502066374f, 0.9539185762f, 0.957634449f, 0.9613542557f, 0.9650779963f, 0.9688056707f, 0.9725371599f, 0.9762728214f, 0.9800124764f, 0.983756125f, 0.9875037074f, 0.9912554026f, 0.9950110912f, 0.9987710118f, 1.002534866f, 1.006302953f, 1.010075092f, 1.013851404f, 1.017631888f, 1.021416545f, 1.025205374f, 1.028998375f, 1.032795668f, 1.036597133f, 1.040402889f, 1.044212818f, 1.048027158f, 1.051845789f, 1.055668592f, 1.059495926f, 1.063327432f, 1.067163467f, 1.071003675f, 1.074848413f, 1.078697562f, 1.082551122f, 1.086408973f, 1.090271473f, 1.094138384f, 1.098009825f, 1.101885676f, 1.105766058f, 1.109651089f, 1.113540411f, 1.117434502f, 1.121333122f, 1.125236511f, 1.129144192f, 1.13305676f, 1.136973858f, 1.140895605f, 1.144822001f, 1.148753166f, 1.15268898f, 1.156629443f, 1.160574675f, 1.164524794f, 1.168479443f, 1.172439098f, 1.176403403f, 1.180372596f, 1.184346557f, 1.188325405f, 1.192309022f, 1.196297526f, 1.200290918f, 1.204289317f, 1.208292484f, 1.212300658f, 1.216313839f, 1.220331907f, 1.224354982f, 1.228383064f, 1.232415915f, 1.236454129f, 1.240497231f, 1.244545341f, 1.248598576f, 1.252656817f, 1.256720304f, 1.260788798f, 1.264862418f, 1.268941164f, 1.273025036f, 1.277114153f, 1.281208396f, 1.285307884f, 1.289412618f, 1.293522477f, 1.297637701f, 1.301758051f, 1.305883765f, 1.310014725f, 1.314151049f, 1.318292618f, 1.322439551f, 1.326591969f, 1.330749512f, 1.334912539f, 1.33908093f, 1.343254805f, 1.347433925f, 1.351618767f, 1.355808854f, 1.360004425f, 1.36420548f, 1.368412018f, 1.372624159f, 1.376841784f, 1.381064892f, 1.385293603f, 1.389527798f, 1.393767595f, 1.398013115f, 1.402264118f, 1.406520724f, 1.410783052f, 1.415050983f, 1.419324636f, 1.423603892f, 1.427888989f, 1.43217957f, 1.436476111f, 1.440778255f, 1.445086241f, 1.449399829f, 1.453719258f, 1.458044529f, 1.462375641f, 1.466712594f, 1.471055388f, 1.475403905f, 1.479758382f, 1.484118819f, 1.488484859f, 1.492857099f, 1.497235179f, 1.50161922f, 1.506009102f, 1.510405064f, 1.514806986f, 1.519214869f, 1.523628831f, 1.528048754f, 1.532474637f, 1.536906719f, 1.541344643f, 1.545788884f, 1.550239086f, 1.554695368f, 1.559157968f, 1.563626409f, 1.568101287f, 1.572582126f, 1.577069283f, 1.581562519f, 1.586061954f, 1.590567589f, 1.595079541f, 1.599597573f, 1.604121923f, 1.608652592f, 1.613189697f, 1.617732882f, 1.622282386f, 1.626838207f, 1.631400466f, 1.635969043f, 1.640543938f, 1.645125389f, 1.649713039f, 1.654307127f, 1.658907652f, 1.663514495f, 1.668127894f, 1.67274785f, 1.677374125f, 1.682006836f, 1.686646223f, 1.691291928f, 1.695944309f, 1.700603247f, 1.705268621f, 1.709940553f, 1.71461916f, 1.719304323f, 1.723996043f, 1.728694439f, 1.733399391f, 1.738111019f, 1.742829323f, 1.747554302f, 1.752285957f, 1.757024288f, 1.761769295f, 1.766520977f, 1.771279573f, 1.776044846f, 1.780816793f, 1.785595655f, 1.790381074f, 1.795173526f, 1.799972653f, 1.804778576f, 1.809591532f, 1.814411163f, 1.819237709f, 1.82407105f, 1.828911304f, 1.833758473f, 1.838612676f, 1.843473673f, 1.848341703f, 1.853216529f, 1.858098507f, 1.86298728f, 1.867883086f, 1.872785926f, 1.877695799f, 1.882612705f, 1.887536645f, 1.892467618f, 1.897405624f, 1.902350664f, 1.907302856f, 1.912262201f, 1.91722858f, 1.92220211f, 1.927182794f, 1.93217051f, 1.937165499f, 1.94216764f, 1.947176933f, 1.952193499f, 1.957217097f, 1.962248087f, 1.967286348f, 1.972331762f, 1.977384448f, 1.982444406f, 1.987511516f, 1.992586017f, 1.997667909f, 2.002757072f, 2.007853508f, 2.012957335f, 2.018068552f, 2.023186922f, 2.028312922f, 2.033446312f, 2.038586855f, 2.043735027f, 2.048890591f, 2.054053545f, 2.05922389f, 2.064401865f, 2.069587231f, 2.074779987f, 2.079980135f, 2.08518815f, 2.090403318f, 2.095626354f, 2.100856543f, 2.106094599f, 2.111340046f, 2.116593361f, 2.121853828f, 2.127122164f, 2.132398129f, 2.137681484f, 2.142972708f, 2.148271322f, 2.153577805f, 2.158891916f, 2.164213657f, 2.169543266f, 2.174880266f, 2.180225134f, 2.185577631f, 2.190937996f, 2.19630599f, 2.201681852f, 2.207065582f, 2.212456942f, 2.21785593f, 2.223263025f, 2.22867775f, 2.234100103f, 2.239530563f, 2.244968891f, 2.250414848f, 2.255868912f, 2.261330843f, 2.266800642f, 2.27227807f, 2.277763605f, 2.283257008f, 2.288758516f, 2.294267654f, 2.299785137f, 2.305310249f, 2.310843468f, 2.316384792f, 2.321933746f, 2.327491045f, 2.333056211f, 2.338629484f, 2.344210625f, 2.34980011f, 2.355397224f, 2.361002684f, 2.366616249f, 2.372237921f, 2.37786746f, 2.383505344f, 2.389151335f, 2.394805431f, 2.400467634f, 2.406137943f, 2.411816359f, 2.41750288f, 2.423197985f, 2.428900957f, 2.434612274f, 2.440331697f, 2.446059465f, 2.45179534f, 2.457539558f, 2.463291883f, 2.469052553f, 2.474821568f, 2.480598688f, 2.486384153f, 2.492177963f, 2.497980118f, 2.503790617f, 2.509609461f, 2.515436649f, 2.521272182f, 2.527115822f, 2.532968283f, 2.53882885f, 2.544697762f, 2.550575256f, 2.556461096f, 2.56235528f, 2.568258047f, 2.574169159f, 2.580088615f, 2.586016655f, 2.591953278f, 2.597898245f, 2.603851557f, 2.60981369f, 2.615784168f, 2.621763229f, 2.627750635f, 2.633746862f, 2.639751434f, 2.645764589f, 2.651786327f, 2.657816648f, 2.663855553f, 2.66990304f, 2.67595911f, 2.682024002f };
    ncnn::Mat x_mat = rand64(seed % 1000);
    std::vector<float> sigma(step);
    float delta = -999.0f / (step - 1);
    for (int i = 0; i < step; i++)
    {
        float t = 999.0 + i * delta;
        int low_idx = std::floor(t);
        int high_idx = std::ceil(t);
        float w = t - low_idx;
        sigma[i] = std::exp((1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]);
    }
    sigma.push_back(0.f);
    float _norm_[4] = { sigma[0], sigma[0], sigma[0], sigma[0] };
    x_mat.substract_mean_normalize(0, _norm_);
    for (int i = 0; i < static_cast<int>(sigma.size()) - 1; i++) {
        double t1 = ncnn::get_current_time();
        ncnn::Mat denoised = CFGDenoiserCompVisDenoiser(net, log_sigmas, x_mat, sigma[i], c, uc);
        double t2 = ncnn::get_current_time();
        _log("[-Info-]: Denoiser step: " + std::to_string(i) + " took " + std::to_string(t2 - t1) + " Milliseconds.\n");
        float sigma_up = std::min(sigma[i + 1], std::sqrt(sigma[i + 1] * sigma[i + 1] * (sigma[i] * sigma[i] - sigma[i + 1] * sigma[i + 1]) / (sigma[i] * sigma[i])));
        float sigma_down = std::sqrt(sigma[i + 1] * sigma[i + 1] - sigma_up * sigma_up);
        std::srand(std::time(NULL));
        ncnn::Mat randn = rand64(rand() % 1000);
        for (int c = 0; c < 4; c++)
        {
            float* x_ptr = x_mat.channel(c);
            float* d_ptr = denoised.channel(c);
            float* r_ptr = randn.channel(c);
            for (int hw = 0; hw < 64 * 64; hw++)
            {
                *x_ptr = *x_ptr + ((*x_ptr - *d_ptr) / sigma[i]) * (sigma_down - sigma[i]) + *r_ptr * sigma_up;
                x_ptr++;
                d_ptr++;
                r_ptr++;
            }
        }
    }
    ncnn::Mat clone;
    clone.clone_from(x_mat);
    auto elapsed = ncnn::get_current_time() - startingTime;
    _log(std::string{ "Denoising via (UNetModel FP16) took " } + std::to_string(elapsed) + std::string{ " milliseconds to complete.\n" });
    return clone;
}
/*
* The parenthesis "()" is to increase the importance, while the square brackets "[]" is to reduce the importance
*/
std::vector<std::pair<std::string, float>> tinyDream::parsePromptAttention(std::string& texts)
{
    std::vector<std::pair<std::string, float>> res;
    std::stack<int> round_brackets;
    std::stack<int> square_brackets;
    const float round_bracket_multiplier = 1.1;
    const float square_bracket_multiplier = 1 / 1.1;
    std::vector<std::string> ms;
    for (char c : texts) {
        std::string s = std::string(1, c);
        if (s == "(" || s == "[" || s == ")" || s == "]") {
            ms.push_back(s);
        }
        else {
            if (ms.size() < 1)
                ms.push_back("");
            std::string last = ms[ms.size() - 1];
            if (last == "(" || last == "[" || last == ")" || last == "]") {
                ms.push_back("");
            }
            ms[ms.size() - 1] += s;
        }
    }
    for (std::string text : ms) {
        if (text == "(") {
            round_brackets.push(res.size());
        }
        else if (text == "[") {
            square_brackets.push(res.size());
        }
        else if (text == ")" && round_brackets.size() > 0)
        {
            for (unsigned long p = round_brackets.top(); p < res.size(); p++)
            {
                res[p].second *= round_bracket_multiplier;
            }

            round_brackets.pop();
        }
        else if (text == "]" and square_brackets.size() > 0)
        {
            for (unsigned long p = square_brackets.top(); p < res.size(); p++)
            {
                res[p].second *= square_bracket_multiplier;
            }

            square_brackets.pop();
        }
        else {
            res.push_back(make_pair(text, 1.0));
        }
    }
    while (!round_brackets.empty())
    {
        for (unsigned long p = round_brackets.top(); p < res.size(); p++)
        {
            res[p].second *= round_bracket_multiplier;
        }

        round_brackets.pop();
    }
    while (!square_brackets.empty())
    {
        for (unsigned long p = square_brackets.top(); p < res.size(); p++)
        {
            res[p].second *= square_bracket_multiplier;
        }

        square_brackets.pop();
    }
    unsigned long i = 0;
    while (i + 1 < res.size())
    {
        if (res[i].second == res[i + 1].second){
            res[i].first += res[i + 1].first;
            auto it = res.begin();
            res.erase(it + i + 1);
        }else{
            i += 1;
        }
    }
    return res;
}
ncnn::Mat tinyDream::getTextEmbedding(ncnn::Net& net, std::string prompt)
{
    std::vector<std::pair<std::string, float>> parsed = parsePromptAttention(prompt);
    std::vector<std::vector<int>> tokenized;
    for (auto p : parsed){
        std::vector<std::string> tokens = split(p.first);
        std::vector<int> ids;
        for (std::string token : tokens)
            ids.push_back(idxTokens[token]);
        tokenized.push_back(ids);
    }
    std::vector<int> remade_tokens;
    std::vector<float> multipliers;
    {
        int last_comma = -1;
        for (auto it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++)
        {
            std::vector<int> tokens = tokenized[it_tokenized];
            float weight = parsed[it_tokenized].second;
            unsigned long i = 0;
            while (i < tokens.size())
            {
                int token = tokens[i];
                if (token == 267){
                    last_comma = remade_tokens.size();
                }else if ((std::max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) && (remade_tokens.size() - last_comma <= 20)){
                    last_comma += 1;
                    std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
                    std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
                    std::vector<int> _remade_tokens_(remade_tokens.begin(), remade_tokens.begin() + last_comma);
                    remade_tokens = _remade_tokens_;
                    int length = remade_tokens.size();
                    int rem = std::ceil(length / 75.0) * 75 - length;
                    std::vector<int> tmp_token(rem, 49407);
                    remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
                    remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
                    std::vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
                    std::vector<int> tmp_multipliers(rem, 1.0f);
                    _multipliers_.insert(_multipliers_.end(), tmp_multipliers.begin(), tmp_multipliers.end());
                    _multipliers_.insert(_multipliers_.end(), reloc_mults.begin(), reloc_mults.end());
                    multipliers = _multipliers_;
                }
                remade_tokens.push_back(token);
                multipliers.push_back(weight);
                i += 1;
            }
        }
        int prompt_target_length = std::ceil(std::max(int(remade_tokens.size()), 1) / 75.0) * 75;
        int tokens_to_add = prompt_target_length - remade_tokens.size();
        std::vector<int> tmp_token(tokens_to_add, 49407);
        remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
        std::vector<int> tmp_multipliers(tokens_to_add, 1.0f);
        multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
    }
    ncnn::Mat embVec(768, 0);
    {
        while (remade_tokens.size() > 0)
        {
            std::vector<int> rem_tokens(remade_tokens.begin() + 75, remade_tokens.end());
            std::vector<float> rem_multipliers(multipliers.begin() + 75, multipliers.end());
            std::vector<int> current_tokens;
            std::vector<float> current_multipliers;
            if (remade_tokens.size() > 0)
            {
                current_tokens.insert(current_tokens.end(), remade_tokens.begin(), remade_tokens.begin() + 75);
                current_multipliers.insert(current_multipliers.end(), multipliers.begin(), multipliers.begin() + 75);
            }
            else
            {
                std::vector<int> tmp_token(75, 49407);
                current_tokens.insert(current_tokens.end(), tmp_token.begin(), tmp_token.end());
                std::vector<int> tmp_multipliers(75, 1.0f);
                current_multipliers.insert(current_multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
            }
            {
                ncnn::Mat token_mat = ncnn::Mat(77);
                token_mat.fill(int(49406));
                ncnn::Mat multiplier_mat = ncnn::Mat(77);
                multiplier_mat.fill(1.0f);
                int* token_ptr = token_mat;
                float* multiplier_ptr = multiplier_mat;

                for (int i = 0; i < 75; i++)
                {
                    token_ptr[i + 1] = int(current_tokens[i]);
                    multiplier_ptr[i + 1] = current_multipliers[i];
                }

                ncnn::Extractor ex = net.create_extractor();
                ex.set_light_mode(true);
                ex.input("token", token_mat);
                ex.input("multiplier", multiplier_mat);
                ex.input("cond", embVec);
                ncnn::Mat new_conds;
                ex.extract("conds", new_conds);
                embVec = new_conds;
            }
            remade_tokens = rem_tokens;
            multipliers = rem_multipliers;
        }
    }
    return embVec;
}
std::pair<ncnn::Mat, ncnn::Mat> tinyDream::promptEncoder(const std::string& positivePrompt, const std::string& negativePrompt)
{
    ncnn::Net net;
    _log("[-Info-]: Token Embedding - Using Frozen CLIP Text Encoder for encoding prompts...\n");
    auto startingTime = ncnn::get_current_time();
    net.opt.use_vulkan_compute = false;
    net.opt.use_winograd_convolution = false;
    net.opt.use_sgemm_convolution = false;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_packing_layout = true;
    net.load_param(joinPaths("FrozenCLIPEmbedder-fp16.param").c_str());
    net.load_model(joinPaths("FrozenCLIPEmbedder-fp16.bin").c_str());
    auto elapsed = ncnn::get_current_time() - startingTime;
    _log(std::string{ "Frozen CLIP Text Encoder took " } + std::to_string(elapsed) + std::string{ " milliseconds to complete.\n" });
    return std::make_pair(getTextEmbedding(net, positivePrompt), getTextEmbedding(net, negativePrompt));
}
bool tinyDream::loadTokens()
{
    if (idxTokens.empty()) {
        _log("[-Info-]: Token Embedding - Building Tokens Table from Vocabulary\n");
        std::string pathname = joinPaths("vocab.txt");
        std::ifstream infile;
        infile.open(pathname.c_str());
        if (!infile.is_open()) {
            _log(std::string{ "[-ERROR-]: IO error while opening: '" } + pathname + std::string{ "'. Make sure your Pre-Trained Assets & Models are located on a directory accessible to this executable. Download the Pre-Trained Assets & Models from https://pixlab.io/tiny-dream#downloads\n" });
            return false;
        }
        std::string s;
        int idx = 0;
        while (getline(infile, s))
        {
            idxTokens.insert(std::make_pair(s, idx));
            idx++;
        }
        infile.close();
        _log(std::string{ "[-Info-]: Token Embedding - Finished Building (" } + std::to_string(idxTokens.size()) + std::string{ ") Tokens Table\n" });
    }
    else
        _log("[-Info-]: Token Embedding  - Tokens Table Already Loaded. Skipping\n");
    return true;
}
bool tinyDream::dream(const std::string& positivePrompt, const std::string& negativePrompt, std::string& outputImgPath, bool upScale = true, int seed = 42, int step = 30)
{
    if (positivePrompt.empty()) {
        _log(std::string{ "[-ERROR-]: Missing Positive Prompt (Keywords): Describe something you'd like to see generated using words separated by commas. High priority or meta instructions must be surrounded by parenthesis.\n" });
        return false;
    }
    if (negativePrompt.empty()) {
        // Not so fatal
        _log(std::string{ "[-Notice-]: Missing Negative Prompt (Keywords): Describe something you don't like to see generated. Example of such words are: blood, mutilation, gore, genitals, nudity, etc.\n" });
    }
    if (xLog) {
        std::string logmsg{ "[-Info-]: Starting Stable Diffusion Inference with the following configuration:\n" };
        logmsg += std::string{ "\n\tPositive Prompt: " } + positivePrompt;
        if (negativePrompt.size() > 0)
            logmsg += std::string{ "\n\tNegative Prompt: " } + negativePrompt;
        logmsg += std::string{ "\n\tInference Steps: " } + std::to_string(step);
        if (step > 35) {
            logmsg += std::string{ "\n\t\t[-Notice-]: Using more inference steps may produce a slightly different picture, but not necessarily better quality. In addition, the iterative nature of the process makes generation slow; the more steps you'll use, the more time it will take to generate an image!" };
        }
        logmsg += std::string{ "\n\tCurrent Seed Generation: " } + std::to_string(seed) + std::string{ "\n" };
        _log(logmsg);
    }
    auto startingTime = ncnn::get_current_time();
    if (!loadTokens())
        return false; // Check your log handler callback for any error messages
    auto vecEmb = promptEncoder(positivePrompt, negativePrompt);
    auto sample = diffusionSolver(vecEmb.first, vecEmb.second, seed, step);
    ncnn::Mat outMat = decoderSolver(sample);
    auto elapsed = ncnn::get_current_time() - startingTime;
    _log(std::string{ "[-Info-]: 512x512 inference output took " } + std::to_string(elapsed) + std::string{ " milliseconds to complete.\n" });
    // Export Pixel Data
    int w = 512;
    int h = 512;
    if (upScale) {
        outMat = upScale4xEsprGan(outMat);
        w *= 4;
        h *= 4;
        _log(std::string{ "[-Info-]: Upscale of original 512x512 pixels to 2048x2048 high resolution pixels completed." });
    }
    auto zBuf = std::make_unique<unsigned char[]>(w * h * 3/*R G B*/);
    outMat.to_pixels(zBuf.get(), ncnn::Mat::PIXEL_RGB);
    outputImgPath = outputPath + imgPrefix + randomString() + std::string{ "_" } + std::to_string(step) + std::string{ "_" } + std::to_string(seed) + std::string{ ".png" };
    stbi_write_png(outputImgPath.c_str(), w, h, 3 /*R G B*/, zBuf.get(), 3 * w);
    _log(std::string{ "[-Inference Completed-]: Output image: '" } + outputImgPath + std::string{ "' saved to disk." });
    return true;
}