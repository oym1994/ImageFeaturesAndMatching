#pragma once

// Original code from https://github.com/enazoe/yolo-tensorrt
#include "NvInfer.h"
#include "chunk.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <string>

struct TensorInfo {
    std::string blobName;
    float *hostBuffer{nullptr};
    uint64_t volume{0};
    int bindingIndex{-1};
};

class TicAndToc
{
  public:
    TicAndToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

class Logger : public nvinfer1::ILogger
{
private:
    Severity mReportableSeverity;

public:
    Logger(Severity severity = Severity::kINFO) : mReportableSeverity(severity) {}

    //!
    //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
    //!
    //! Note samples should not be calling this function directly; it will
    //! eventually go away once we eliminate the inheritance from
    //! nvinfer1::ILogger
    //!
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= mReportableSeverity) std::cout << msg << std::endl;
        // LogStreamConsumer(mReportableSeverity, severity)
        //     << "[TRT] " << std::string(msg) << std::endl;
    }

    Severity getSeverityLevel() const { return mReportableSeverity; }
};

nvinfer1::ICudaEngine *loadTRTEngine(const std::string planFilePath, Logger &logger);

nvinfer1::ICudaEngine *loadTRTONNXEngine(const std::string planFilePath, Logger &logger);

class TensorRTInferenceGeneric
{
protected:
    Logger m_Logger;
    nvinfer1::ICudaEngine *m_Engine = nullptr;
    int m_InputBindingIndex;
    uint64_t m_InputSize;
    std::vector<uint64_t> m_InputSize_vec;
    nvinfer1::IExecutionContext *m_Context;
    std::vector<void *> m_DeviceBuffers;
    cudaStream_t m_CudaStream;
    mutable std::vector<TensorInfo> m_OutputTensors;
    int m_BatchSize = 1;
    const std::string m_InputBlobName;
    int width  = 640;
    int height = 480;

    const std::vector<std::string> m_InputBlobNameGroup;
    std::vector<int> m_InputBindingIndexGroup;

    mutable float total_time_cost_;
    mutable int total_call_times_;

public:
    TensorRTInferenceGeneric(std::vector<std::string> input_blob_names, int _width, int _height,
                             std::vector<uint64_t> InputSize_vec);

    virtual void doInference(const unsigned char *input, const uint32_t batchSize);

    virtual void doInference(const cv::Mat &input);

    virtual void doInference(const std::vector<cv::Mat> &inputs) const;

    bool verifyEngine();

    void allocateBuffers();

    void init(const std::string &engine_path);

    virtual std::unordered_map<std::string, std::vector<std::vector<int>>> getDynamicInputSizes() = 0;

    nvinfer1::ICudaEngine *getEngine() { return m_Engine; }

    virtual ~TensorRTInferenceGeneric()
    {
        std::cout << "Deconstruct the TensorRTInferenceGeneric!" << std::endl;
        std::cout << "Average cost time(ms) and total call times: " << total_time_cost_/total_call_times_ << " " << total_call_times_  <<std::endl;      
        // delete m_Engine;
        // m_Engine = nullptr;
        // delete m_Context;
        // m_Context = nullptr;
        // m_Engine->destroy();
        // m_Context->destroy();
          
    }

    bool dynamic_axes_;
};
