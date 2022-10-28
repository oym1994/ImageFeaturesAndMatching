#include "tensorRT/tensorrt_generic.h"
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <memory>
#include <unistd.h>
using namespace nvinfer1;
uint64_t get3DTensorVolume4(nvinfer1::Dims inputDims);

nvinfer1::ICudaEngine *loadTRTEngine(const std::string planFilePath, /*PluginFactory* pluginFactory,*/
                                     Logger &logger)
{
    // reading the model in memory
    std::cout << "Loading TRT Engine..." << std::endl;
    // assert(std::fileExists(planFilePath));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(planFilePath, std::ios::binary | std::ios::in);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculating model size
    trtModelStream.seekg(0, std::ios::end);
    const auto modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void *modelMem = malloc(modelSize);
    trtModelStream.read((char *)modelMem, modelSize);

    nvinfer1::IRuntime *runtime   = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(modelMem, modelSize /*, pluginFactory*/);
    free(modelMem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}

// tensorRT V8
nvinfer1::ICudaEngine *loadTRTONNXEngine(const std::string onnxFile, Logger &logger, TensorRTInferenceGeneric *trt_model)
{
    std::cout << "Loading TRT ONNX..." << std::endl;

    IBuilder *builder = createInferBuilder(logger);

    auto config = builder->createBuilderConfig();

    if (trt_model->dynamic_axes_) {
        auto profile              = builder->createOptimizationProfile();
        auto dynamic_inputs_sizes = trt_model->getDynamicInputSizes();
        for (auto dynamic_input_sizes : dynamic_inputs_sizes) {
            // assert(trt_model->getEngine()->getBindingIndex(dynamic_input_sizes.first.c_str()) != -1 &&
            //        "Invalide dynamic axis name");
            // assert(dynamic_input_sizes.second.size() == 3 && "Invalide dynamic axis size");

            if (dynamic_input_sizes.second[0].size() == 3) {  // temprally just for superglue
                profile->setDimensions(dynamic_input_sizes.first.c_str(), OptProfileSelector::kMIN,
                                       Dims3(dynamic_input_sizes.second[0][0], dynamic_input_sizes.second[0][1],
                                             dynamic_input_sizes.second[0][2]));
                profile->setDimensions(dynamic_input_sizes.first.c_str(), OptProfileSelector::kOPT,
                                       Dims3(dynamic_input_sizes.second[1][0], dynamic_input_sizes.second[1][1],
                                             dynamic_input_sizes.second[1][2]));
                profile->setDimensions(dynamic_input_sizes.first.c_str(), OptProfileSelector::kMAX,
                                       Dims3(dynamic_input_sizes.second[2][0], dynamic_input_sizes.second[2][1],
                                             dynamic_input_sizes.second[2][2]));

            } else if (dynamic_input_sizes.second[0].size() == 2) {
                profile->setDimensions(dynamic_input_sizes.first.c_str(), OptProfileSelector::kMIN,
                                       Dims2(dynamic_input_sizes.second[0][0], dynamic_input_sizes.second[0][1]));
                profile->setDimensions(dynamic_input_sizes.first.c_str(), OptProfileSelector::kOPT,
                                       Dims2(dynamic_input_sizes.second[1][0], dynamic_input_sizes.second[1][1]));
                profile->setDimensions(dynamic_input_sizes.first.c_str(), OptProfileSelector::kMAX,
                                       Dims2(dynamic_input_sizes.second[2][0], dynamic_input_sizes.second[2][1]));
            }
        }
        config->addOptimizationProfile(profile);
    }

    if (builder->platformHasFastFp16()) {
        std::cout << "float16 is supported in this platform" << std::endl;
        // builder->setFp16Mode(true); //TensorRT 7
        // config->setFlags(static_cast<nvinfer1::BuilderFlags>(nvinfer1::BuilderFlag::kFP16));  // TensorRT8
    }

    config->setMaxWorkspaceSize(1 << 30);

    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);

    builder->setMaxBatchSize(1);
    INetworkDefinition *network = builder->createNetworkV2(1);
    auto parse                  = nvonnxparser::createParser(*network, logger);

    parse->parseFromFile(onnxFile.c_str(), static_cast<int>(logger.getSeverityLevel()));  // ILogger::Severity::kWARNING

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

    if (!engine) {
        std::cout << "cant create engine" << std::endl;
    }

    // save engine
    auto pos               = onnxFile.find('.');
    std::string engin_path = onnxFile.substr(0, pos) + ".engine";
    std::cout << "engine path: " << engin_path << std::endl;

    bool file_exist = (access(engin_path.c_str(), 0) == 0);

    if (!file_exist) {
        std::ofstream p(engin_path, std::ios::binary);
        if (!p) {
            std::cerr << "could not engine output file" << std::endl;
        }

        nvinfer1::IHostMemory *seriallizedModel = engine->serialize();
        p.write(reinterpret_cast<const char *>(seriallizedModel->data()), seriallizedModel->size());
        seriallizedModel->destroy();
    }

    builder->destroy();
    parse->destroy();
    network->destroy();

    return engine;
}

// TensorRT7
/*nvinfer1::ICudaEngine *loadTRTONNXEngine(const std::string onnxFile, Logger &logger)
{
    IBuilder *builder = createInferBuilder(logger);

    if (builder->platformHasFastFp16()) {
        std::cout << "float16 is supported in this platform" << std::endl;
        // builder->setFp16Mode(true);
    }

    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);  // 2^30
    INetworkDefinition *network = builder->createNetworkV2(1);
    auto parse                  = nvonnxparser::createParser(*network, logger);

    parse->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kINFO));  // ILogger::Severity::kWARNING

    nvinfer1::ICudaEngine *engine = builder->buildCudaEngine(*network);

    if (!engine) {
        std::cout << "cant create engine" << std::endl;
    }

    // save engine
    auto pos               = onnxFile.find('.');
    std::string engin_path = onnxFile.substr(0, pos) + ".engine";
    std::cout << "engine path: " << engin_path << std::endl;

    bool file_exist = (access(engin_path.c_str(), 0) == 0);

    if (!file_exist) {
        std::ofstream p(engin_path, std::ios::binary);
        if (!p) {
            std::cerr << "could not engine output file" << std::endl;
        }

        nvinfer1::IHostMemory *seriallizedModel = engine->serialize();
        p.write(reinterpret_cast<const char *>(seriallizedModel->data()), seriallizedModel->size());
        seriallizedModel->destroy();
    }

    builder->destroy();
    parse->destroy();
    network->destroy();

    return engine;
}*/

TensorRTInferenceGeneric::TensorRTInferenceGeneric(std::vector<std::string> input_blob_names, int _width, int _height,
                                                   std::vector<uint64_t> InputSize_vec) :
    m_InputBlobName(input_blob_names[0]),
    width(_width),
    height(_height),
    m_InputBlobNameGroup(input_blob_names),
    m_InputSize_vec(InputSize_vec),
    m_InputSize(InputSize_vec[0]),
    dynamic_axes_(false),
    total_call_times_(0)
{
    assert(InputSize_vec.size() == input_blob_names.size());
}

void TensorRTInferenceGeneric::init(const std::string &onnxFile)
{
    // find engine
    auto pos                = onnxFile.find('.');
    std::string engine_path = onnxFile.substr(0, pos) + ".engine";
    std::cout << "engine path: " << engine_path << std::endl;

    bool file_exist = (access(engine_path.c_str(), 0) == 0);

    if (file_exist) {
        m_Engine = loadTRTEngine(engine_path, m_Logger);
    } else {
        file_exist = (access(onnxFile.c_str(), 0) == 0);
        if (!file_exist) {
            std::cerr << onnxFile + " not exist" << std::endl;
            abort();
        }
        m_Engine = loadTRTONNXEngine(onnxFile, m_Logger, this);
    }
    assert(m_Engine != nullptr);

    m_Context = m_Engine->createExecutionContext();
    assert(m_Context != nullptr);

    if (dynamic_axes_) m_Context->setOptimizationProfile(0);

    for (unsigned int i = 0; i < m_Engine->getNbBindings(); i++) {
        std::string name(m_Engine->getBindingName(i));
        std::cout << "TensorRT binding index " << i << " name " << name << std::endl;
    }

    m_InputBindingIndex = m_Engine->getBindingIndex(m_InputBlobName.c_str());

    m_InputBindingIndexGroup.resize(m_InputBlobNameGroup.size());
    for (int i = 0; i < m_InputBlobNameGroup.size(); i++) {
        m_InputBindingIndexGroup[i] = m_Engine->getBindingIndex(m_InputBlobNameGroup[i].c_str());
        assert(m_InputBindingIndexGroup[i] != -1);
        std::cout << "input binding index: " << m_InputBindingIndexGroup[i] << std::endl;
        std::cout << "input name: " << m_InputBlobNameGroup[i] << std::endl;
        std::cout << "MaxBatchSize: " << m_Engine->getMaxBatchSize() << std::endl;
    }

    assert(m_InputBindingIndex != -1);
    std::cout << "MaxBatchSize" << m_Engine->getMaxBatchSize() << std::endl;
    assert(m_BatchSize <= static_cast<uint32_t>(m_Engine->getMaxBatchSize()));
    allocateBuffers();
    NV_CUDA_CHECK(cudaStreamCreate(&m_CudaStream));
    if (!dynamic_axes_) assert(verifyEngine());

    // std::cout << "TensorRT workspace " << m_Engine->getWorkspaceSize() / 1024.0 << "kb" << std::endl;
}

void TensorRTInferenceGeneric::doInference(const cv::Mat &input)
{
    // assert(input.channels() == 1 && "Only support 1 channel now");
    // This function is very slow event on i7, we need to optimize it
    // But not now.
    doInference(input.data, 1);
}

void TensorRTInferenceGeneric::doInference(const std::vector<cv::Mat> &inputs)
{
    const uint32_t batchSize = 1;

    assert(inputs.size() == m_InputBlobNameGroup.size() && "Invalid input sizes");

    // Timer timer;
    assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");

    for (int i = 0; i < m_InputBlobNameGroup.size(); i++) {
        const unsigned char *input = inputs[i].data;
        NV_CUDA_CHECK(cudaMemcpyAsync(m_DeviceBuffers.at(m_InputBindingIndexGroup[i]), input,
                                      batchSize * inputs[i].cols * inputs[i].rows * inputs[i].channels() * sizeof(float),
                                      cudaMemcpyHostToDevice, m_CudaStream));
    }
    m_Context->enqueueV2(m_DeviceBuffers.data(), m_CudaStream, nullptr);  // executeV2
    for (auto &tensor : m_OutputTensors) {
        NV_CUDA_CHECK(cudaMemcpyAsync(tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
                                      batchSize * tensor.volume * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream));
    }
    cudaStreamSynchronize(m_CudaStream);
}

void TensorRTInferenceGeneric::doInference(const unsigned char *input, const uint32_t batchSize)
{
    // Timer timer;
    assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
    NV_CUDA_CHECK(cudaMemcpyAsync(m_DeviceBuffers.at(m_InputBindingIndex), input,
                                  batchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream));

    m_Context->enqueue(batchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
    for (auto &tensor : m_OutputTensors) {
        NV_CUDA_CHECK(cudaMemcpyAsync(tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
                                      batchSize * tensor.volume * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream));
    }
    cudaStreamSynchronize(m_CudaStream);
    //	timer.out("inference");
}

bool TensorRTInferenceGeneric::verifyEngine()
{
    assert((m_Engine->getNbBindings() == (m_InputBindingIndexGroup.size() + m_OutputTensors.size()) &&
            "Binding info doesn't match between cfg and engine file \n"));

    for (auto tensor : m_OutputTensors) {
        assert(!strcmp(m_Engine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str()) &&
               "Blobs names dont match between cfg and engine file \n");
        std::cout << "output tensor name: " << tensor.blobName.c_str() << ", "
                  << get3DTensorVolume4(m_Engine->getBindingDimensions(tensor.bindingIndex)) << ":" << tensor.volume
                  << std::endl;
        assert(get3DTensorVolume4(m_Engine->getBindingDimensions(tensor.bindingIndex)) == tensor.volume &&
               "Tensor volumes dont match between cfg and engine file \n");
    }

    for (unsigned int i = 0; i < m_InputBindingIndexGroup.size(); i++) {
        assert(m_Engine->bindingIsInput(m_InputBindingIndexGroup[i]) && "Incorrect input binding index \n");
        assert(m_Engine->getBindingName(m_InputBindingIndexGroup[i]) == m_InputBlobNameGroup[i] &&
               "Input blob name doesn't match between config and engine file");
        assert(get3DTensorVolume4(m_Engine->getBindingDimensions(m_InputBindingIndexGroup[i])) == m_InputSize_vec[i]);
    }

    assert(m_Engine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
    assert(m_Engine->getBindingName(m_InputBindingIndex) == m_InputBlobName &&
           "Input blob name doesn't match between config and engine file");
    assert(get3DTensorVolume4(m_Engine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);
    return true;
}

void TensorRTInferenceGeneric::allocateBuffers()
{
    m_DeviceBuffers.resize(m_Engine->getNbBindings(), nullptr);
    assert(m_InputBindingIndex != -1 && "Invalid input binding index");

    for (unsigned int i = 0; i < m_InputBindingIndexGroup.size(); i++) {
        NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndexGroup[i]),
                                 m_BatchSize * m_InputSize_vec[i] * sizeof(float)));
    }

    for (auto &tensor : m_OutputTensors) {
        tensor.bindingIndex = m_Engine->getBindingIndex(tensor.blobName.c_str());
        std::cout << "Tensor " << tensor.blobName.c_str() << " bind to " << tensor.bindingIndex << " dim "
                  << m_Engine->getBindingDimensions(tensor.bindingIndex).d[0] << " "
                  << m_Engine->getBindingDimensions(tensor.bindingIndex).d[1] << " "
                  << m_Engine->getBindingDimensions(tensor.bindingIndex).d[2] << " "
                  << m_Engine->getBindingDimensions(tensor.bindingIndex).d[3] << std::endl;
        assert((tensor.bindingIndex != -1) && "Invalid output binding index");
        NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(tensor.bindingIndex), m_BatchSize * tensor.volume * sizeof(float)));
        NV_CUDA_CHECK(cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float)));
    }
}

uint64_t get3DTensorVolume4(nvinfer1::Dims inputDims)
{
    int ret = 1;
    for (int i = 0; i < inputDims.nbDims; i++) {
        ret = ret * inputDims.d[i];
    }
    return ret;
}
