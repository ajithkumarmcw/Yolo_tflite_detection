#include "yolov5_tflite.h"

void YOLOV5::getLabelsName(std::string path, std::vector<std::string> &labelNames)
{
    // Open the File
    std::ifstream in(path.c_str());
    // Check if object is valid
    if (!in)
        throw std::runtime_error("Can't open ");
    std::string str;
    // Read the next line from File until it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if (str.size() > 0)
            labelNames.push_back(str);
    }
    // Close The File
    in.close();
}

void YOLOV5::loadModel(const  std::string path)
{

    _model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!_model)
    {
        std::cout << "\nFailed to load the model.\n"
                  << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nModel loaded successfully.\n";
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);
    TfLiteStatus status = _interpreter->AllocateTensors();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to allocate the memory for tensors.\n"
                  << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nMemory allocated for tensors.\n";
    }

    // input information
    _input = _interpreter->inputs()[0];
    TfLiteIntArray *dims = _interpreter->tensor(_input)->dims;
    _in_height = dims->data[1];
    _in_width = dims->data[2];
    _in_channels = dims->data[3];
    _in_type = _interpreter->tensor(_input)->type;
    _input_8 = _interpreter->typed_tensor<uint8_t>(_input);
    _float_16 = _interpreter->typed_tensor<float>(_input);

    _interpreter->SetNumThreads(nthreads);
}

void YOLOV5::preprocess(cv::Mat &image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(_in_height, _in_width), cv::INTER_CUBIC);
    //image.convertTo(image, CV_16U);
}

template <typename T>
void YOLOV5::fill(T *in, cv::Mat &src)
{
    int n = 0, nc = src.channels(), ne = src.elemSize();
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            for (int c = 0; c < nc; ++c)
                in[n++] = src.data[y * src.step + x * ne + c];
}

std::vector<std::vector<float>> YOLOV5::tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum)
{

    auto scale = pOutputTensor->params.scale;
    auto zero_point = pOutputTensor->params.zero_point;
    std::vector<std::vector<float>> v;
    for (int32_t i = 0; i < row; i++)
    {
        std::vector<float> _tem;
        for (int j = 0; j < colum; j++)
        {
            float val_float = (((int32_t)pOutputTensor->data.uint8[i * colum + j]) - zero_point) * scale;
            _tem.push_back(val_float);
        }
        v.push_back(_tem);
    }
    return v;
}

std::vector<std::vector<float>> YOLOV5::tensorToVector2Dfloat32(TfLiteTensor *pOutputTensor, const int &row, const int &colum)
{

    auto scale = pOutputTensor->params.scale;
    auto zero_point = pOutputTensor->params.zero_point;
    std::vector<std::vector<float>> v;
    for (int32_t i = 0; i < row; i++) {
      std::vector<float> _tem;
      for (int32_t j = 0; j <colum; j++) {
        float val_float = pOutputTensor->data.f[i * colum + j];
        _tem.push_back(val_float);
      }
      v.push_back(_tem);
    }
    return v;

  // std::vector<std::vector<float>> tensorData;
  // //if (pOutputTensor->dims->size == 2) {

  //   const float* inputData = pOutputTensor->data.f;
  //   const int numRows = pOutputTensor->dims->data[1];
  //   const int numCols = pOutputTensor->dims->data[2];

  //   tensorData.resize(numRows, std::vector<float>(numCols));

  //   for (int i = 0; i < numRows; i++) {
  //     for (int j = 0; j < numCols; j++) {
  //       tensorData[i][j] = inputData[i*numCols + j];
  //     }
  //   }
  // //}
  // return tensorData;

  // std::vector<std::vector<float>> tensorData;
  // //if (pOutputTensor->dims->size == 2) {
  //   auto scale = pOutputTensor->params.scale;
  //   auto zero_point = pOutputTensor->params.zero_point;


  //   const float* inputData = pOutputTensor->data.f;
  //   const int numRows = pOutputTensor->dims->data[1];
  //   const int numCols = pOutputTensor->dims->data[2];

  //   tensorData.resize(numRows, std::vector<float>(numCols));

  //   for (int i = 0; i < numRows; i++) {
  //     for (int j = 0; j < numCols; j++) {
  //       tensorData[i][j] = inputData[i*numCols + j] - zero_point * scale;
  //       //float val_float = (((int32_t)pOutputTensor->data.uint8[i * colum + j]) - zero_point) * scale;
  //     }
  //   }
  // //}
  // return tensorData;


}


void YOLOV5::nonMaximumSupprition(
    std::vector<std::vector<float>> &predV,
    const int &row,
    const int &colum,
    std::vector<cv::Rect> &boxes,
    std::vector<float> &confidences,
    std::vector<int> &classIds,
    std::vector<int> &indices)

{

    std::vector<cv::Rect> boxesNMS;
    int max_wh = 40960;
    std::vector<float> scores;
    double confidence;
    cv::Point classId;

    for (int i = 0; i < row; i++)
    {
        if (predV[i][4] > confThreshold)
        {
            // height--> image.rows,  width--> image.cols;
            int left = (predV[i][0] - predV[i][2] / 2) * _img_width;
            int top = (predV[i][1] - predV[i][3] / 2) * _img_height;
            int w = predV[i][2] * _img_width;
            int h = predV[i][3] * _img_height;

            for (int j = 5; j < colum; j++)
            {
                // # conf = obj_conf * cls_conf
                scores.push_back(predV[i][j] * predV[i][4]);
            }

            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            scores.clear();
            int c = classId.x * max_wh;
            if (confidence > confThreshold)
            {
                boxes.push_back(cv::Rect(left, top, w, h));
                confidences.push_back(confidence);
                classIds.push_back(classId.x);
                boxesNMS.push_back(cv::Rect(left, top, w, h));
            }
        }
    }
    cv::dnn::NMSBoxes(boxesNMS, confidences, confThreshold, nmsThreshold, indices);
}

void YOLOV5::run(cv::Mat frame, Prediction &out_pred)
{

    _img_height = frame.rows;
    _img_width = frame.cols;
     int _input_type_i;
    _input_type_i = _interpreter->tensor(_input)->type;

    preprocess(frame);
   
    if (_input_type_i ==  kTfLiteFloat32){ 
      std::cout << "reach input float" <<std::endl;     
      fill(_float_16, frame);
    } else if (_input_type_i ==  kTfLiteUInt8) {
      std::cout << "reach input 8" <<std::endl;
      fill(_input_8, frame);
    } else {
      std::cout << "YOLO Model Input type donot support yet\n";
      exit(0);
    }

    // Inference
    TfLiteStatus status = _interpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to run inference!!\n";
        exit(1);
    }

    int _out = _interpreter->outputs()[0];
    TfLiteIntArray *_out_dims = _interpreter->tensor(_out)->dims;
    int _out_row   = _out_dims->data[1];   // 25200
    int _out_colum = _out_dims->data[2];   // class number + 5 ---> 85     bbox cond class
    // int _out_type  = _interpreter->tensor(_out)->type;
    std::vector<std::vector<float>> predV ;
    TfLiteTensor *pOutputTensor = _interpreter->tensor(_interpreter->outputs()[0]);
    if ((pOutputTensor->type == kTfLiteInt8) || (pOutputTensor->type == kTfLiteUInt8)) {
      predV = tensorToVector2D(pOutputTensor, _out_row, _out_colum);
    } else if (pOutputTensor->type == kTfLiteFloat32) {
       std::cout << "reach ktflite32 poutput"<< std::endl;
       predV = tensorToVector2Dfloat32(pOutputTensor, _out_row, _out_colum);
    } else {
      std::cout << "Unsupported output type!\n" << pOutputTensor->type << "\n";
      //predV = tensorToVector2D(pOutputTensor, _out_row, _out_colum);
      std::cout << __FILE__ << ": " << __LINE__ << std::endl;
      //exit(-1);
    }

    std::vector<int> indices;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    nonMaximumSupprition(predV, _out_row, _out_colum, boxes, confidences, classIds, indices);

    for (int i = 0; i < indices.size(); i++)
    {
        out_pred.boxes.push_back(boxes[indices[i]]);
        out_pred.scores.push_back(confidences[indices[i]]);
        out_pred.labels.push_back(classIds[indices[i]]);
    }
    std::cout << "completed post processing" << std::endl;
};