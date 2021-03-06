#include <iostream>
#include <jetson-inference/detectNet.h>
#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <signal.h>

#ifdef HEADLESS
	#define IS_HEADLESS() "headless"	// run without display
#else
	#define IS_HEADLESS() (const char*)NULL
#endif

//floats
float currentX, currentY , nextX, nextY;

bool signal_received = false;

void MyLog(const char* value)
{
    std::cout << value << std::endl;
}

void sig_handler(int signo)
{
    if(signo == SIGINT)
    {
        LogVerbose("received SIGINT\n");
        signal_received = true;
    }    
}

int main(int argc, char** argv)
{

    commandLine cmdLine(argc, argv, IS_HEADLESS());

    std::cout << "my detection ... project Master" << std::endl;
    
    if(signal(SIGINT, sig_handler) == SIG_ERR)
    {
       LogError("can't catch SIGINT \n");
    }

    videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

    if(!input)
    {
        LogError("detectnet:  failed to create input stream \n");
        return 0;        
    }

    videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));

    if(!output)
    {
        LogError("detectnet:  failed to create output stream \n");
        return 0;
    }

    detectNet* net = detectNet::Create(cmdLine);

    if(!net)
    {
        LogError("detectNet: failed to load the model \n");
        return 0;
    }

    const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));

    while(!signal_received)
    {
        uchar3* img = NULL;
      
        if(!input->Capture(&img, 1000))
        {
            if(!input->IsStreaming())
            {
                break;
            }

            LogError("detectNet: failed to capture video \n");
            continue;
        }

        detectNet::Detection* detections = NULL;

        const int numDetections = net->Detect(img, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);

        if(numDetections > 0)
        {
            LogVerbose("%i objects detected \n", numDetections);

            //for(int n=0; n< numDetections; n++)
            //{
                //LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
                //LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height());
            //}
            
            detections[0].Center(&nextX, &nextY);

            float distanceX = (nextX - currentX);
            float distanceY = (nextY - currentY);

            currentX = nextX;
            currentY = nextY;

            LogVerbose("currentX: %f , currentY: %f \n", currentX, currentY);
            LogVerbose("nextX: %f , nextY: %f \n", nextX, nextY);

            LogVerbose(" moved X %f, moved Y %f \n", distanceX, distanceY);

//            detections[n].Left
//            detections[n].Top, 
//            detections[n].Right, 
//            detections[n].Bottom, 
//            detections[n].Width(), 
//            detections[n].Height()
        }

        if (output != NULL)
        {
            output->Render(img, input->GetWidth(), input->GetHeight());

            char str[256];

            sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());

            output->SetStatus(str);

            if(!output->IsStreaming())
            {
                signal_received = true;
            }
        }

        // net->PrintProfilerTime();
    }

    LogVerbose("detectnet:  shutting down...\n");

    SAFE_DELETE(input);
    SAFE_DELETE(output);
    SAFE_DELETE(net);

    LogVerbose("detectnet:  shutdown complete.\n");

    return 0;

}

