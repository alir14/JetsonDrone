#include <jetson-inference/detectNet.h>
#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <signal.h>

#include "udp/udpMessagePublisher.h"
#include <cmath>

#ifdef HEADLESS
	#define IS_HEADLESS() "headless"	// run without display
#else
	#define IS_HEADLESS() (const char*)NULL
#endif

bool signal_received = false;

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
    float currX, currY, prevX=0, prevY=0, prevArea=0;
    bool shouldSend = false;

    commandLine cmdLine(argc, argv, IS_HEADLESS());

    LogInfo("Start command detection ...");
    
    UdpMessagePublisher _udpClient;

    _udpClient.initSocket();

    _udpClient.ConnectToServer();

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

            int highConfIndex = 0;

            for(int n=0; n< numDetections; n++)
            {
                if(detections[n].Confidence > detections[highConfIndex].Confidence)
                    highConfIndex = n;
            }

            LogVerbose(" highest index %i and hisghers conf %f \n", highConfIndex,detections[highConfIndex].Confidence);

            detections[highConfIndex].Center(&currX, &currY);
            int distanceX = (currX - prevX);
            int distanceY = (currY - prevY);
            int areaScale = (int)(detections[highConfIndex].Area() - prevArea);

            if(std::abs(distanceX) > 30)
            {
                prevX = currX;
                shouldSend = true;
            }

            if(std::abs(distanceY) > 30)
            {
                prevY = currY;
                shouldSend = true;
            }

            if(std::abs(areaScale) > 30000)
            {
                LogVerbose(" a= %f | pA = %f \n",detections[highConfIndex].Area(), prevArea);
                prevArea = detections[highConfIndex].Area();
                shouldSend = true;
            }

            if(shouldSend)
            {
                std::string distanceCordinate = std::to_string(distanceX) +","+std::to_string(distanceY);
                std::string centerCordnate = std::to_string((int)currX) +"," + std::to_string((int)currY);
                std::string boxCorrdinate = std::to_string((int)detections[highConfIndex].Left)+","+std::to_string((int)detections[highConfIndex].Top)+","+std::to_string((int)detections[highConfIndex].Right)+","+std::to_string((int)detections[highConfIndex].Bottom);
                std::string cmdName = net->GetClassDesc(detections[highConfIndex].ClassID);

                _udpClient.PublishMessage(distanceCordinate+"|" +centerCordnate+"|" +boxCorrdinate+"|"+std::to_string(areaScale)+"|"+cmdName);

                shouldSend = false;
            }
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

    _udpClient.CloseSocket();
    SAFE_DELETE(input);
    SAFE_DELETE(output);
    SAFE_DELETE(net);

    LogVerbose("detectnet:  shutdown complete.\n");

    return 0;
}

