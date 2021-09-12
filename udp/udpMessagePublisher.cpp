#include "udpMessagePublisher.h"

void UdpMessagePublisher::error(const char* msg)
{
    perror(msg);
    exit(0);
}

bool UdpMessagePublisher::initSocket()
{
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    if (sockfd < 0)
    {
        error("ERROR opening socket");
        return false;
    }

    bzero((char*)&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;

    inet_pton(AF_INET, "192.168.1.79", &server_addr.sin_addr);

    server_addr.sin_port = htons(54000);

    return true;
}

void UdpMessagePublisher::ConnectToServer()
{
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0)
        error("ERROR connecting");
}

void UdpMessagePublisher::PublishMessage(std::string message)
{
    int msgSize = message.length() + 1;
    
    printf("message size: %i\n", msgSize);

    int sendOk = sendto(sockfd, message.c_str(), msgSize, 0, (struct sockaddr*)&server_addr, sizeof(server_addr));

    printf("result : %i\n", sendOk);
}

void UdpMessagePublisher::CloseSocket()
{
    close(sockfd);
}
