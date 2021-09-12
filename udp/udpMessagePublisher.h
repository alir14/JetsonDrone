#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <string>

class UdpMessagePublisher {
	
public:
	int sockfd;
	struct sockaddr_in server_addr;
	UdpMessagePublisher() {};
	bool initSocket();
	void ConnectToServer();
	void PublishMessage(std::string message);
	void CloseSocket();

private:
	void error(const char* msg);

};
