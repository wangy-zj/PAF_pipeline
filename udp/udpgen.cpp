#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <getopt.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#include <sys/socket.h>
#include <arpa/inet.h>

#include "udp.h"

// The following copied from https://stackoverflow.com/questions/41390824/%C2%B5s-precision-wait-in-c-for-linux-that-does-not-put-program-to-sleep
# define tscmp(a, b, CMP)			\
  (((a)->tv_sec == (b)->tv_sec) ?		\
   ((a)->tv_nsec CMP (b)->tv_nsec) :		\
   ((a)->tv_sec CMP (b)->tv_sec))
# define tsadd(a, b, result)				\
  do {							\
    (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;	\
    (result)->tv_nsec = (a)->tv_nsec + (b)->tv_nsec;	\
    if ((result)->tv_nsec >= 1000000000)		\
      {							\
	++(result)->tv_sec;				\
	(result)->tv_nsec -= 1000000000;		\
      }							\
  } while (0)
# define tssub(a, b, result)				\
  do {							\
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;	\
    (result)->tv_nsec = (a)->tv_nsec - (b)->tv_nsec;	\
    if ((result)->tv_nsec < 0) {			\
      --(result)->tv_sec;				\
      (result)->tv_nsec += 1000000000;			\
    }							\
  } while (0)

int main(int argc, char *argv[]){
  signal(SIGINT, catch_int);

  int nsecond = 10;
  
  struct option options[] = {
    {"nsecond", required_argument, 0, 'n'},
    {"help",    no_argument,       0, 'h'}, 
    {0,         0, 0, 0}
  };

  /* parse command line arguments */
  while (1) {
    int ss;
    int opt=getopt_long_only(argc, argv, "n:h", 
			     options, NULL);
    
    if (opt==EOF)
      break;

    switch (opt) {

    case 'n':
      ss = sscanf(optarg, "%d", &nsecond);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR: Could not parse nsecond from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;


    case 'h':
      fprintf(stdout,
	      "udpgen - A program to generate UDP packets\n"
	      "\n"
	      "Usage: udpgen [options]\n"
	      " -nsecond/-n <int> Require to send number of seconds data, [default %d]\n"
	      " -help/-h          Show help\n",
	      nsecond);
      exit(EXIT_FAILURE);
    }
  }
  
  fprintf(stdout, "UDPGEN_INFO: Require to send %d seconds data\n", nsecond);

  int socks[NSTREAM_UDP] = {0};
  for(int i = 0; i < NSTREAM_UDP; i++) {    
    /* Setup source socket */
    socks[i] = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (setsockopt(socks[i], SOL_SOCKET, SO_SNDTIMEO, (const char*)&tout, sizeof(tout))){
      fprintf(stderr, "UDPGEN_ERROR: Could not setup RECVTIMEO to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      IP_SEND, PORT_SEND, __FILE__, __LINE__);
      
      close(socks[i]);
      exit(EXIT_FAILURE);
    }

    if (setsockopt(socks[i], SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))){
      fprintf(stderr, "UDPGEN_ERROR: Could not enable REUSEADDR to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      IP_SEND, PORT_SEND, __FILE__, __LINE__);
    
      close(socks[i]);
      exit(EXIT_FAILURE);
    }

    if (setsockopt(socks[i], SOL_SOCKET, SO_SNDBUF, &window_bytes, sizeof(window_bytes))) {
      fprintf(stderr, "UDPGEN_ERROR: Could not set socket RCVBUF to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      IP_SEND, PORT_SEND, __FILE__, __LINE__);
    
      close(socks[i]);
      exit(EXIT_FAILURE);
    }

    
    // source address
    struct sockaddr_in src = {0};
    socklen_t src_len = sizeof(struct sockaddr_in);
    uint32_t port = PORT_SEND+i;        
    src.sin_family      = AF_INET;
    src.sin_port        = htons(port);
    src.sin_addr.s_addr = inet_addr(IP_SEND);
    
    fprintf(stdout, "UDPGEN_INFO: required port is %d\n", port);
    fprintf(stdout, "UDPGEN_INFO: actual port is %d\n", ntohs(src.sin_port));    
        
    if (bind(socks[i], (struct sockaddr *)&src, src_len)){
      fprintf(stderr, "UDPGEN_ERROR: Can not bind to %s_%d, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      inet_ntoa(src.sin_addr), ntohs(src.sin_port), __FILE__, __LINE__);
    
      close(socks[i]);
      exit(EXIT_FAILURE);
    }
    
    fprintf(stdout, "UDPGEN_INFO: bind to %s_%d\n",
	  inet_ntoa(src.sin_addr), ntohs(src.sin_port));
  }

  // dest address 
  struct sockaddr_in dest = {0};
  socklen_t dest_len = sizeof(dest);
  dest.sin_family      = AF_INET;
  dest.sin_port        = htons(PORT_RECV);
  dest.sin_addr.s_addr = inet_addr(IP_RECV);

  struct timespec now   = {0};
  struct timespec then  = {0};
  struct timespec start = {0};
  uint32_t nsleep = 1000*PKT_DURATION - nsleep_dead;  
  struct timespec sleep = {0, nsleep};
  fprintf(stdout, "UDPGEN_INFO: nsleep is %d\n", nsleep);

  /* Get packet and send it */
  char buf[PKTSZ] = {'0'};
  packet_header_t *packet_header = (packet_header_t *)buf;
  //packet_header->flag = 1;
  
  // setup counter with real time stamp  
  time_t tmi;
  time(&tmi);
  struct tm* utc = gmtime(&tmi);
  int hour = utc->tm_hour;
  int min  = utc->tm_min;
  int sec  = utc->tm_sec;
  double microsecond_offset = 1E6*(hour*3600+min*60+sec); // can only be as accurate as 1 second
  packet_header->counter = (uint64_t)(microsecond_offset/(double)PKT_DURATION);
  fprintf(stdout, "UDPGEN_INFO: microsecond_offset is %.6f microseconds\n", microsecond_offset);
  fprintf(stdout, "UDPGEN_INFO: First counter is %" PRIu64 "\n", packet_header->counter);
  
  uint64_t npacket = 1E6*nsecond/(double)PKT_DURATION;
  double nsecond_send = 1.0E-6*npacket*PKT_DURATION;
  fprintf(stdout, "UDPGEN_INFO: Will send % " PRIu64 " packets for each stream\n", npacket);
  fprintf(stdout, "UDPGEN_INFO: Will send %.6f seconds data for each stream\n", nsecond_send);

  uint64_t npacket_report = 1E6*nsecond_report/(double)PKT_DURATION;
  double time_report      = 1.0E-6*npacket_report*PKT_DURATION;
  uint64_t bytes_report   = PKTSZ*NSTREAM_UDP*npacket_report;
  fprintf(stdout, "UDPGEN_INFO: Will report traffic every % " PRIu64 " packets\n", npacket_report);
  fprintf(stdout, "UDPGEN_INFO: Will report traffic every % " PRIu64 " bytes\n", bytes_report);
  fprintf(stdout, "UDPGEN_INFO: Required to report traffic every %.6f seconds\n\n", time_report);

  int npacket_sent = 0;
  struct timeval previous_time = {0};
  struct timeval current_time  = {0};
  
  gettimeofday(&previous_time, NULL);
  while(npacket_sent < npacket){
    // Start time
    clock_gettime( CLOCK_REALTIME, &start);    
    tsadd( &start, &sleep, &then );

    for(int i = 0; i < NSTREAM_UDP; i++){
      packet_header->flag = AD0+i;
      sendto(socks[i], buf, PKTSZ, 0, (struct sockaddr *)&dest, dest_len);
    }
    packet_header->counter++;
    npacket_sent++;

    if(npacket_sent%npacket_report == 0){
      gettimeofday(&current_time, NULL);      
      double elapsed_time = (current_time.tv_sec - previous_time.tv_sec) +
	    (current_time.tv_usec - previous_time.tv_usec)/1.0E6L;
      double data_rate = 1.0E-6*bytes_report/elapsed_time;
      
      fprintf(stdout, "UDPGEN_INFO: Required to report traffic every %.6f seconds\n", time_report);
      fprintf(stdout, "UDPGEN_INFO: Report traffic after %.6f seconds\n", elapsed_time);
      fprintf(stdout, "UDPGEN_INFO: Data rate is %.6f Mbps\n\n", data_rate);
      
      gettimeofday(&previous_time, NULL);      
    }
    
    // Busy loop until we get required delay
    clock_gettime( CLOCK_REALTIME, &now);
    while ( tscmp( &now, &then, < )  ){
      clock_gettime( CLOCK_REALTIME, &now);
    }
  }

  // close sockets
  for(int i = 0; i < NSTREAM_UDP; i++){
    close(socks[i]);
  }
  
  return EXIT_SUCCESS;
}
