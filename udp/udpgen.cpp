#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <getopt.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#include <sys/socket.h>
#include <arpa/inet.h>

#include "../include/udp.h"

const struct timeval tout = {0, 0}; ///< time out for sending
const uint32_t nsleep_dead = 45;  ///< Estimated dead time of sleep 

int generate(generate_t conf);

void usage(){
  fprintf(stdout,
	  "udpgen - A simple program to generate UDP packets\n"
	  "\n"
	  "Usage:\tudpgen [options]\n"
	  " -ip_src/-i        <string> Source IP address\n"
	  " -port_src/-p      <int>    Source port number\n"
	  " -ip_dest/-I       <string> Destination IP address\n"
	  " -port_dest/-P     <int>    Destination port number\n"
	  " -rate/-r          <double> Required data rate in Gbps\n"
	  " -help/-h                   Show help\n"
	  );
}

// ./udpgen -i 10.17.4.2 -p 12346 -I 10.17.4.2 -P 12345 -r 10
// LD_PRELOAD=libvma.so ./udpgen -i 10.17.4.2 -p 12346 -I 10.17.4.2 -P 12345 -r 10

int main(int argc, char *argv[]){
  struct option options[] = {
			     {"ip_src",     required_argument, 0, 'i'},
			     {"port_src",   required_argument, 0, 'p'},
			     {"ip_dest",    required_argument, 0, 'I'},
			     {"port_dest",  required_argument, 0, 'P'},
			     {"rate",       required_argument, 0, 'r'},
			     {"help",       no_argument, 0, 'h'}, 
			     {0,            0, 0, 0}
  };

  generate_t conf = {0};
  
  /* parse command line arguments */
  while (1) {
    int ss;
    int opt=getopt_long_only(argc, argv, "I:P:i:p:r:h", 
			     options, NULL);
    
    if (opt==EOF)
      break;

    switch (opt) {
    case 'i':
      ss = sscanf(optarg, "%s", conf.ip_src);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.ip_src from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;
    
    case 'p':
      ss = sscanf(optarg, "%d", &conf.port_src);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.port_src from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'I':
      ss = sscanf(optarg, "%s", conf.ip_dest);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.ip_dest from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;
    
    case 'P':
      ss = sscanf(optarg, "%d", &conf.port_dest);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.port_dest from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;

    case 'r':
      ss = sscanf(optarg, "%lf", &conf.rate);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.rate from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;
          
    case 'h':
      usage();
      exit(EXIT_FAILURE);
    }
  }
  
  /* Print out command line options */
  fprintf(stdout, "UDPGEN_INFO:\tsource is %s:%d\n",      conf.ip_src, conf.port_src);
  fprintf(stdout, "UDPGEN_INFO:\tdestination is %s:%d\n", conf.ip_dest, conf.port_dest);
  fprintf(stdout, "UDPGEN_INFO:\trequired bandwidth is %f Gbps\n", conf.rate);

  generate(conf);
  
  return EXIT_SUCCESS;
}

int generate(generate_t conf){
  
  /* Setup source socket */
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  // TIMEOUT HERE MAYBE SENSITIVITY  
  if (setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tout, sizeof(tout))){
    fprintf(stderr, "UDPGEN_ERROR:\tCould not setup RECVTIMEO to %s_%d, "
  	    "which happens at \"%s\", line [%d], has to abort.\n",
  	    conf.ip_src, conf.port_src, __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))){
    fprintf(stderr, "UDPGEN_ERROR:\tCould not enable REUSEADDR to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    conf.ip_src, conf.port_src, __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &window_bytes, sizeof(window_bytes))) {
    fprintf(stderr, "UDPGEN_ERROR:\tCould not set socket RCVBUF to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    conf.ip_src, conf.port_src, __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }

  struct sockaddr_in src_addr = {0};
  socklen_t src_addrlen = sizeof(src_addr);
  src_addr.sin_family      = AF_INET;
  src_addr.sin_port        = htons(conf.port_src);
  src_addr.sin_addr.s_addr = inet_addr(conf.ip_src);
  //src_addr.sin_addr.s_addr = INADDR_ANY;
  
  if (bind(sock, (struct sockaddr *)&src_addr, src_addrlen)){
    fprintf(stderr, "UDPGEN_ERROR:\tCan not bind to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    inet_ntoa(src_addr.sin_addr), ntohs(src_addr.sin_port), __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }

  /* Get packet and send it */
  char buf[PKTSZ] = {'0'};
  packet_header_t *packet_header = (packet_header_t *)buf;
  packet_header->flag    = 1;
  packet_header->counter = 0;
  
  struct sockaddr_in dest_addr = {0};
  socklen_t dest_addrlen = sizeof(dest_addr);
  dest_addr.sin_family      = AF_INET;
  dest_addr.sin_port        = htons(conf.port_dest);
  dest_addr.sin_addr.s_addr = inet_addr(conf.ip_dest);

  struct timespec now   = {0};
  struct timespec then  = {0};
  struct timespec start = {0};
  uint32_t nsleep = (uint32_t)(1.E9*PKT_DTSZ*8/(bits2gbits*conf.rate)) - nsleep_dead;
  
  fprintf(stdout, "UDPGEN_INFO:\tnsleep is %d\n", nsleep);
  
  struct timespec sleep = {0, nsleep};

  struct timeval previous_time = {0};
  struct timeval current_time  = {0};

  gettimeofday(&current_time, NULL);
  previous_time = current_time;
  uint64_t reference_second = current_time.tv_sec;

  uint64_t counter = 0;
  while(true){
    // Start time
    clock_gettime( CLOCK_REALTIME, &start);    
    tsadd( &start, &sleep, &then ); // then = start + sleep

    sendto(sock, buf, PKTSZ, 0, (struct sockaddr *)&dest_addr, dest_addrlen);

    packet_header->counter++;
    counter++;
    
    // Check traffic status
    gettimeofday(&current_time, NULL);
    
    // Report traffic status when we at report time interval
    if(((current_time.tv_sec * 1000000 + current_time.tv_usec) -
	(previous_time.tv_sec * 1000000 + previous_time.tv_usec)) > report_period_microsecond){
      
      uint64_t diff_second = current_time.tv_sec - reference_second;
      char diff_dhms[STR_BUFLEN] = {0};
      seconds2dhms(diff_second, diff_dhms);

      fprintf(stdout, "UDPGEN_INFO:\ttime so far %s, data rate in the last %3.1f seconds is %6.3f Gbps, "
	      "%6" PRIu64 " packet generated\n",
	      diff_dhms, report_period_second, counter*PKT_DTSZ*8/(report_period_second*bits2gbits),
	      counter);
            
      counter = 0;
      previous_time = current_time;
    }
    
    // Busy loop until we get required delay
    clock_gettime( CLOCK_REALTIME, &now);
    while ( tscmp( &now, &then, < )  ){
      clock_gettime( CLOCK_REALTIME, &now);
    }

#ifdef TIMEIT
    double elapsed_time = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec)/1.0E9L;
    fprintf(stdout, "UDPGEN_INFO:\telapsed_time for a single sendto loop is %E nanosecond\n", 1.0E9*elapsed_time);
#endif
  }

  return EXIT_SUCCESS;
}

