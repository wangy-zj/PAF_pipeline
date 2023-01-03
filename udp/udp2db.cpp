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
#include "../include/dada_header.h"

// ./udp2db -i 10.17.4.2 -p 12345 -f /data/den15c/.udp-pipeline/header/512MHz_1ant1pol.header -k a000 -m 56400

const struct timeval tout = {0, 0};

int capture(receiver_t conf);

void usage(){
  fprintf(stdout,
	  "udp2db - A simple program to receive UDP packet\n"
	  "\n"
	  "Usage:\tudp2db [options]\n"
	  " -ip/-i        <string> Source IP address\n"
	  " -port/-p      <int>    Source port number\n"
	  " -fname/-f     <string> DADA header file name\n"
	  " -mjd_start/-m <double> Start MJD\n"
	  " -key/-k       <key>    Hexadecimal shared memory key of PSRDADA ring buffer for data capture \n"
	  " -help/-h               Show help\n"
	  );
}

int main(int argc, char *argv[]){
  receiver_t conf = {0};
  
  struct option options[] = {
			     {"ip",        required_argument, 0, 'i'},
			     {"port",      required_argument, 0, 'p'},
			     {"fname",     required_argument, 0, 'f'},
			     {"mjd_start", required_argument, 0, 'm'},
			     {"key",       required_argument, 0, 'k'},
			     {"help",      no_argument,       0, 'h'}, 
			     {0,           0, 0, 0}
  };
  
  /* parse command line arguments */
  while (1) {
    int ss;
    int opt=getopt_long_only(argc, argv, "i:p:m:f:k:h", 
			     options, NULL);
    
    if (opt==EOF)
      break;

    switch (opt) {
      
    case 'i':
      ss = sscanf(optarg, "%s", conf.ip);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.ip from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;
    
    case 'p':
      ss = sscanf(optarg, "%d", &conf.port);
      if (ss!=1){
	fprintf(stderr, "UDPGEN_ERROR:\tCould not parse conf.port from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'f':
      ss = sscanf(optarg, "%s", conf.fname);
      if (ss!=1){
	fprintf(stderr, "UDP2DB_ERROR:\tCould not parse DADA header file name from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
    break;

    case 'm':
      ss = sscanf(optarg, "%lf", &conf.mjd_start);
      if (ss!=1){
	fprintf(stderr, "UDP2DB_ERROR:\tCould not parse mjd_start from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	exit(EXIT_FAILURE);
      }
    break;

    case 'k':
      ss = sscanf(optarg, "%x", &conf.key);
      if(ss != 1) {
	fprintf(stderr, "UDP2DB_ERROR:\tCould not parse key from %s, \n", optarg);
	fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	usage();
	
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'h':
      usage();
      exit(EXIT_FAILURE);
    }
  }
  
  /* Print out command line options */
  fprintf(stdout, "UDP2DB_INFO:\tDADA header file name is %s\n", conf.fname);
  fprintf(stdout, "UDP2DB_INFO:\tmjd_start is %f\n", conf.mjd_start);
  fprintf(stdout, "UDP2DB_INFO:\tkey is %x\n", conf.key);
  
  /* Read in DADA header file and get more required configurations */
  read_dada_header_from_file(conf.fname, &conf.dada_header);
  fprintf(stdout, "UDP2DB_INFO:\tDone with dada header configuration\n");

  /* Now do real thing */
  capture(conf);
  
  return EXIT_SUCCESS;
}

int capture(receiver_t conf){
  key_t key = conf.key;
  char *ip = conf.ip;
  int port = conf.port;
  double tsamp  = conf.dada_header.tsamp;
  
  /* Create socket and set it up */
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP); // domain, type, protocol

  // TIMEOUT HERE MAYBE SENSITIVITY  
  if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout))){
    fprintf(stderr, "UDP2DB_ERROR:\tCould not setup RECVTIMEO to %s_%d, "
  	    "which happens at \"%s\", line [%d], has to abort.\n",
  	    ip, port, __FILE__, __LINE__);
    
    close(sock);
    
    exit(EXIT_FAILURE);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))){
    fprintf(stderr, "UDP2DB_ERROR:\tCould not enable REUSEADDR to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    ip, port, __FILE__, __LINE__);
    
    close(sock);

    exit(EXIT_FAILURE);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &window_bytes, sizeof(window_bytes))) {
    fprintf(stderr, "UDP2DB_ERROR:\tCould not set socket RCVBUF to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    ip, port, __FILE__, __LINE__);
    
    close(sock);

    exit(EXIT_FAILURE);
  }

  struct sockaddr_in sa = {0};
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(port);
  sa.sin_addr.s_addr = inet_addr(ip);
  
  if (bind(sock, (struct sockaddr *)&sa, sizeof(sa))){
    fprintf(stderr, "UDP2DB_ERROR:\tCan not bind to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }

  // receive the first packet to figure out reference time
  char dbuf[PKTSZ] = {0};
  struct sockaddr_in fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  
  // To get the first packet of the while test
  // this is supposed to happen after the first start command is issued
  if (recvfrom(sock, (void *)dbuf, PKTSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1){
    fprintf(stderr, "UDP2DB_ERROR:\tCan not receive data from %s_%d, "
	    "which happens at \"%s\", line [%d]\n", 
	    inet_ntoa(sa.sin_addr), ntohs(sa.sin_port),
	    __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  } // if

  packet_header_t *packet_header = (packet_header_t *)dbuf;
  uint64_t counter = packet_header->counter;

  /***************************
   // get mjd_start
   // this may not be precise enough
   **************************/
  double mjd_start = conf.mjd_start + counter*tsamp/(1E6*SECDAY);
  
  /* Setup DADA ring buffer */
  dada_hdu_t *hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(hdu, key);
  if(dada_hdu_connect(hdu) < 0){ 
    fprintf(stderr, "UDP2DB_ERROR:\tCan not connect to hdu, \n");
    fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }
  
  // Now we can lock HDU for write and get data and header block
  if(dada_hdu_lock_write(hdu) < 0) {
    fprintf(stderr, "UDP2DB_ERROR:\tError locking HDU, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  } // if

  /*  Get data and header block ready */
  ipcbuf_t *data_block   = (ipcbuf_t *)(hdu->data_block);
  ipcbuf_t *header_block = (ipcbuf_t *)(hdu->header_block);

  /*
    register dada header, we should get more information in real case 
    for now just copy it from header template file */
  char *hdrbuf = ipcbuf_get_next_write(header_block);
  
  if(fileread(conf.fname, hdrbuf, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "UDP2DB_ERROR:\tError reading header file, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }

  // setup mjd_start for reference time 
  if (ascii_header_set(hdrbuf, "MJD_START", "%f", mjd_start) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  // donot set header parameters anymore
  if(ipcbuf_mark_filled(header_block, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "UDP2DB_ERROR:\tError header_fill, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
  // Create buffer to hold data for one block
  uint64_t bufsz = ipcbuf_get_bufsz(data_block);
  int npacket_perblock = (int)bufsz/(double)PKT_DTSZ;
  if(bufsz%(PKT_DTSZ) != 0){
    fprintf(stderr, "UDP2DB_ERROR:\tbad choice of ring buffer block size, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }
  char *databuf = ipcbuf_get_next_write(data_block); // Open a ring buffer block
  
  // Setup reference and previous information
  // previous information is for traffic monitoring
  // reference information is for block location calculation
  int referencecounter = counter;
  int previouscounter  = counter;

  int64_t current_losscounter = 0;
  uint64_t current_pktcounter = 0;
  int64_t previous_losscounter = current_losscounter;
  uint64_t previous_pktcounter = current_pktcounter;
 
  struct timeval previous_time = {0};
  struct timeval current_time  = {0};

  gettimeofday(&current_time, NULL);
  previous_time = current_time;
  uint64_t reference_second = current_time.tv_sec;
    
  while (true){    
    // receive data
    char dbuf[PKTSZ] = {0};
    struct sockaddr_in fromsa = {0};
    socklen_t fromlen = sizeof(fromsa);
  
    // To get the first packet of the while test
    // this is supposed to happen after the first start command is issued
    if (recvfrom(sock, (void *)dbuf, PKTSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1){
      fprintf(stderr, "UDP2DB_ERROR:\tCan not receive data from %s_%d, "
	      "which happens at \"%s\", line [%d]\n", 
	      inet_ntoa(sa.sin_addr), ntohs(sa.sin_port),
	      __FILE__, __LINE__);

      exit(EXIT_FAILURE);
    } // if

    packet_header_t *packet_header = (packet_header_t *)dbuf;
    uint64_t thiscounter = packet_header->counter;

#ifdef DEBUG
    fprintf(stdout, "UDP2DB_DEBUG:\tcounter is %" PRIu64 ", and from %s at \"%s\", line [%d]\n",
    	    thiscounter, inet_ntoa(fromsa.sin_addr), __FILE__, __LINE__);
#endif
    
    // really necessary to lock and unlock here?    
    current_losscounter+=(thiscounter-previouscounter-1);
    current_pktcounter++;
    
    //当前数据包时间减去参考时间，对包定位
    int loc_packet = thiscounter - referencecounter;
    
    // WE may have multiple blocks lost ...
    // loc_packet可能会比npacket_perblock大好几倍，循环进行block填充
    while(loc_packet >= npacket_perblock){
      ipcbuf_mark_filled(data_block, bufsz); // Open a ring buffer block
      
      referencecounter += npacket_perblock;
      loc_packet -= npacket_perblock;
      databuf = ipcbuf_get_next_write(data_block); // Open a ring buffer block
    }
    
    uint64_t offset = loc_packet*PKT_DTSZ;
    memcpy(databuf+offset, dbuf+PKT_HDRSZ, PKT_DTSZ);
    
    previouscounter = thiscounter;

    // Check traffic status
    gettimeofday(&current_time, NULL);

    // Report traffic status when we at report time interval
    if(((current_time.tv_sec * 1000000 + current_time.tv_usec) -
	(previous_time.tv_sec * 1000000 + previous_time.tv_usec)) > report_period_microsecond){

      uint64_t diff_pktcounter = current_pktcounter - previous_pktcounter;
      int64_t diff_losscounter = current_losscounter - previous_losscounter;

      // Figure out time so far
      uint64_t diff_second = current_time.tv_sec - reference_second;
      char diff_dhms[STR_BUFLEN] = {0};
      seconds2dhms(diff_second, diff_dhms);
      
      // Only print out information when there is data in the last monitor period
      fprintf(stdout, "UDP2DB_INFO:\ttime so far %s, in the last %3.1f seconds data rate is %6.3f Gbps, \n",
      	      diff_dhms, report_period_second,
      	      diff_pktcounter*PKT_DTSZ*8/(report_period_second*bits2gbits));
      
      fprintf(stdout, "UDP2DB_INFO:\ttime so far %s, %" PRIu64 " packet received, %" PRId64 " packet lost, packet loss rate is %3.1E\n",
      	      diff_dhms, diff_pktcounter, diff_losscounter, diff_losscounter/(double)(diff_pktcounter+diff_losscounter));
      
      fprintf(stdout, "UDP2DB_INFO:\ttime so far %s, %" PRIu64 " packet received, %" PRId64 " packet lost, packet loss rate is %3.1E so far\n\n",
      	      diff_dhms, current_pktcounter, current_losscounter, current_losscounter/(double)(current_pktcounter+current_losscounter));
      
      // Update counter and timer
      previous_pktcounter  = current_pktcounter;
      previous_losscounter = current_losscounter;
      previous_time = current_time;
    }
  }
  close(sock);

  return EXIT_SUCCESS;
}
