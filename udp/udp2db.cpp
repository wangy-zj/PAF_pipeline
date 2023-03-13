#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <getopt.h>
#include <assert.h>
#include <string.h>

#include <sys/time.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <time.h>

#include "udp.h"

//#define DEBUG

int main(int argc, char *argv[]){

  signal(SIGINT, catch_int);

  char fname[STR_BUFLEN] = {0};
  double freq = 1400.0;
  int nblock  = 100;
  int nsecond = 10;
  key_t key = 0x0000a000;
  
  sprintf(fname, "../../header/paf_test.header");
  
  struct option options[] = {
    {"fname",   required_argument, 0, 'f'},
    {"freq",    required_argument, 0, 'F'},
    {"nblock",  required_argument, 0, 'n'},
    {"nsecond", required_argument, 0, 'N'},
    {"key",     required_argument, 0, 'k'},
    {"help",    no_argument,       0, 'h'}, 
    {0,         0, 0, 0}
  };
  
  /* parse command line arguments */
  while (1) {
    int ss;
    int opt=getopt_long_only(argc, argv, "f:F:n:N:k:h", 
			     options, NULL);
    
    if (opt==EOF)
      break;

    switch (opt) {
            
    case 'f':
      ss = sscanf(optarg, "%s", fname);
      if (ss!=1){
	      fprintf(stderr, "UDP2DB_ERROR: Could not parse DADA header file name from %s, \n", optarg);
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	      exit(EXIT_FAILURE);
      }
      break;

    case 'F':
      ss = sscanf(optarg, "%lf", &freq);
      if (ss!=1){
	      fprintf(stderr, "UDP2DB_ERROR: Could not parse freq from %s, \n", optarg);
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	      exit(EXIT_FAILURE);
      }
      break;

    case 'n':
      ss = sscanf(optarg, "%d", &nblock);
      if (ss!=1){
	      fprintf(stderr, "UDP2DB_ERROR: Could not parse nblock from %s, \n", optarg);
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	      exit(EXIT_FAILURE);
      }
      break;

    case 'N':
      ss = sscanf(optarg, "%d", &nsecond);
      if (ss!=1){
	      fprintf(stderr, "UDP2DB_ERROR: Could not parse nsecond from %s, \n", optarg);
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);
	
	      exit(EXIT_FAILURE);
      }
      break;

    case 'k':
      ss = sscanf(optarg, "%x", &key);
      if(ss != 1) {
        fprintf(stderr, "UDP2DB_ERROR: Could not parse key from %s, \n", optarg);
        fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n",  __FILE__, __LINE__);	
        exit(EXIT_FAILURE);
      }
      break;
      
    case 'h':
      fprintf(stdout,
	      "udp2db - A program to receive UDP packets and write data into ring buffer blocks\n"
	      "\n"
	      "Usage: udp2db [options]\n"
	      " -fname/-f   <string> DADA header template file name, [default %s]\n"
	      " -freq/-F    <double> Center frequency in MHz, [default %.6f MHz]\n"
	      " -nblock/-n  <int>    Report traffic status every these number of blocks, [default %d]\n"
	      " -nsecond/-N <int>    Number of seconds data to each DADA file, [default %d]\n"
	      " -key/-k     <key>    Hexadecimal shared memory key of PSRDADA ring buffer to write data, [default %x] \n"
	      " -help/-h             Show help\n",
	      fname, freq, nblock, nsecond, key);
      exit(EXIT_FAILURE);
    }
  }
  
  /* Print out command line options */
  fprintf(stdout, "UDP2DB_INFO: DADA header file name is %s\n", fname);
  fprintf(stdout, "UDP2DB_INFO: nblock is %d\n", nblock);
  fprintf(stdout, "UDP2DB_INFO: nsecond is %d\n", nsecond);
  fprintf(stdout, "UDP2DB_INFO: key is %x\n", key);
  
  /* Create socket and set it up */
  int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout))){
    fprintf(stderr, "UDP2DB_ERROR: Could not setup RECVTIMEO to %s_%d, "
  	    "which happens at \"%s\", line [%d], has to abort.\n",
  	    IP_RECV, PORT_RECV, __FILE__, __LINE__);
    
    close(sock);
    
    exit(EXIT_FAILURE);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse))){
    fprintf(stderr, "UDP2DB_ERROR: Could not enable REUSEADDR to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    IP_RECV, PORT_RECV, __FILE__, __LINE__);
    
    close(sock);

    exit(EXIT_FAILURE);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &window_bytes, sizeof(window_bytes))) {
    fprintf(stderr, "UDP2DB_ERROR: Could not set socket RCVBUF to %s_%d, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    IP_RECV, PORT_RECV, __FILE__, __LINE__);
    
    close(sock);

    exit(EXIT_FAILURE);
  }

  struct sockaddr_in sa = {0};
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(PORT_RECV);
  sa.sin_addr.s_addr = inet_addr(IP_RECV);
  
  if (bind(sock, (struct sockaddr *)&sa, sizeof(sa))){
    fprintf(stderr, "UDP2DB_ERROR: Can not bind to %s_%d, "
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
    fprintf(stderr, "UDP2DB_ERROR: Can not receive data from %s_%d, "
	    "which happens at \"%s\", line [%d]\n", 
	    inet_ntoa(sa.sin_addr), ntohs(sa.sin_port),
	    __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  } // if
  
  packet_header_t *packet_header = (packet_header_t *)dbuf;
  // this will discard the first time stamp
  // we need it when there is multiple data streams to make traffic report looks better
  uint64_t counter = packet_header->counter + 1; 
  fprintf(stdout, "UDP2DB_INFO: counter is %" PRIu64 "\n", counter);
  
  /***************************
  assume that: 
  1. mjd is a integer number
  2. counter defines offset from mjd 
  **************************/
  time_t tmi;
  time(&tmi);
  struct tm* utc = gmtime(&tmi);
  int year = utc->tm_year + 1900;
  int mon  = utc->tm_mon + 1;
  int mday = utc->tm_mday;
  uint64_t mjd = gregorian_calendar_to_mjd(year, mon, mday);

  fprintf(stdout, "UDP2DB_INFO: year is %d\n", year);
  fprintf(stdout, "UDP2DB_INFO: mon is  %d\n", mon);
  fprintf(stdout, "UDP2DB_INFO: mday is %d\n", mday);
  fprintf(stdout, "UDP2DB_INFO: mjd is %" PRIu64 "\n", mjd);
  
  double timestamp     = (double)counter*PKT_DURATION; // current time stamp in microseconds from mjd
  double microseconds  = fmod(timestamp, 1.0E6); // fraction seconds in microseconds
  uint64_t picoseconds = microseconds*1000; // microseconds to picoseconds
  
  double seconds   = 1.0E-6*(timestamp-microseconds); // minus fraction seconds
  double mjd_start = mjd + seconds/(double)SECDAY; //

  fprintf(stdout, "UDP2DB_INFO: PKT_DURATION is %E microsecond\n", PKT_DURATION);
  fprintf(stdout, "UDP2DB_INFO: timestamp is %E microsecond\n", timestamp);
  fprintf(stdout, "UDP2DB_INFO: fractional seconds is %.3f microsecond\n", microseconds);
  fprintf(stdout, "UDP2DB_INFO: picoseconds is %" PRIu64 "\n", picoseconds);
  fprintf(stdout, "UDP2DB_INFO: integer seconds %.0f seconds\n", seconds);
  fprintf(stdout, "UDP2DB_INFO: mjd_start is %.15f \n", mjd_start);
  
  /* Setup DADA ring buffer */
  dada_hdu_t *hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(hdu, key);
  if(dada_hdu_connect(hdu) < 0){ 
    fprintf(stderr, "UDP2DB_ERROR: Can not connect to HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }
  
  // Now we can lock HDU for write and get data and header block
  if(dada_hdu_lock_write(hdu) < 0) {
    fprintf(stderr, "UDP2DB_ERROR: Error locking HDU, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  } // if

  /*  Get data and header block ready */
  ipcbuf_t *data_block   = (ipcbuf_t *)(hdu->data_block);
  ipcbuf_t *header_block = (ipcbuf_t *)(hdu->header_block);

  // check to see if we can fit integer number of packets of all streams into a single ring buffer block
  uint64_t bufsz = ipcbuf_get_bufsz(data_block);
  if(bufsz%(PKT_DTSZ*NSTREAM_UDP) != 0){
    fprintf(stderr, "UDP2DB_ERROR: bad choice of ring buffer block size, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }
  int npacket = bufsz/(PKT_DTSZ*NSTREAM_UDP); //单个port发送包的数目
  fprintf(stdout, "UDP2DB_INFO: bufsz is % " PRIu64"\n", bufsz);
  fprintf(stdout, "UDP2DB_INFO: npacket is % " PRIu64"\n", npacket);

  uint64_t npacket_expected = npacket*NSTREAM_UDP*nblock; // number of packet expected of each report cycle
  double block_duration  = 1.0E-6*npacket*PKT_DURATION;   //单个block的数据时间
  double report_interval = nblock*block_duration;         //报告间隔，每隔nblock报告一次
  fprintf(stdout, "UDP2DB_INFO: npacket_expected is % " PRIu64"\n", npacket_expected);
  fprintf(stdout, "UDP2DB_INFO: block_duration is %.6f seconds\n", block_duration);
  fprintf(stdout, "UDP2DB_INFO: report_interval is %.6f seconds\n", report_interval);

  int nblock_expected   = (int)(nsecond/block_duration);    //给定时间内预期接收block数目
  double nsecond_record = nblock_expected*block_duration;   //实际接收到预期block所用的时间
  fprintf(stdout, "UDP2DB_INFO: nblock_expected is %d\n", nblock_expected);
  fprintf(stdout, "UDP2DB_INFO: asked for %d seconds data, will record %.6f seconds data\n", nsecond, nsecond_record);

  /*
    register dada header, we should get more information in real case 
    for now just copy it from header template file 
  */
  char *hdrbuf = ipcbuf_get_next_write(header_block);
  if(fileread(fname, hdrbuf, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "UDP2DB_ERROR: Error reading header file, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }

  // setup mjd_start for reference time
  // we should not use the tmi as tmi is clock time on local computer, not time stamps in data
  // 需要从udp包计算时间，目前是直接获取计算机的当前时间 
  char utc_start[STR_BUFLEN] = {0};
  time_t seconds_from1970 = SECDAY*(mjd-MJD1970) + seconds;
  strftime (utc_start, STR_BUFLEN, DADA_TIMESTR, gmtime(&seconds_from1970));

  fprintf(stdout, "UDP2DB_INFO: seconds_from1970 is %d\n", seconds_from1970);
  fprintf(stdout, "UDP2DB_INFO: utc_start is %s\n", utc_start);

  // need to understand how data streams are sorted
  // otherwise I can not make bandwidth and nchan right
  uint64_t bytes_per_second = 1E6*PKT_DTSZ*NSTREAM_UDP/(double)PKT_DURATION;  //每秒传输的数据量
  uint64_t file_size        = bytes_per_second*nsecond_record;                //计算实际每个file的大小
  fprintf(stdout, "UDP2DB_INFO: bytes_per_second is %" PRIu64 "\n", bytes_per_second);
  fprintf(stdout, "UDP2DB_INFO: file_size is %" PRIu64 "\n", file_size);
  
  if (ascii_header_set(hdrbuf, "TSAMP", "%f", TSAMP) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  if (ascii_header_set(hdrbuf, "FREQ", "%f", freq) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting FREQ, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  if (ascii_header_set(hdrbuf, "MJD_START", "%.15f", mjd_start) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  if (ascii_header_set(hdrbuf, "UTC_START", "%s", utc_start) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting UTC_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  if (ascii_header_set(hdrbuf, "PICOSECONDS", "%" PRIu64 "", picoseconds) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting PICOSECONDS, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  
  if (ascii_header_set(hdrbuf, "BYTES_PER_SECOND", "%" PRIu64 "", bytes_per_second) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting BYTES_PER_SECOND, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(hdrbuf, "FILE_SIZE", "%" PRIu64 "", file_size) < 0)  {
    fprintf(stderr, "UDP2DB_ERROR: Error setting FILE_SIZE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  // donot set header parameters anymore
  if(ipcbuf_mark_filled(header_block, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "UDP2DB_ERROR: Error header_fill, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
     
  // start to receive data
  // discard the first packet
  int counter0 = counter;
  int nblock_recorded = 0;
  uint64_t npacket_recorded = 0;
  
  char *databuf = ipcbuf_get_next_write(data_block); // Open a ring buffer block
  while (true){    
    // receive data
    char dbuf[PKTSZ] = {0};
    struct sockaddr_in fromsa = {0};
    socklen_t fromlen = sizeof(fromsa);
  
    // To get the first packet of the while test
    // this is supposed to happen after the first start command is issued
    if (recvfrom(sock, (void *)dbuf, PKTSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1){
      fprintf(stderr, "UDP2DB_ERROR: Can not receive data from %s_%d, "
	      "which happens at \"%s\", line [%d]\n", 
	      inet_ntoa(sa.sin_addr), ntohs(sa.sin_port),
	      __FILE__, __LINE__);

      exit(EXIT_FAILURE);
    } // if

    packet_header_t *packet_header = (packet_header_t *)dbuf;
    uint64_t counter = packet_header->counter;
    uint64_t ad      = packet_header->flag - AD0;
    
#ifdef DEBUG
    fprintf(stdout, "UDP2DB_DEBUG: counter is %" PRIu64 ", and from %s_%d at \"%s\", line [%d]\n",
    	    counter, inet_ntoa(fromsa.sin_addr), ntohs(fromsa.sin_port), __FILE__, __LINE__);
#endif

    int diff_counter = counter - counter0;
    int loc_packet   = diff_counter*NSTREAM_UDP+ad;  //包位置信息，考虑到不同ad的包
    
    // We only cope with packet loss within a single buffer block
    if(diff_counter >= npacket){
      ipcbuf_mark_filled(data_block, bufsz); // Open a ring buffer block
      
      counter0   += npacket;
      loc_packet -= (npacket*NSTREAM_UDP);
      nblock_recorded++;
      
      if(nblock_recorded%nblock == 0){
	      // now we can report traffic status
	      int64_t npacket_lost = npacket_expected - npacket_recorded;
	      double lost_rate     = npacket_lost/(double)npacket_expected;
	
	      fprintf(stdout, "UDP2DB_INFO: %d buffer block recorded\n", nblock_recorded);	
        fprintf(stdout, "UDP2DB_INFO: time so far %15.6f seconds\n", nblock_recorded*block_duration);
        fprintf(stdout, "UDP2DB_INFO: report interval is %.6f seconds\n", report_interval);
        fprintf(stdout, "UDP2DB_INFO:  %" PRIu64 " packets expected\n", npacket_expected);
        fprintf(stdout, "UDP2DB_INFO:  %" PRIu64 " packets recorded\n", npacket_recorded);
        fprintf(stdout, "UDP2DB_INFO:  %" PRId64 " packets lost\n", npacket_lost);
        fprintf(stdout, "UDP2DB_INFO:  packet loss rate is %.6f\n\n", lost_rate);
	
	      npacket_recorded=0;
      }
      //接收到所有的包之后，关闭传输
      if(nblock_recorded == nblock_expected){

	      fprintf(stdout, "UDP2DB_INFO: got %d blocks, expected %d blocks, exit\n", nblock_recorded, nblock_expected);
	      close(sock);
	      if(dada_hdu_unlock_write(hdu) < 0) {
	        fprintf(stderr, "UDP2DB_ERROR:\tunlock hdu write failed, "
		        "which happens at \"%s\", line [%d], has to abort.\n",
		        __FILE__, __LINE__);
	        exit(EXIT_FAILURE);
	      } // if

	      dada_hdu_destroy(hdu); // it has disconnect
	
	      return EXIT_SUCCESS;
      }
      
      databuf = ipcbuf_get_next_write(data_block); // Open a ring buffer block
    }
    
    if(diff_counter >= 0){
      // here we discard late packets
      uint64_t offset = loc_packet*PKT_DTSZ;
      memcpy(databuf+offset, dbuf+PKT_HDRSZ, PKT_DTSZ);
      npacket_recorded++;
    }
  }
}
