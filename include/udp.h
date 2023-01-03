#ifndef _UDP_H
#define _UDP_H

#include "dada_util.h"
#include "dada_header.h"

#include <math.h>

//#define TIMEIT
#define MJD1970 40587.0 ///< MJD days in 1970.01.01
#define SECDAY  86400.0 ///< Seconds in a days

#define STR_BUFLEN 1024

//#define PKT_DTSZ   8192
#define PKT_DTSZ   4096
#define PKT_HDRSZ  8
#define PKTSZ      (PKT_HDRSZ+PKT_DTSZ)

#define NTHREADS_GENERATE 1
#define NTHREADS_UDPGEN   (NTHREADS_GENERATE+1)

#define NTHREADS_UDP2DB 2

#define CODIF_SUMMARY_MAX_GROUPS CODIF_SUMMARY_MAX_THREADS

const int    report_period_microsecond = 1000000; ///< report interval in microseconds
const double report_period_second      = report_period_microsecond/1.0E6; ///< report interval in seconds
const double bits2gbits                = pow(1000.0, 3);  ///< Convert from bites to Gb
//const double bits2gbits     = pow(1024.0, 3);  ///< Convert from bites to Gb 

const unsigned window_bytes = 1000*pow(1024.0, 2); ///< 1000 MB in bytes
const int reuse = 1; ///< reust address 

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

// packet header has a one-byte flag and a 7-bytes counter
typedef struct packet_header_t{
  uint64_t flag : 8;
  uint64_t counter : 56; // for 512 MHz bandwidth, it could cover about 800 days???
}packet_header_t;

typedef struct generate_t{
  dada_header_t dada_header;    ///< configuration fromd dada header
  double rate;              ///< Required data rate in Gbps
  char ip_src[STR_BUFLEN];  ///< Source IP, I do not want the program to choose one by itself
  int port_src;             ///< Source port, I do not want the program to choose one by itself
  char ip_dest[STR_BUFLEN]; ///< Destination IP, the program could not choose one by itself
  int port_dest;            ///< Destination port, the program could not choose one by itself
}generate_t;

// For now we only need nant and npol
// We may need more in future, that is why I have a struct here
typedef struct receiver_t{
  key_t key;
  char fname[STR_BUFLEN];
  char ip[STR_BUFLEN];
  int port;

  int nframe;

  double mjd_start;
  dada_header_t dada_header;
}receiver_t;

int seconds2dhms(uint64_t seconds, char dhms[STR_BUFLEN]);

#endif
