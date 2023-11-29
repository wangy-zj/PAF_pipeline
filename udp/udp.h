/* udp.h

  contains all functions and defines related to the udp2dp

*/

#ifndef _UDP_H
#define _UDP_H

#include <math.h>
#include <signal.h>

#include "ipcio.h"
#include "futils.h"
#include "ipcbuf.h"
#include "dada_def.h"
#include "ascii_header.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "dada_def.h"

#define MJD1970     40587.0f ///< MJD days in 1970.01.01
#define SECDAY      86400.0f ///< Seconds in a days

#define STR_BUFLEN  1024

// defines of UDP packet 
#define PKT_DTSZ    4096
#define PKT_HDRSZ   8
#define PKTSZ       (PKT_HDRSZ+PKT_DTSZ)

#define N_CHAN      256
#define FREQ        1400.0f //MHz
#define CHAN_WIDTH  0.5f // MHz
#define SAMPSZ      2 // 2 bytes, 1 real and 1 imag

#define N_TIMESTEP_PER_PKT      (PKT_DTSZ/(N_CHAN*SAMPSZ)) // number of time steps per packet
#define TSAMP          (1/CHAN_WIDTH) // microseconds of each time step
#define PKT_DURATION   (N_TIMESTEP_PER_PKT*TSAMP)  // microseconds of each packet

// defines of network interfaces
#define IP_SEND     "10.101.5.33"
#define IP_RECV     "10.101.5.33"
#define PORT_SEND   10000 
#define PORT_RECV   60000
#define AD0         96
#define N_ANTENNA   224

// defines of DADA
#define N_PKT_PER_BLOCK  256
#define N_TIMESTEP_PER_BLOCK (N_TIMESTEP_PER_PKT*N_PKT_PER_BLOCK)
#define BLOCK_SIZE       (PKT_DTSZ*N_ANTENNA*N_PKT_PER_BLOCK)
#define BLOCK_PER_BUFFER 8
#define BUFFER_SIZE      (BLOCK_PER_BUFFER*BLOCK_SIZE)

// defines of socket
const unsigned window_bytes = 64*pow(1000.0, 2); ///< 64 MB in bytes
const int reuse = 1; ///< reust address 

const struct timeval tout  = {0, 0};
const uint32_t nsleep_dead = 45;  ///< Estimated dead time of sleep
const int nsecond_report   = 1;   ///< Report traffic status every second

// packet header has a 1-byte flag and a 7-bytes counter
typedef struct packet_header_t{
  uint64_t flag : 8;        // represent the antenna number
  uint64_t counter : 56;    // represent the time sample number
}packet_header_t;

void catch_int(int sig_num);

uint64_t gregorian_calendar_to_jd(int y, int m, int d);
uint64_t gregorian_calendar_to_mjd(int y, int m, int d);

#endif
