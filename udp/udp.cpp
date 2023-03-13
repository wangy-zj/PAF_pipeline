#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "udp.h"

int seconds2dhms(uint64_t seconds, char dhms[STR_BUFLEN]){

  uint64_t n = seconds;
  
  int days = n / (24 * 3600);
  
  n = n % (24 * 3600);
  int hours = n / 3600;
  
  n %= 3600;
  int minutes = n / 60 ;
  
  n %= 60;

  sprintf(dhms, "%d:%02d:%02d:%02d", days, hours, minutes, (int)n);
  
  return EXIT_SUCCESS;
}

uint64_t gregorian_calendar_to_jd(int y, int m, int d){
  y+=8000;
  if(m<3) { y--; m+=12; }
  return (y*365) +(y/4) -(y/100) +(y/400) -1200820
    +(m*153+3)/5-92
    +d-1;
}

uint64_t gregorian_calendar_to_mjd(int y, int m, int d){
  return 367 * y
    - 7 * (y + (m + 9) / 12) / 4
    - 3 * ((y + (m - 9) / 7) / 100 + 1) / 4
    + 275 * m / 9
    + d + 1721029 - 2400001;
}

/* first, here is the signal handler */
void catch_int(int sig_num){
  /* re-set the signal handler again to catch_int, for next time */
  signal(SIGINT, catch_int);
  /* and print the message */
  fprintf(stdout, "UDP2DB_INFO: killed by ctrl+c\n");
  fflush(stdout);
  
  exit(EXIT_SUCCESS);
}
