#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/udp.h"


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
