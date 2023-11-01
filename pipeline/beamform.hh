/* beamform.hh

    contains functions and defines related to beamform

*/

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <unistd.h>
#include "../udp/udp.h"

#ifndef DEBUG
	#include <algorithm>
	#include <stdlib.h>
	#include <math.h>
	#include <string.h>
	#include <netdb.h>
	#include <sys/socket.h>
	#include <sys/types.h>
	#include <netinet/in.h>
	#include <time.h>

	#include "dada_client.h"
	#include "dada_def.h"
	#include "dada_hdu.h"
	#include "multilog.h"
	#include "ipcio.h"
	#include "ipcbuf.h"
	#include "dada_affinity.h"
	#include "ascii_header.h"
#endif

// defines of beamform
#define N_BEAM 180
#define N_FREQ N_CHAN
#define COEFFICIENT_SIZE (N_FREQ*N_ANTENNA*N_BEAM)

// defines of CUDA
#define BF_SAMPSZ 2
#define INPUT_BLOCK_SIZE (BLOCK_SIZE)
#define BF_BLOCK_SIZE (N_TIMESTEP_PER_BLOCK*N_BEAM*N_FREQ*BF_SAMPSZ)

// defines of integration
#define N_AVERAGE 8
#define INTE_SAMPSZ 4
#define N_TIMESTEP_PER_INIT_BLOCK (N_TIMESTEP_PER_BLOCK/N_AVERAGE)
#define INTE_BLOCK_SIZE (N_TIMESTEP_PER_BLOCK*N_BEAM*N_FREQ*INTE_SAMPSZ/N_AVERAGE)


