#include <chrono>
#include <cstdlib>
#include "timing.h"


long getTimeMilliseconds()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


