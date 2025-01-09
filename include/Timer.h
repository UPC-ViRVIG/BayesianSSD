#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
	void start()
	{
		lastTime = std::chrono::steady_clock::now();
	}

	float getElapsedSeconds()
	{
		return float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - lastTime).count())/1000000.0f;
	}

	float getElapsedMicroseconds()
	{
		return float(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - lastTime).count())/1000.0f;
	}

	int getElapsedMilliseconds()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - lastTime).count();
	}
private:
	std::chrono::time_point<std::chrono::steady_clock> lastTime;
};

#endif