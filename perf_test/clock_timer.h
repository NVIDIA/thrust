#pragma once

#include <ctime>

struct clock_timer
{
  std::clock_t start;

  clock_timer()
    : start(std::clock())
  {}

  void restart()
  {
    start = std::clock();
  }

  double elapsed_seconds()
  {
    return double(std::clock() - start) / CLOCKS_PER_SEC;
  }
};

