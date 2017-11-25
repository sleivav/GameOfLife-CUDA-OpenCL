//
// Created by checho on 9/26/17.
//

#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <afxres.h>
#include "serial_gol.h"

unsigned char* m_data;
unsigned char* m_resultData;

unsigned char countAliveCells(size_t x0, size_t x1, size_t x2,
                              size_t y0, size_t y1, size_t y2) {
  return m_data[x0 + y0] + m_data[x1 + y0] + m_data[x2 + y0] +
         m_data[x0 + y1] + m_data[x1 + y1] + m_data[x2 + y1] +
         m_data[x0 + y2] + m_data[x1 + y2] + m_data[x2 + y2];
}

void compute(size_t m_worldWidth, size_t m_worldHeight) {
  for(size_t y = 0; y < m_worldHeight; ++y) {
    size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
    size_t y1 = y * m_worldHeight;
    size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

    for (size_t x = 0; x < m_worldWidth; ++x) {
      size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
      size_t x2 = (x + 1) % m_worldWidth;

      unsigned char aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
      m_resultData[y1 + x] = aliveCells == 3 ||
                             (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
    }
  }
  unsigned char* a = m_data;
  m_data = m_resultData;
  m_resultData = a;
}

void compute_highLife(size_t m_worldWidth, size_t m_worldHeight) {
  for(size_t y = 0; y < m_worldHeight; ++y) {
    size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
    size_t y1 = y * m_worldHeight;
    size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

    for (size_t x = 0; x < m_worldWidth; ++x) {
      size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
      size_t x2 = (x + 1) % m_worldWidth;

      unsigned char aliveCells = countAliveCells(x0, x, x2, y0, y1, y2);
      m_resultData[y1 + x] = (!m_data[x+y] && (aliveCells == 3 || aliveCells == 6) ||
                             (m_data[x+y] && (aliveCells == 2 || aliveCells == 3))) ? 1 : 0;
    }
  }
  unsigned char* a = m_data;
  m_data = m_resultData;
  m_resultData = a;
}

int read_data(int height, int width) {
  std::ifstream input(height + "x" + width);
  int z = 0;
  while(!input.eof() && z < height * width) {
    char c;
    input.get(c);
    if(c == 48) {
      m_data[z] = 0;
    } else {
      m_data[z] = 1;
    }
      z++;
  }
}

void experiment(int width, int height, int iterations) {
  m_data = static_cast<unsigned char *>(malloc(height * width * sizeof(unsigned char*)));
  m_resultData = static_cast<unsigned char *>(malloc(height * width * sizeof(unsigned char*)));
  for(int i = 0; i < height * width; i++) {
    m_data[i] = m_resultData[i] = 0;
  }
  read_data(height, width);
  long int t0 = GetTickCount();
  for(int i = 0; i < iterations; i++) {
    compute_highLife(width, height);
  }
  long int t1 = GetTickCount();
  long int myint = t1 - t0;
  std::ofstream ofs;
  ofs.open("salida.txt", std::ios_base::app);
  ofs << width << ", " << myint << '\n';
  ofs.close();
}

int main(int argv, char** args) {
  for(int i = 0; i < 12; i++) {
    int exp = static_cast<int>(exp2(i));
    int height = exp;
    int width = exp;
    int iterations = 1000;
    for(int expr = 0; expr < 10; expr++) {
      experiment(width, height, iterations);
    }
  }
  return 0;
}