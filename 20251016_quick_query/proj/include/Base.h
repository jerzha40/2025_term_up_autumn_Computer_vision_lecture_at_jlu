#ifndef _BASE_H_
#define _BASE_H_
#include "math.h"
#include "io.h"
#include "stdlib.h"
#ifndef I420
#define I420 808596553
#endif
#ifndef RGB8
#define RGB8 3828804474
#endif
#ifndef RGB555
#define RGB555 3828804476
#endif
#ifndef RGB24
#define RGB24 3828804477
#endif
#ifndef RGB32
#define RGB32 3828804478
#endif
#ifndef UYVY
#define UYVY 1498831189
#endif
#ifndef YUY2
#define YUY2 844715353
#endif
#ifndef YVU9
#define YVU9 961893977
#endif
#ifndef YV12
#define YV12 842094169
#endif
#ifndef INT_TYPE
#define INT_TYPE 4
#endif
#ifndef BYTE_TYPE
#define BYTE_TYPE 1
#endif
#include "windows.h"
//typedef double REAL;
//typedef float REAL;

//#define precision
#ifdef precision
//#define REAL double
typedef double REAL;
#else
//#define REAL float
typedef float REAL;
#endif
#pragma warning (disable: 4244) 
struct Image
{
	int iWidth;
	int iHeight;
	LPBYTE ImageData;
	unsigned long ImageType;
};
struct LOC
{
	int x;
	int y;
};
struct dImage
{
	int iWidth;
	int iHeight;
	double *Data;
};
struct ThreadParam
{
	HANDLE DistanceThreadEvent;
    HANDLE DisplayThreadEvent;
	int Stop;
	HANDLE hDis;
	HANDLE hCam;
	HWND hWnd;
};

struct WFilter
{
	double *Lo_D;
	double *Hi_D;
	int LoL;
	int HiL;
};
struct WDecCoeff
{
	double *LL;
	double *HL;
	double *LH;
	double *HH;
	int iWidth;
	int iHeight;
};

struct Complex
{
	double R;
	double I;
};
struct ComMatrix
{
	int iWidth;
	int iHeight;
	Complex *CC;
};


struct VecEng
{
	double EngAll;
	double EngMin;
	double EngAve;
	double EngVar;
	double bilv;
	int iCountAll;
	int iCount;
};

void GlobalFreeAlign(void *p);

void *GlobalAllocAlign(unsigned int size_t, unsigned int align);
void *ReturnMemoryAlign(void *p, unsigned int align);
#endif
