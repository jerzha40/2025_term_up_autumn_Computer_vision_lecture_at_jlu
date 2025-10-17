#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_
#include "Base.h"

void SHOW(HWND hwnd, LPBYTE image, int iWidth, int iHeight, int top,int bottom, int left, int right, unsigned long lType);
Image *READ_BMP_FROM_MEMORY(LPBYTE LP);
Image *READ_BMP_FROM_FILE(char *FileName);
BOOL CHANGE_IMAGE_TO_GRAY(LPBYTE LP, LPBYTE SP, int iWidth, int iHeight, unsigned long lType);
BOOL SAVE_BMP(LPBYTE LP, int iWidth, int iHeight, char *FileName,unsigned long lType);
#endif