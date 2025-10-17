#include "stdafx.h"
#include "Base.h"
#include "Image Processing.h"
#include "math.h"
#include "stdio.h"
#include "string.h"
#include "TCHAR.H" 
#include "LIMITS.H"
#include <io.h> 

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
void SHOW(HWND hwnd, LPBYTE image, int iWidth, int iHeight, int top,int bottom, int left, int right, unsigned long lType)
{
	BYTE BitInfoTemp[1280];
	BITMAPINFO *BitInfo=(BITMAPINFO *)BitInfoTemp;
	BitInfo->bmiHeader.biSize =  sizeof(BITMAPINFOHEADER);
	BitInfo->bmiHeader.biWidth = iWidth;
	BitInfo->bmiHeader.biHeight = iHeight;
	BitInfo->bmiHeader.biPlanes= 1; 
	BitInfo->bmiHeader.biCompression = BI_RGB; 
	BitInfo->bmiHeader.biSizeImage = 0; 
	BitInfo->bmiHeader.biClrUsed = 0; 
	BitInfo->bmiHeader.biClrImportant = 0; 	
	HDC hdc = GetDC(hwnd);
    long lStillWidth = right-left;
    long lStillHeight = bottom - top;    
	for(int i=0; i< 256;i++){*((DWORD *)BitInfo->bmiColors+i)=i | (i<<8) | (i<<16) | (0<<24);}   

	if(lType==RGB8){BitInfo->bmiHeader.biBitCount = 8;}
	else if(lType==RGB24){BitInfo->bmiHeader.biBitCount = 24;}
	else if(lType==RGB32){BitInfo->bmiHeader.biBitCount = 32; }
	if((iWidth*BitInfo->bmiHeader.biBitCount/8)%4==0)
	{
		StretchDIBits(hdc, left, top, lStillWidth, lStillHeight, 0, 0, iWidth, iHeight, image, BitInfo, DIB_RGB_COLORS, SRCCOPY);
	}
	else
	{
		int RWidth=(((iWidth*BitInfo->bmiHeader.biBitCount/8)+3)/4)*4;
		LPBYTE ima=(LPBYTE)GlobalAlloc(GPTR,sizeof(BYTE)*RWidth*iHeight);
		for(int i=0; i<iHeight; i++){memcpy(ima+i*RWidth,image+i*iWidth*BitInfo->bmiHeader.biBitCount/8,iWidth*BitInfo->bmiHeader.biBitCount/8);}
		StretchDIBits(hdc, left, top, lStillWidth, lStillHeight, 0, 0, iWidth, iHeight, ima, BitInfo, DIB_RGB_COLORS, SRCCOPY);
		GlobalFree(ima);
	}
	ReleaseDC(hwnd, hdc); 	
}

Image *READ_BMP_FROM_FILE(char *FileName)
{	
	FILE *f=NULL;
	if((fopen_s(&f,FileName,"rb"))!=0){return NULL;}

	BITMAPINFO fileinfo; BITMAPFILEHEADER fileheader; BYTE ColorMap[1024];
	fread(&fileheader,1,sizeof(BITMAPFILEHEADER),f);
	fread(&fileinfo,1,sizeof(BITMAPINFO),f);
	fseek(f,sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER),SEEK_SET);
	fread(ColorMap,sizeof(RGBQUAD),(fileheader.bfOffBits-54)/4,f);	
	int color=fileinfo.bmiHeader.biBitCount;
	Image *image=(Image *)GlobalAlloc(GPTR, sizeof(Image));
	int iGray=1; image->ImageType=RGB8;
	for(int i=0; i<(int)((fileheader.bfOffBits-54)/4); i++)
	{
		if(ColorMap[4*i+0]!=ColorMap[4*i+1] || ColorMap[4*i+0]!=ColorMap[4*i+2]){iGray=3;image->ImageType=RGB24;}
	}
	int realbytes=((__max(color/8,1)*iGray*fileinfo.bmiHeader.biWidth+3)/4)*4;
	image->iWidth=fileinfo.bmiHeader.biWidth; image->iHeight=fileinfo.bmiHeader.biHeight;
	image->ImageData=(LPBYTE)GlobalAlloc(GPTR,sizeof(BYTE)*realbytes*image->iHeight);
	if(color<=8)
	{
		BYTE b[8]; int BitNumber=8/color;
		int linebytes=((image->iWidth+BitNumber-1)/BitNumber+3)/4*4;	
		LPBYTE ima1=(BYTE *)GlobalAlloc(GPTR, sizeof(BYTE)*linebytes*image->iHeight);
		fread(ima1,sizeof(BYTE),linebytes*image->iHeight,f);
		for(int i=0; i<BitNumber; i++)
		{
			b[BitNumber-1-i]=0;
			for(int j=0; j<color; j++){b[BitNumber-1-i]+=(1<<(j+i*color));}
		}

		for(int i=0; i<image->iHeight; i++)
		{
			for(int j=0; j<linebytes; j++)
			{
				for(int i1=0; i1<BitNumber; i1++)
				{
					if(BitNumber*j+i1<image->iWidth)
					{
						int iTemp = ((*(ima1+i*linebytes+j)&(b[i1]))>>(BitNumber-1-i1)*color);
						int k=(i*realbytes+iGray*(BitNumber*j+i1));
						for(int j1=0; j1<iGray; j1++)
						{
							*(image->ImageData+k+j1)=*(ColorMap+4*iTemp+j1);
						}
					}
				}
			}
		}
		GlobalFree(ima1);
	}						
	else if(color==24)
	{
		image->ImageType=RGB24;
		fread(image->ImageData,sizeof(BYTE),realbytes*image->iHeight,f);
	}
	else if(color==32)
	{
		image->ImageType=RGB32;
		fread(image->ImageData,sizeof(BYTE),realbytes*image->iHeight,f);
	}
	if(f!=NULL){fclose(f); f=NULL;}
	return image; 
}	

BOOL SAVE_BMP(LPBYTE LP, int iWidth, int iHeight, char *FileName,unsigned long lType)
{
	if(LP==NULL || iWidth<=0 || iHeight<=0){return FALSE;}

	RGBQUAD Palate[256];
	int iColorByte=0;int iPalateCount=0;
	if(lType==RGB32){iColorByte=4;}
	else if(lType==RGB24){iColorByte=3;}
	else if(lType==RGB8)
	{
		iColorByte=1;iPalateCount=1;
		for(int i=0; i<256; i++)
		{
			Palate[i].rgbRed=i;
			Palate[i].rgbGreen=i;
			Palate[i].rgbBlue=i;
			Palate[i].rgbReserved=0;
		}
	}

	BITMAPFILEHEADER hFile;
	hFile.bfType=19778;
	hFile.bfReserved1=0;
	hFile.bfReserved2=0;
	hFile.bfSize=((iColorByte*iWidth+3)/4)*4*iHeight+54+iPalateCount*4*256;
	hFile.bfOffBits=54+iPalateCount*4*256;

	BITMAPINFO BINFO;
	BINFO.bmiHeader.biSize =  sizeof(BITMAPINFOHEADER);
	BINFO.bmiHeader.biWidth = iWidth;
	BINFO.bmiHeader.biHeight = iHeight;
	BINFO.bmiHeader.biPlanes= iPalateCount; 
	BINFO.bmiHeader.biCompression = BI_RGB; 
	BINFO.bmiHeader.biBitCount = iColorByte*8; 
	BINFO.bmiHeader.biSizeImage = ((iColorByte*iWidth+3)/4)*4*iHeight;
	BINFO.bmiHeader.biClrUsed = 0; 
	BINFO.bmiHeader.biClrImportant = 0;
	
	LPBYTE img = (LPBYTE)GlobalAlloc(GPTR,BINFO.bmiHeader.biSizeImage);

	int RealByteCount = ((iColorByte*iWidth+3)/4)*4;

	for(int i=0; i<iHeight; i++)
	{
		for(int j=0; j<iWidth; j++)
		{
			for(int k=0; k<iColorByte; k++)
			{
				img[i*RealByteCount+iColorByte*j+k] = *((LPBYTE)LP+iColorByte*i*iWidth+iColorByte*j+k);
			}
		}
	}

	FILE *f=NULL;
	if(fopen_s(&f,FileName,"wb")!=0)
	{
		if(f!=NULL){fclose(f); f=NULL;} return FALSE;
	}
	else
	{
		fwrite(&hFile,sizeof(BITMAPFILEHEADER),1,f);
		fwrite(&BINFO,sizeof(BITMAPINFO),1,f);
		fwrite(Palate,1,sizeof(RGBQUAD)*256*iPalateCount,f);
		fseek(f,hFile.bfOffBits,SEEK_SET);
		fwrite(img,1,BINFO.bmiHeader.biSizeImage,f);
	}
	if(f!=NULL){fclose(f); f=NULL;}
	GlobalFree(img);
	return TRUE;
}

BOOL CHANGE_IMAGE_TO_GRAY(LPBYTE LP, LPBYTE SP, int iWidth, int iHeight, unsigned long lType)
{
	if(LP==NULL || SP==NULL || iWidth<=0 || iHeight<=0){return FALSE;}
	if(lType==RGB24)
	{
		for(int i=0; i<iWidth*iHeight; i++)
		{
			SP[i]=(LP[3*i+0]+LP[3*i+1]+LP[3*i+2])/3;
		}
	}
	else if(lType==RGB32)
	{
		for(int i=0; i<iWidth*iHeight; i++)
		{
			SP[i] =(*(LP+4*i+0) + *(LP+4*i+1) + *(LP+4*i+2))/3;
		}
	}
	else if(lType==RGB555)
	{
		LPBYTE PS=SP;
		for(int i=0; i<iWidth*iHeight; i++)
		{
			int iTemp = *((unsigned short *)LP+i); 
			*(PS++) = (BYTE)((((iTemp&31)<<3) + ((iTemp&992)>>2) + ((iTemp&31744)>>7))/3);
		}
	}
	else if(lType==I420 || lType==YVU9 || lType==YV12)
	{
		for(int i=0; i<iHeight; i++)
		{
			memcpy(SP+i*iWidth,LP+(iHeight-1-i)*iWidth,sizeof(BYTE)*iWidth);
		}
	}
	else if(lType==YUY2)
	{
		for(int i=0; i<iHeight; i++)
		{
			unsigned short *PL=(unsigned short *)LP+i*iWidth; LPBYTE PS=SP+(iHeight-1-i)*iWidth;
			for(int j=0; j<iWidth; j++)
			{
				*(PS++) = (*(PL++))&255;
			}
		}
	}
	else if(lType==UYVY)
	{
		for(int i=0; i<iHeight; i++)
		{
			unsigned short *PL=(unsigned short *)LP+i*iWidth; LPBYTE PS=SP+(iHeight-1-i)*iWidth;
			for(int j=0; j<iWidth; j++)
			{
				*(PS++) = (*(PL++))>>8;
			}
		}
	}
	return TRUE;
}
