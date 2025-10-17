#include "stdafx.h"
#include "Base.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
void *GlobalAllocAlign(unsigned int size_t,unsigned int align)
{
	void *p0, *p;
	p0 = GlobalAlloc(GPTR, (size_t + align));if (p0 == NULL) { return NULL; }
	p = (void *)(((unsigned int)p0 + align) & (~((unsigned int)(align - 1))));
	*((void **)p - 1) = p0;
	return p;
	SIZE_T c;
}
void GlobalFreeAlign(void *p)
{
     if (p) GlobalFree(*((void **) p - 1));
}

void *ReturnMemoryAlign(void *p, unsigned int align)
{
	void *p0 = (void *)(((unsigned int)p + align) & (~((unsigned int)(align - 1))));
	return p0;
}