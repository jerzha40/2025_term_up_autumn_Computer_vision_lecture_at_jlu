## Build the extension

Run this in your project folder (with cl from msvc):

```bash
.\venv\Lib\site-packages\nvidia\cu13\bin\nvcc.exe fastcuda.cu --shared -o fastgpu.dll
