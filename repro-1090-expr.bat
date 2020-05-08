C:\PROGRA~1\NVIDIA~2\CUDA\v10.2\bin\nvcc.exe ^
  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA ^
  -I.\ -Idependencies\cub ^
  -Xcompiler="-MD" ^
  -DNDEBUG ^
  -c repro-1090.cu
