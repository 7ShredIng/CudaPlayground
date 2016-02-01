# CudaPlayground
CUDA BY EXAMPLE 
An Introduction to General-Purpose GPU Programming
Jason Sanders, Edward Kandrot
ISBN-13: 978-0-13-138768-3

## Development Environment
- Visual Studio 2010
- CUDA 7.5
http://http.developer.nvidia.com/NsightVisualStudio/2.2/Documentation/UserGuide/HTML/Content/Timeout_Detection_Recovery.htm

## Notes
- Many errors in code samples. watch out for missing includes, undefined variables.. https://developer.nvidia.com/cuda-example-errata-page
- Simple time measuring does not work. Use cudaEvents
- Problem in chapter 7.3.4 -- tex1Dfetch() is not available in cuda v7+
- Problem in chapter 7.3.4 -- 2D texture does not compile 
