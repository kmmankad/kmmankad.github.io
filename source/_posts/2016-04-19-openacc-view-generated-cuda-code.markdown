---
layout: post
title: "OpenACC: View generated CUDA code"
date: 2016-04-19 07:27:14 +0530
comments: true
categories: CUDA, OpenACC
---

This is going to be a short post on how one can view the actual compiler generated CUDA code when running OpenACC on NVIDIA hardware. It is a warmup to my upcoming post on where-data-goes-with-OpenACC's-various-data-directives.

When compiling a OpenACC accelerated program, heres what a canonical compile command would look like for the [PGI compiler](http://www.pgroup.com/resources/cuda-x86.htm)

```sh
pgcc -I. -g -fast -Minfo -acc -ta=nvidia -o hex2base64.exe hex2base64.c main.c
```
(Yes, this is from my [first OpenACC tutorial](http://kmmankad.github.io/blog/2016/04/03/openacc-analyze/))

With these set of options, the intermediate PTX or CUDA code is not visible to the user. However, if we add `keepgpu,nollvm` to the `-ta=nvidia` option, then the compiler dumps those for us to see. With only `keepgpu`, you would get only the PTX source code and binaries.

```sh
pgcc -I. -g -fast -Minfo -acc -ta=nvidia:keepgpu,nollvm -o hex2base64.exe hex2ba se64.c main.c
```

With this, ordinary OpenACC'd code like this:
```c++
unsigned int encode_block( char *input, unsigned int size, char *output){
       // ....shortened for brevity....
	#pragma acc data present(input[0:size]), present(base64_LUT[64]), copyout(output[0:4*size/3])
	#pragma acc kernels 
	#pragma acc loop private(decoded_octets, k)
	for (i=0; i<size; i=i+3){ 
		// Calculate the output array position based
		// on the input array position (loop iteration)
		k = (4*i)/3;
		
		decoded_octets[0] = input[i] >> 2;
		output[k] = base64_LUT[decoded_octets[0]];
       // ....shortened for brevity....
```

Would generate an intermediate file that looks like this:
```c++
#include "cuda_runtime.h"
#include "pgi_cuda_runtime.h"
#include "hex2base64.n001.h"
	extern "C" __global__ __launch_bounds__(32) void
encode_block_92_gpu(
		unsigned int tc1,
		signed char* __restrict__ p3/* input */,
		signed char* __restrict__ p4/* decoded_octets */,
		signed char* __restrict__ p5/* base64_LUT */,
		signed char* p6/* output */,
		unsigned int x5/* size */)
{
	int _i_1, _i_2, _i_3;
	unsigned int _ui_1;
	signed char* _p_1, *_p_2, *_p_3;
	unsigned int x6/* k */;
	unsigned int i25s;
	unsigned int i26i;
	signed char* p27/* decoded_octets */;
	i25s = 0;
_BB_8: ;
       i26i = ((((int)blockIdx.x)*(32))+((int)threadIdx.x))+((int)(i25s));
       if( ((unsigned int)(i25s)>=(unsigned int)(tc1)))  goto _BB_9;
       if( ((unsigned int)(i26i)>=(unsigned int)(tc1)))  goto _BB_9;
       p27/* decoded_octets */ = (p4/* decoded_octets */)+((long long)(((((int)blockIdx.x)*((int)blockDim.x))+((int)threadIdx.x))*(4)));
       _ui_1 = (i26i)*(3);
       // ....shortened for brevity....
```


While this is legal CUDA code, its quite cluttered. With a neato perl oneliner, we can get that a bit cleaner and easier to read:

```sh
perl -pe 's/\w\d\/\*\s(\w+)\s\*\//$1/g' cluttered_openacc_cuda_code.c
```

Code is now:

```c++
#include "cuda_runtime.h"
#include "pgi_cuda_runtime.h"
#include "hex2base64.n001.h"
extern "C" __global__ __launch_bounds__(32) void
encode_block_92_gpu(
        unsigned int tc1,
        signed char* __restrict__ input,
        signed char* __restrict__ decoded_octets,
        signed char* __restrict__ base64_LUT,
        signed char* output,
        unsigned int size)
{
    int _i_1, _i_2, _i_3;
    unsigned int _ui_1;
    signed char* _p_1, *_p_2, *_p_3;
    unsigned int k;
    unsigned int i25s;
    unsigned int i26i;
    signed char* decoded_octets;
    i25s = 0;
_BB_8: ;
       i26i = ((((int)blockIdx.x)*(32))+((int)threadIdx.x))+((int)(i25s));
       if( ((unsigned int)(i25s)>=(unsigned int)(tc1)))  goto _BB_9;
       if( ((unsigned int)(i26i)>=(unsigned int)(tc1)))  goto _BB_9;
       decoded_octets = (decoded_octets)+((long long)(((((int)blockIdx.x)*((int)blockDim.x))+((int)threadIdx.x))*(4)));
       _ui_1 = (i26i)*(3);
       k = ((_ui_1)*(4))/(3);
       _p_1 = (signed char*)((input)+((long long)(_ui_1)));
       *( signed char*)(decoded_octets) = (int)(*( signed char*)(( signed char*)_p_1))>>(2);
       *( signed char*)((output)+((long long)(k))) = (*( signed char*)((base64_LUT)+((long long)((*( signed char*)(decoded_octets))))));
       *( signed char*)((decoded_octets)+(1LL)) = ((*( signed char*)(( signed char*)_p_1))&(3))<<(4);
       if( ((unsigned int)(((int)(_ui_1))+(1))>=(unsigned int)(size)))  goto _BB_22;
       _ui_1 = (i26i)*(3);
       // ....shortened for brevity....
```
Much better.

Thats it! Hope this (and my next OpenACC post `TODO:insert link` ) helps you guys.

PS: I wonder if a screencast is a better medium for my (b)log.
