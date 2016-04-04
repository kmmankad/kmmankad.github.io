---
layout: post
title: "OpenACC: Analyze, Express, Tweak! - Part 1"
date: 2016-04-03 11:22:49 +0530
comments: true
categories: openacc, parallel, cuda
---
## Whats OpenACC?
From [http://developer.nvidia.com/openacc](http://developer.nvidia.com/openacc):
> OpenACC is a directive-based programming model designed to provide a simple yet powerful approach to accelerators without significant programming effort.

What that is means is, you can pickup existing code written for an x86 CPU, and add some compiler `#pragmas`, compile with an OpenACC capable compiler - and voila! You get accelerated binaries for a range of hardware accelerators - Nvidia GPUs, AMD GPUs and even Intel multi-core CPUs. Thats really the USP of OpenACC - a single copy of the source code will deliver performance portability across this range of hardware platforms. 
So, to be successful with OpenACC all you need are strong concepts in parallel programming, some know-how about OpenACC syntax and you’re good to go! You dont need to really know too many lower level hardware details with OpenACC, as opposed to, maybe CUDA C. However, this is a double edged sword - I will revisit this later in this post. Remember, OpenACC is about expressing parallelism - its not GPU programming.

There are some really good tutorials on OpenACC itself available online:  
1. [Jeff Larkin's post on the Parallel Forall blog](https://devblogs.nvidia.com/parallelforall/getting-started-openacc/)  
2. Jeff Larkin's sessions from GTC 2013 - recordings on Youtube here : [Part1](https://www.youtube.com/watch?v=0e5TiwZd_wE) [Part2](https://www.youtube.com/watch?v=YueszvniRUE)

The recommended approach for parallelism anywhere is to:  
1. Try and use existing parallel optimized libraries like cuBLAS, cuDNN etc. if they exist for your application.  
2. If you dont get those, try OpenACC on your code. That should get you about 80% of the maximum available performance.  
_Ofcourse, that is a very rough number and is subject to, you guessed it, your code and the GPU hardware you're running._
3. Roll your own CUDA kernels. This is definitely the most involved of the 3 options, but it will allow you to squeeze
every last drop of that good perf juice from your software and hardware.  

OpenACC tutorials online often use the Jacobi Iteration/sAXPY example to demonstrate OpenACC, but all that those examples teach us are syntax constructs. However, if you use OpenACC in the real world, you’ll know it's all about how you analyze your source code, understand its scope for parallelism and finally express that formally via OpenACC syntax. What this post is really about is about the analysis of a simple program, which is hopefully a little less trivial than the Jacobi type examples all over the net. Also, this is not one of those _100X in 2 hours_ posts, because that does not always happen.

## Setup
First off, some logistics about tool installation and setup.

* We will be using the PGI Compiler today, which you can get from the [PGroup's site](http://www.pgroup.com/support/download_pgi2016.php?view=current)  
* You can also download the [OpenACC toolkit from NVIDIA](https://developer.nvidia.com/openacc-toolkit)

If you have everything correctly setup, try `pgcc --version` as shown below
```sh
PGI Workstation 15.10 (64)
PGI$ pgcc --version

pgcc 15.10-0 64-bit target on x86-64 Windows -tp haswell
The Portland Group - PGI Compilers and Tools
Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
```

## The target

Now, onto our target today - a subroutine that converts a hexadecimal string to base64. I picked this up from the [matasano cryptography challenges](http://cryptopals.com/) I'm attempting on the side and decided it'd be a good example for this tutorial.

Heres a brief overview of the algorithm itself:  
1. Take 3 bytes of input hex data at a time,  
2. Do some bitwise concatenation (shift and OR) and get indexes of 4 base64 characters that these 3 bytes are encoded into  
3. Lookup the actual base64 characters using these indices.  
..and heres a quick diagram to explain that:

![Diagram showing Hex to Base64 conversion](https://github.com/kmmankad/kmmankad.github.io/raw/master/images/openacc/ASCII_to_b64.PNG "Figure 1: Hex to Base64")

Lets look at what we'll start here:
```c++
unsigned int encode_block( char* input, unsigned int size, char* output){

	char decoded_octets[4];
	unsigned int i,j=0;

	for (i=0; i<size; i=i+3){
		decoded_octets[0] = input[i] >> 2;
		output[j++] = base64_LUT[decoded_octets[0]];
		decoded_octets[1] = (input[i] & 0x03) << 4;
		if (i+1 < size){
			decoded_octets[1] |= ((input[i+1] & 0xF0) >> 4);
			output[j++] = base64_LUT[decoded_octets[1]];

			decoded_octets[2] = ((input[i+1] & 0x0F) << 2);
			// Check if we have an (i+2)th input element
			if (i+2 < size){
				decoded_octets[2] |= ((input[i+2] & 0xC0) >> 6);
				output[j++] = base64_LUT[decoded_octets[2]];

				decoded_octets[3] = input[i+2] & 0x3F;
				output[j++] = base64_LUT[decoded_octets[3]];
			} else {
				output[j++] = base64_LUT[decoded_octets[2]];
				output[j++] = '=';
			}
		}else{
			output[j++] = base64_LUT[decoded_octets[1]];
			output[j++] = '='; 
			output[j++] = '='; 
		}
	
	}
	// Return the code length
	return (j);
}
```
Usually, you'd just throw some `#pragma acc`s at the around loops in the problem and let the compiler guide you. But, the idea of this tutorial is to help develop some analysis skills, so we'll look through the program first.
     
Now, the function basically takes in a character array of a fixed size, and generates an output array also of a known size (4 x input_size/3). The sizes are important to know, because the compiler needs to know how many bytes to transfer over the CPU<->GPU link. (Side note - if you dont specify those sizes clearly, the compiler will throw - `Accelerator restriction: size of the GPU copy of output is unknown`) We need to copy over the input array from the CPU to the GPU - or, Host and Device respectively in CUDA terminology. Sometimes, OpenACC documentation refers to the CPU as 'Self' and GPU as 'Device'. And when it is done processing, we must copy the output array back to the CPU. And, the `base64_LUT` is a common array used by all threads. So, that too will need to be on the GPU. So thats the basic data movement defined right there that you should aim to isolate first. _"Whats my input? Whats my output?"_

That `for (i=0..` loop can be parallelized to operate on chunks of the input in parallel. But, hang on. The next thing I'd like to draw your attention to is - **data dependence between loop iterations**. What? Where? Well, if you take a closer look at how we're updating the output array, you'll quickly realize that `j++` implies that you rely on the previous value of `j` - i.e. the previous iteration. Why is that a problem? Well, for us to run the conversion in parallel, each thread must know its input index and output index without communicating with other threads. Because, if it needed to, that'll defeat the purpose of parallelization - thats as good as sequential CPU code. So, thats the first thing that needs fixing.  Dont worry, the compiler will warn you about this, but it helps to develop what I like to call _dependence vision_ - the ability to "see" the data dependence. That'll help you with complex code bases where things are not so obvious. Moral of the story: _Try to code in a way that keeps the array indices independent of the previous loop iteration, and hopefully dependent on only the current iteration_

Going further, the `decoded_octets` variable is used as a scratch variable to hold 4 values that we eventually push to the output array. This means, each iteration of the loop uses it for itself - something we need to tell the compiler. This is a private variable for each iteration, or each parallel thread.

Because we're dealing with pointers to access data arrays, there is an additional complication - but I'll get to that later.

Armed with this non-zero knowledge of not-so-hidden parallelism in the program, we will now use OpenACC directives to express these ideas of parallelism and data movement.

```c++
unsigned int encode_block( char *input, unsigned int size, char *output){
	// Variables for timekeeping
	timestruct t1,t2;
	long long time_elapsed;
	
	char decoded_octets[4];
	printf ("hex2base64::encode_block: Input Len: %d\n",size);
	
	// i variable will track the input array position
	// k variable will track the output array position
	unsigned int i, k;
	
	// Mark the start time
	gettime( &t1 );
	
	#pragma acc data present(input[0:size]), present(base64_LUT[64]), copyout(output[0:4*size/3])
	#pragma acc kernels 
	#pragma acc loop private(decoded_octets, k)
	for (i=0; i<size; i=i+3){ 
		// Calculate the output array position based
		// on the input array position (loop iteration)
		k = (4*i)/3;
		
		decoded_octets[0] = input[i] >> 2;
		output[k] = base64_LUT[decoded_octets[0]];
		
		decoded_octets[1] = (input[i] & 0x03) << 4;
		
		if (i+1 < size){
			decoded_octets[1] |= ((input[i+1] & 0xF0) >> 4);
			output[k+1] = base64_LUT[decoded_octets[1]];
			decoded_octets[2] = ((input[i+1] & 0x0F) << 2);	
			if (i+2 < size){
				decoded_octets[2] |= ((input[i+2] & 0xC0) >> 6);
				output[k+2] = base64_LUT[decoded_octets[2]];
				decoded_octets[3] = input[i+2] & 0x3F;
				output[k+3] = base64_LUT[decoded_octets[3]];
			} else {
				output[k+2] = base64_LUT[decoded_octets[2]];
				output[k+3] = '=';
			}
		}else{
			output[k+1] = base64_LUT[decoded_octets[1]];
			output[k+2] = '='; 
			output[k+3] = '='; 
		}
	
	}
	// Mark the end time
	gettime(&t2);
	
	// Compute time elapsed
	time_elapsed = usec(t1,t2);
	
	// Display that to the user
	printf("encode_block: %03ld microseconds\n", time_elapsed );
	
	// Return the code length
	return ((4*size)/3);
}
```
Lets look at the `#pragma`s required to express this parallelism. I've also added some crude instrumentation to measure the elapsed time for the function as a whole.

1. `#pragma acc kernels`
This tells the compiler - "Hey, I think this section of code can be parallelized. Go try and do that for me." Remember, pragmas are for the immediate next code block. So, this one applies to the `for (i=0..` loop. As you will soon learn, adding this macro does not mean that parallel code will be generated. The compiler will try and might fail - so watch the compile output closely for such cases.

2. `#pragma acc data present(input[0:size]), present(base64_LUT[64]), copyout(output[0:4*size/3])`
Here, we're using the `present` clause to tell the compiler about data arrays that we will copy into GPU memory beforehand. Specifically, I have done that just before the function call to `encode_block` using the `copyin` clause. The `copyout` clause as the name suggests directs the compiler to copy out an array `output[0:4*size/3]` from the GPU to the CPU _at the end of the parallel thread's execution_.

3. `#pragma acc loop private(decoded_octets, k)`
This one tells the compiler - "look, the variables `decoded_octets` and `k` are _private_ to each iteration of the loop, or each parallel thread. So create private copies of those variables and dont think they depend between loop iterations.

With these changes in place, try giving it a whirl - run `make`. This is what you can expect:
```sh
PGI$ make
rm -rf hex2base64.obj main.obj hex2base64.exe hex2base64.dwf hex2base64.pdb
pgcc -I. -g -fast -Minfo -acc -ta=nvidia  -o hex2base64.exe hex2base64.c main.c
hex2base64.c:
encode:
 _<boring part excluded for brevity>_
     89, Generating present(input[:size],base64_LUT[:64])
         Generating copyout(output[:?])
     92, Complex loop carried dependence of input->,output-> prevents parallelization
         Loop carried dependence due to exposed use of output[:?] prevents parallelization
         Loop carried dependence of output-> prevents parallelization
         Loop carried backward dependence of output-> prevents vectorization
         Loop carried dependence of input-> prevents parallelization
         Loop carried backward dependence of input-> prevents vectorization
         Accelerator scalar kernel generated
```
Those slew of messages based on line 92 - thats our `for (i=0..` loop. Lets look at what these messages mean:
1. `Loop carried dependence due to exposed use of output[:?] prevents parallelization`
What do you mean exposed? Enter: the **restrict keyword.** By default, the compiler will assume that the underlying data object that a pointer points to can be manipulated by other pointers from other threads too. Super paranoid (as it should be!). So, this is perceived as 'data dependence' and the whole story goes south. So, as a programmer we must give the compiler the assurance that only the specified pointer variable (or expressions using it) will be used to access that underlying data. So, in our case - 
```c++
unsigned int encode_block( char *restrict input, unsigned int size, char *restrict output){
```
A compile will this change will see most of the issues above resolved. But the compiler still thinks there is some lingering data dependence. But, our analysis shows its all good and thread-safe. Lets reassure the compiler about the same by adding the `indepdent` clause to the `#pragma acc loop` line.

```c++
	#pragma acc loop private(decoded_octets, k) indepdent
```

The compiler will successfully generate a parallel _kernel_ (CUDA speak for GPU function). Heres what that'll look like:
```
PGI$ make
rm -rf hex2base64.obj main.obj hex2base64.exe hex2base64.dwf hex2base64.pdb
pgcc -I. -g -fast -Minfo -acc -ta=nvidia  -o hex2base64.exe hex2base64.c main.c
hex2base64.c:
encode:
 _<boring part excluded for brevity>_
     89, Generating present(input[:size],base64_LUT[:64])
         Generating copyout(output[:?])
92, Loop is parallelizable
         Accelerator kernel generated
         Generating Tesla code
         92, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
```
Finally! The line `Generating Tesla Code` simply implies that it will generate parallel code for NVIDIA hardware. Doesnt mean that my 760m GPU is a tesla class card =D. The part about 'gang' and 'vector(128)' is to do with the CUDA programming model.

Basically in CUDA, we have threads. And a collection of threads forms a thread-block. A collection of thread-blocks forms a grid. And you can express the number of threads, blocks and grids as 3 dimensional co-ordinates. Pretty handy for intuition when working with images and such.

Heres how that maps to OpenACC's hardware agnostic hierarchy:  

CUDA | OpenACC
------------ | -------------
Set of blocks (blockIdx.x) | Gang
Set of blocks (blockIdx.y)| Worker
Set of threads | Vector

So, it has produced 1 gang of 128 threads (didnt create an additional notion of workers here). Thats a default value, so you can use pragma's to fix that to a more realistic value for our problem size. Say, 32?
```c++
	#pragma acc loop private(decoded_octets, k) indepdent device_type(nvidia) vector(32)
```
One should always tweak the `vector()` and `gang()` constructs for optimum device utilization. Value for cores? (like Value for Money..). Most modern GPUs can support thousands of threads, but generaing extra empty threads will eat into performance because they will also be scheduled just the same as active threads and will consume slots that could have been used for some real active work on the GPU.

Note the `device_type(nvidia)` clause which means that this `vector(32)` will be applied only for NVIDIA devices. And with [OpenACC-2.0](https://devblogs.nvidia.com/parallelforall/7-powerful-new-features-openacc-2-0/), you can have different configurations of these for different devices - giving you control without sacrificing performance portability:
```c++
	#pragma acc loop private(decoded_octets, k) indepdent device_type(nvidia) vector(32) \
	device_type(radeon) vector_length(256) \
	vector_length(16)
```
So, its 32 for NVIDIA cards, 256 for AMD Radeon (LoL) and 16 by default if the device is neither.

Hope this wall of text has helped you better understand OpenACC and parallel programming in general. Thats where Part-1 of this ends. Part-2 will cover profiling, tweaking and more best practices.

I'd like to thank [@JeffLarkin](http://www.twitter.com/JeffLarkin) for releasing all this awesome training content on the internet and for patiently guiding a newbie like myself through some of the trickier bits.
