---
layout: post
title: "Performance Prediction of CUDA kernels"
date: 2016-04-14 06:38:26 +0530
comments: true
categories: CUDA, OpenACC 
---
This post is going to be more loud thinking, and less code - ('cuz there isn't any yet :) )

NVIDIA CUDA is supported on a wide range of hardware platforms they sell, right from their Tegra Mobile SoCs, notebook GPUs (thats what I have right now), desktop and workstation class GPUs, server class Tesla series, the most recent (and powerful) of which is the mighty [Tesla P100](http://www.nvidia.com/object/tesla-p100.html).

Over side a wide range of hardware and their relative cost, there must be a way for a potential buyer or user to establish (even approximately) how much $$ spent would result in what improvement in performance of their CUDA or OpenACC accelerated code. A couple of ways come to mind on how to achieve this, listed below.

1. Create C++/SystemC models of _all_ these available GPUs, and run your program on these to artifically judge their performance. Or,

2. Maybe they(NVDA) could provide the capability to 'test-drive' these GPUs, say in the cloud for some trial period for users to judge their potential return on hardware investment. I recall seeing a link for something this somewhere in [CUDAZone](https://developer.nvidia.com/cuda-zone) - though only for their Tesla M40 series. My guess is this service is made available via an [GPU enabled AWS instance](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html) that actually has an M40. So this idea could work, but is not scalable because we'd need a whole room full of the entire product portfolio that we need to then allow N users access to. Bleh.

3. Hey, its April 2016. No, its not super relevant other than the fact that it'll mark a few months of my machine learning experience and that the world has just been introduced to DGX-1. More the latter :) The point I'm making is we could adopt a **Machine Learning based approach**! Let me summarize this below:  

	* Create a feature vector for each CUDA kernel - This could be a bunch of stats from [nvprof](http://docs.nvidia.com/cuda/profiler-users-guide/#axzz46EPgphj2), NVIDIA's handy univeral profiler for all GPGPU code. Selecting the right features is the first step here. We'd then do all the usual tricks with it. Normalize, scale etc, etc.

	* Then, once we can represent any CUDA kernel in the world with a feature vector, we'd now need the data. Now the labelled training data in this case would be pairs of features-wallclock time. We could have the same kernels run on a range of hardware and generate the full spread of traning data. Given that nvprof already knows a whole lot about your kernel, this data collection would be best handled by it. Maybe, and I'm getting crazy here - we could even crowdsource that data! Many programs already do that type of thing for usage data, so NVIDIA could add that to nvprof ('_Do you want to submit anonymous report stats to NVIDIA? Click Yes to help the machines rise._') That way, a ton of data would pour in (and keep pouring in) from customers, developers and datacenters all around the world. (Well thats only if the option to upload said data is not at an annoying point - like firefox meekly asks after it crashes. Do you want to report this.. *angry shouts* I dont care about you, damn mozilla! I just lost all my browsing data!)
I see the availability of data as the real bottleneck here for someone to create that. Once again, an example of the fact that [advances in machine learning are not going to come from people with the best ideas, algorithms or even the best hardware - but by who has the data](https://medium.com/summer-ai/ai-s-big-trade-secret-a0d59110d6e3#.7z3h0ak3t). Information is power.

	* Say you have this data. Then you could run your favorite regression algorithm to predict this! Bloody Brilliant! ..the added awesomeness comes from the fact that the crowd sourced data is like free fuel for this prediction engine!

But, I dont have that data, or the hardware to collect it. So, this idea kind of hits a dead end but I'm leaving it around on the blog for now. I wonder if I could publish someplace...
