---
layout: post
title: "So you want to CUDA?"
date: 2016-02-29 19:12:59 +0530
comments: true
categories: CUDA
---

This is a post about various available resources, and how you could go about becoming a real CUDA pro. This post isn't about convincing you about why you should definitely learn CUDA - I'll leave that to the voices in and around your head.

To start out, I would highly recommend going through the free MOOC from Udacity - [Intro to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344). This is a course that isn't too technical right off the bat and yet its assignments are non-trivial and could also be a bit challenging for some. But they really help you get some real world exposure to parallel programming in general, apart from the CUDA specific knowledge you would gain in the process. The course really helps develop a 'think parallel' mindset - which I feel is as important, (if not more) compared to the knowledge of the actual semantics of a specific programming language or platform. The best part? You can do this without any special hardware - its all in the cloud!

Along with the udacity course, there are a couple of great texts I would urge you guys to get:

1. [Sanders & Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming](http://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685/ref=pd_bbs_sr_1/103-9839083-1501412?ie=UTF8&s=books&qid=1186428068&sr=1-1)
This first one is a good text for beginners because it presents a very approachable learning curve. It has lots of small code examples, something I personally like. It lives up to its title in that respect. Having digestible code examples allow you to tinker with different concepts till you get the hang of things, without the overhead of programming full assignments. The book's code is available for download on [NVIDIA's site here](https://developer.nvidia.com/cuda-example) and serves as handy reference later on as well. However, this book does not go too deep into the application side, and the 'bigger picture' of parallel programming. Thats where the next book is better.

2. [Kirk & Hwu. Programming Massively Parallel Processors](http://store.elsevier.com/Programming-Massively-Parallel-Processors/David-Kirk/isbn-9780124159921/)
This book definitely dives a bit deeper with regard to the technical aspects. Since it was created keeping in mind a typical graduate-level course on this subject, each chapter has exercises as well. Chapter 6 on performance considerations, and Chapter 7 on floating point math are two I consider particularly important for a learner to understand early on. The chapters on computational thinking and OpenCL make this a complete text on parallel programming. In addition, the code for the case studies discussed has been made available [freely available online.](http://www.ks.uiuc.edu/Research/vmd/projects/ece498/)

And as you get more hands-on with the programming aspects of it, you will be able to appreciate the wealth of info in the [CUDA Best Practices Guide](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). I actually have a printed copy I refer to often.

Among must-read blogs, there is [NVIDIA's Parallel Forall blog](https://devblogs.nvidia.com/parallelforall/) that has some really well written articles on a wide variety of topics and applications in accelerated computing. Most of the CUDA related content posted here is best understood by someone who already has a higher-than-basic understanding of CUDA. Still, do subscribe.

I almost forgot to mention the [hands-on labs offered by NVIDIA via qwiklabs](https://nvidia.qwiklab.com/). While these aren't anywhere as fully featured as the resources mentioned above, these serve as good exercises nonetheless. These are also in the cloud, hosted on [GPU enabled AWS instances](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using_cluster_computing.html).

Though there are lots of such free(-ish) learning resources out there, you really need access to some hardware in order to really sharpen your skills. But this does not mean you need to spend big bucks. Lots of older GPUs support CUDA, and if you're part of an academic institution, you could also look at [Nvidia's hardware grant program](https://developer.nvidia.com/academic_hw_seeding). You can also run your CUDA code on your multicore CPU (coming-soon-a-link-to-a-tutorial-on-how-to-do-that)

And finally, you need to have a project that you really want to invest your sweat and skills into. Something to tie all of this together. It could be a cryptographic algorithm, or a massively parallel numerical method or perhaps something cool in the field of machine learning. Maybe you could build a encoder/decoder for an image format. Basically, you can CUDA-fy mostly anything compute intensive around you. I'm not saying that _everything_ is going to work well with CUDA - thats the topic for another blog post. But as someone starting out, one shouldn't be overly picky about that.

Oh, and theres always [stackoverflow](http://www.stackoverflow.com/questions/tagged/cuda), [/r/CUDA](www.reddit.com/r/CUDA) and [NVIDIA's developer forum](https://devtalk.nvidia.com/default/board/53/accelerated-computing/) if you get stuck somewhere - or even just want to discuss your ideas.

As with any new endeavor, you will fail and learn a lot. But the key as always is to persevere and accept experience that comes your way, whatever the form.
