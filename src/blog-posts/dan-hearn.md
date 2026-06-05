---
title: "Latent space artists: A conversation with Dan Hearn"
description: In this series of interviews, we invited artists to share the design thinking and experience behind the making process of their demo projects. 
layout: libdoc_page.liquid
permalink: "{{ libdocConfig.blogSlug }}/conversation-with-dan-hearn/index.html"
tags:
    - post
    - demos
date: 2025-12-14
ogImageUrl: "../assets/dan.jpg"
author: Dan Hearn
tocEnabled: false
---

{% include "artist-interview.md" %}

{% iconCard 'Artist Bio', '**Dan Hearn** is a creative technologist and musician working with interaction, sound, and machine learning. His work centres on building tools and interfaces that invite exploration rather than automation. He works across software, hardware, and machine learning to develop prototypes, instruments, and installations.', 'user' %} 


### An ambient soundtrack  

<audio controls preload="metadata" src="../../assets/NIME-Dan.mp3"></audio>


<br><br>


##### Before starting the nn.terrain project, did you have any prior hands-on experience with Neural Audio Synthesis (NAS) (e.g., RAVE), or more broadly AI music?
**Dan:** In my work and research I have experience working with NAS and AI music, particularly surrounding the interaction affordances of generative models. In my own creative practice as a musician however, I have experimented very little with these models.

##### What was your workflow when making the project? How did you integrate nn~ and nn.terrain~ into your work?
**Dan:** At first, I found it difficult to incorporate nn.terrain~ into my workflow. When making music, I typically work by resampling synths and field recordings, but initially I approached nn.terrain~ more as a synthesiser. Used in this way, I found it difficult to ‘play’ and to synchronise with Ableton.

Through trial and error, I eventually arrived at a workflow in which nn.terrain~ became part of my existing production practice. Rather than functioning as a synthesizer, nn.terrain~ replaced conventional resampling tools in Ableton, such as warping and reversing. I would begin by recording a short five-second sample from a hardware synth or a brief field recording. In MaxMSP, I then encoded this audio and created a trajectory in order to train a terrain on the data. After training, I explored the terrain in play mode, generating short loops by moving the start and end positions across the latent space, and recording the resulting audio back into Ableton. 

I also experimented with scaling and offsetting the nn.terrain~ output signal before the decoder, which allowed me to shape the transients of the generated audio. In some cases, this proved effective for producing sustained drone textures. I think of this overall process as latent resampling.

Compared to the manual process of slicing and warping samples, this approach felt more explorative and serendipitous, and ultimately more expressive and immediate. The use of NAS models introduced additional artefacts that contributed to a distinctive aesthetic, complementing the resampling process, which often aims to reveal unexpected textures within familiar material.

After resampling with nn.terrain~, I used Ableton’s Simpler device to slice the recorded audio by transients and exported these slices to a drum rack for MIDI triggering. I also used Granulator III, Sampler, and Wavetable to create synthesiser instruments from the resampled material. All audio in the final track was generated through this nn.terrain~-based resampling workflow.

Throughout this process, I used the Stable Audio Open (SAO) VAE. Although I experimented with other models, SAO VAE produced the most consistent results across a wide range of input material, making it suitable for resampling rather than purely timbre transfer. While the output audio was generally robust, certain input sounds performed better than others. Drum loops, vocals, and piano samples reconstructed with high fidelity when explored within the terrain, whereas more textural field recordings and pad-like synth sounds often exhibited jittering artefacts. I thought this was likely due to an underrepresentation of such sounds in the model’s training data.

##### Is working with neural audio synthesis different from working with other materials. 

**Dan:** Yes, as I mentioned, there are unique aesthetic qualities that come with using NAS models. Additionally, in my experience they are more difficult to predictably control than conventional methods. Whilst nn.terrain~ significantly improves predictability.

##### Could you highlight some features of your work that make it valuable/unique/special, or that you're proud of?

**Dan:** One aspect of the work I am particularly proud of is the integration of nn.terrain~ into my sample-based workflow. While it may not be immediately apparent to a listener that a NAS model was used in the creation of the track, this is partly intentional. I treated it as a tool, rather than a novelty. 

In retrospect, I could have learned more explicitly into the distinctive aesthetic qualities of NAS models. I value the fact that I was able to adapt this technology to my own way of working, rather than reshaping my workflow around the tool. 

A key motivation behind this work is a response to the increasingly negative sentiment surrounding generative audio technologies in music production. Much of this criticism is directed at tools that aim to automate or replace aspects of musical authorship. I see nn.terrian~ as a valuable counter example, as it opens novel creative possibilities and aesthetics while maintaining the composer’s agency and decision-making at the centre of the process.
