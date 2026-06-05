---
title: Latent Terrain Synthesis
description: Building new musical instruments that compose and interact with AI audio generators.
layout: libdoc_page.liquid
permalink: index.html
date: git Last Modified
tocEnabled: false
---
{% alert 'Added support for [stable-audio-3 SAME autoencoder](https://github.com/jasper-zheng/streamable-same-s), example MaxMSP/Max4Live devices will be available soon.', 'warning', 'News' %}

## Welcome   

Neural audio codec (autoencoder) is a module used in many AI music generation systems. By unpacking a codec, one can directly interact with the sound generation process, and build tools that link to sensors, hardware, gestural controllers...

Latent terrain is a tool to build corpus-based sound spaces/maps/materials to steer neural audio codecs (such as [RAVE](https://github.com/acids-ircam/RAVE), [SAME used by Stable Audio 3](https://github.com/jasper-zheng/streamable-same-s), [Music2Latent](https://github.com/SonyCSLParis/music2latent)). A terrain is a surface map for the codec's latent space, taking coordinates in a control space as inputs, and producing continuous real-time latent vectors that can be used for sound synthesis.

Latent terrain aims to open up the creative possibilities of **latent space walk**, allowing one to adapt the latent space of a codec to easy-to-navigate interfaces. An example latent space walk with Stable Audio Open 1.0:

<video controls="" loop="" playsinline="" aria-labelledby="video-label" src="./assets/stableaudio-demo.mp4" width="70%"></video>

### Example applications

* [Steering a neural audio autoencoder](#) (tutorial coming soon).
* [Building 1D/2D latent granular synthesiser](#) (tutorial coming soon).
* Latent looping device.

## Supported codecs 

Latent terrain can work with any [audio autoencoder](https://github.com/acids-ircam/creative_ml/blob/main/08_variational_ae_flows.pdf) as long as it offers latent variables. However, only a limited number of them have been implemented for MaxMSP, and we have only tested the following models:  

* [RAVE](https://github.com/acids-ircam/RAVE) <br>Realtime Audio Variational autoEncoder for fast and high-quality neural audio synthesis, by Antoine Caillon and Philippe Esling.   
* [Music2Latent-Scripted](https://github.com/jasper-zheng/music2latent-scripted) <br>Music2Latent is a Consistency Autoencoder to encode and decode audio samples, by Marco Pasini, Stefan Lattner, and George Fazekas. We're using a scripted fork of the original repository.   
* [Stable Audio Open 1.0 (autoencoder)](https://github.com/jasper-zheng/streamable-stable-audio-open.git) <br>**[Only supported by nn.terrain oct-2025 version]** The pretransform audoencoder in Stable Audio Open 1.0 (not the latent diffusion model, only the audoencoder).
* [SAME-S (Semantically-Aligned Music Autoencoder)](https://github.com/jasper-zheng/streamable-same-s) <br>**[Only supported by nn.terrain oct-2025 version]** The pretransform audoencoder in Stable Audio 3 (not the latent diffusion model, only the audoencoder).


## Get started 

* [Download and Installation](/installation)
* [Instructions](/instructions)
* [Pre-Trained Terrains (Presets)](/pre-trained)
* [Compiling from Source](/compile)

## Get in touch

Hi, this is Shuoyang (Jasper). `nn.terrain~` is part of my ongoing PhD work on **Discovering Musical Affordances in Neural Audio Synthesis**, supervised by Anna Xambó Sedó and Nick Bryan-Kinns, and part of the work has been (will be) on putting AI audio generators into the hands of composers/musicians.

Therefore, I would love to have you involved in it - if you have any feedback, a features request, a demo / a device / or anything made with nn.terrain, I would love to hear. If you would like to collaborate on anything, please leave a message in this [feedback form](https://forms.office.com/e/EJ4WHfru1A).  

## How To Cite  

If you use the software or the resources, we would appreciate citations to the following reference:  

*Shuoyang Jasper Zheng, Keigo Yoshida, Nico García-Peguinho, Jiatong Liu, Dan Hearn, Anna Xambó Sedó, and Nick Bryan-Kinns*. 2026. **Latent Terrain: Adapting Neural Audio Autoencoders as Design Materials in NIME**. In Proceedings of the International Conference on New Interfaces for Musical Expression. [pdf](https://qmro.qmul.ac.uk/xmlui/handle/123456789/128276)


```
@inproceedings{zheng_latent_2026,
	address = {London, UK},
	title = {Latent Terrain: Adapting Neural Audio Autoencoders as Design Materials in NIME},
	booktitle = {Proceedings of the International Conference on New Interfaces for Musical Expression},
	author = {Zheng, Shuoyang Jasper and Yoshida, Keigo and García-Peguinho, Nico and Liu, Jiatong and Hearn, Dan and Xambó Sedó, Anna and Bryan-Kinns, Nick},
	year = {2026}
}
```

## Acknowledgements

Shuoyang Zheng, the author of this work, is supported by the UKRI Centre for Doctoral Training in Artificial Intelligence and Music [EP/S022694/1].
