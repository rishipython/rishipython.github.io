---
layout: post
title: "egox: seeing through someone else's eyes"
date: 2025-12-09
tags: [computer-vision, video-generation, diffusion-models, lora]
---

nvidia continues cooking goddamn. [arxiv:2512.08269](https://arxiv.org/abs/2512.08269){:target="_blank"}

this basically is able to take in a video of some natural scene, and then generate a video from some subject’s point of view as if that subject was a robot and its point of view is obtained from a camera.

<figure class="blog-figure">
  <img src="{{ '/assets/img/blog/egox-results.png' | relative_url }}" alt="egox results">
  <figcaption>exocentric input (left) to egocentric output (right)</figcaption>
</figure>

## how it works

they first make a 3d recon point cloud from the video, and use that to make a prior for the ego. then they concatenate that width wise with the exocentric video and shove it into a video gen model (which they fine tune with lora) and then that gives the ego output.

## thoughts

ig this has been true for a while now but the ml dream of having a bunch of powerful pretrained models that you can then just fine tune (aka post train) for some cool task is fully real now. 8th grade me would be pogging tf out.
