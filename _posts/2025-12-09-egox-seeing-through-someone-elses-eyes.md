---
layout: post
title: "egox: seeing through someone else's eyes"
date: 2025-12-09
tags: [computer-vision, video-generation, diffusion-models, lora]
---

<div class="blog-post">

## the big idea

nvidia continues cooking, goddamn. this paper introduces **egox**, a framework that takes a regular third-person (exocentric) video of some natural scene and generates a completely new video from a subject's first-person (egocentric) point of view — as if that subject was a robot and you're watching through the camera on their head.

think about it: you film a scene from a normal camera angle, and the model hallucinates what it would look like *from the perspective of someone inside the scene*. that's a wild capability.

<figure class="blog-figure">
  <img src="{{ '/assets/img/blog/egox-results.png' | relative_url }}" alt="egox results: exocentric input vs generated egocentric output">
  <figcaption>left: exocentric (third-person) input frames. right: egox-generated egocentric (first-person) outputs. no ground-truth ego video needed.</figcaption>
</figure>

## how it works

the pipeline is surprisingly elegant for what it pulls off:

1. **3d reconstruction** — they first build a 3d point cloud from the input video. this gives the model geometric understanding of the scene and serves as a spatial prior for the egocentric viewpoint.

2. **concatenation trick** — they take this egocentric prior and concatenate it *width-wise* with the exocentric video frames. this combined representation gets fed into the video generation model.

3. **lora-finetuned diffusion** — the video generation backbone is a large-scale pretrained video diffusion model. instead of training from scratch, they slap on a **lora adapter** and finetune it for this exo-to-ego task. they also use a geometry-guided self-attention mechanism to make sure the model pays attention to the spatially relevant parts of the scene.

the output? coherent, realistic first-person video that preserves the scene geometry and content while synthesizing entirely new viewpoints — including regions that were never visible in the original footage.

## the bigger picture

> i guess this has been true for a while now, but the ml dream of having a bunch of powerful pretrained models that you can then just fine-tune (aka post-train) for some cool downstream task is fully real now. 8th grade me would be pogging tf out.

this paper is a perfect example of the paradigm shift. you don't need to train a massive model from scratch for a novel task anymore. you take something like a video diffusion model that already understands spatiotemporal dynamics, bolt on a lightweight lora, give it the right conditioning signal (the 3d prior + exocentric frames), and suddenly it can do something it was never explicitly trained to do.

the applications are nuts — robotics (imagine generating training data for robot perception from regular video), vr/ar content creation, embodied ai, film production. anytime you want to ask "what would this look like from *their* perspective?" you now have a plausible answer.

**paper:** [arxiv:2512.08269](https://arxiv.org/abs/2512.08269){:target="_blank"}

</div>
