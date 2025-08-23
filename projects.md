---
layout: page
title: projects
permalink: /projects
---

<div class="cards-grid">

  <!-- ALLNet -->
  <article class="card">
    <div class="thumb">
      <img src="{{ '/assets/img/papers/allnet.jpg' | relative_url }}" alt="allnet paper cover">
    </div>
    <div class="content">
      <div class="title">ALLNet</div>
      <div class="subtitle">hybrid cnn for leukemia · 92% acc</div>
      <div class="links">
        <a href="https://ieeexplore.ieee.org/document/9669840/" target="_blank" rel="noopener">ieee publication</a>
        <a href="https://arxiv.org/pdf/2108.08195" target="_blank" rel="noopener">arxiv paper</a>
        <a href="https://github.com/rishipython/ALL-Cell-Classification" target="_blank" rel="noopener">code</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          published in ieee bibm 2021 (4 citations). trained a novel hybrid CNN architecture to identify acute lymphocytic leukemia (the most common childhood cancer) from white blood cell images; beat out contemporary CNN architectures (resnet, inception, vgg) and achieved 92% test accuracy.
        </div>
      </details>
    </div>
  </article>

  <!-- Parkinson’s GB + mRMR -->
  <article class="card">
    <div class="thumb">
      <img src="{{ '/assets/img/papers/parkinsons.jpg' | relative_url }}" alt="parkinson’s paper cover">
    </div>
    <div class="content">
      <div class="title">identifying Parkinson's from a paient's voice</div>
      <div class="subtitle">audio features · 90% acc</div>
      <div class="links">
        <a href="https://link.springer.com/chapter/10.1007/978-3-031-18344-7_24" target="_blank" rel="noopener">ftc publication</a>
        <a href="https://docs.google.com/document/d/1Ia2IHN8m70mVJsTZ982atKG6OguuAN87nQkdJM6tRYw/edit?usp=sharing" target="_blank" rel="noopener">paper</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          published in future technologies conference. trained gradient boosting model to identify parkinson's from features calculated from patient audio data, while using mRMR to prune features (90% accuracy).
        </div>
      </details>
    </div>
  </article>

  <!-- One Eye is All You Need -->
  <article class="card">
    <div class="thumb">
      <img src="{{ '/assets/img/papers/one-eye.jpg' | relative_url }}" alt="one eye is all you need paper cover">
    </div>
    <div class="content">
      <div class="title">cnn for tracking eye movements</div>
      <div class="subtitle">gaze estimation · 1.4–2.3 cm error</div>
      <div class="links">
        <a href="https://arxiv.org/pdf/2211.11936" target="_blank" rel="noopener">paper</a>
        <a href="https://github.com/rishipython/One-Eye-is-All-You-Need-Lightweight-Ensembles-for-Gaze-Estimation-with-Single-Encoders" target="_blank" rel="noopener">code</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          trained a squeeznet model on millions of images with aws ec2 to predict user eye movement from a single eye image (increasing efficiency and robustness in practical applications); single-eye tracking with 1.4 cm (two eyes) and 2.3 cm (single eye), achieving comparable results to mit/google benchmarks (12 citations).
        </div>
      </details>
    </div>
  </article>

</div>
