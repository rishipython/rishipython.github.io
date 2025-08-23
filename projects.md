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
      <div class="title">allnet</div>
      <div class="subtitle">hybrid cnn for leukemia · 92% acc</div>
      <div class="links">
        <a href="#" target="_blank" rel="noopener">paper</a>
        <a href="#" target="_blank" rel="noopener">code</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          published in ieee bibm 2021 (4 citations). trained on wbc images; improved baseline with a lightweight hybrid cnn that kept inference fast while bumping accuracy to 92%.
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
      <div class="title">parkinson’s via gb + mrmr</div>
      <div class="subtitle">audio features · 90% acc</div>
      <div class="links">
        <a href="#" target="_blank" rel="noopener">paper</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          published in future technologies conference. used mrmr to prune features and gradient boosting for robust performance on patient voice data (90%).
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
      <div class="title">one eye is all you need</div>
      <div class="subtitle">gaze estimation · 1.4–2.3 cm</div>
      <div class="links">
        <a href="#" target="_blank" rel="noopener">paper</a>
        <a href="#" target="_blank" rel="noopener">code</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          squeeznet + pytorch on aws; single-eye tracking with 1.4 cm (two eyes) and 2.3 cm (single eye)—comparable to mit/google benchmarks (12 citations).
        </div>
      </details>
    </div>
  </article>

  <!-- Guardian Image Search -->
  <article class="card">
    <div class="thumb">
      <img src="{{ '/assets/img/projects/guardian.jpg' | relative_url }}" alt="guardian image search">
    </div>
    <div class="content">
      <div class="title">guardian image search</div>
      <div class="subtitle">openclip + llm metadata · agentic indexing</div>
      <div class="links">
        <a href="#" target="_blank" rel="noopener">post</a>
        <a href="#" target="_blank" rel="noopener">code</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          pm + lead engineer; built an llm-augmented pipeline with openclip embeddings and qwen2.5 for metadata, improving editorial retrieval quality.
        </div>
      </details>
    </div>
  </article>

  <!-- Web Agents -->
  <article class="card">
    <div class="thumb">
      <img src="{{ '/assets/img/projects/web-agents.jpg' | relative_url }}" alt="web agents">
    </div>
    <div class="content">
      <div class="title">web agents</div>
      <div class="subtitle">navigate/analyze reddit, gitlab, e-com</div>
      <div class="links">
        <a href="#" target="_blank" rel="noopener">code</a>
      </div>
      <details>
        <summary>read more</summary>
        <div class="more">
          agents with tool-use + evaluation harnesses for browsing and interaction; designed for reproducible benchmarking.
        </div>
      </details>
    </div>
  </article>

  <!-- INR Tomography UQ -->
  <article class="card">
    <div class="thumb">
      <img src="{{ '/assets/img/projects/inr-tomo.jpg' | relative_url }}" alt="inr tomography uq">
    </div>
    <div class="content">
      <div class="title">inr tomography uq</div>
      <div class="subtitle">missing-wedge · epistemic uncertainty</div>
      <details>
        <summary>read more</summary>
        <div class="more">
          quantifying epistemic uncertainty of implicit neural representations on tomographic missing-wedge problems; exploratory results + visualizations.
        </div>
      </details>
    </div>
  </article>

</div>
