---
layout: page
title: blog
permalink: /blog
---

<div class="blog-list">
{% for post in site.posts %}
  <a href="{{ post.url | relative_url }}" class="blog-card">
    <div class="blog-card-title">{{ post.title }}</div>
    <div class="blog-card-date">{{ post.date | date: "%b %d, %Y" }}</div>
    {% if post.tags.size > 0 %}
    <div class="blog-card-tags">
      {% for tag in post.tags %}
        <span class="blog-tag">{{ tag }}</span>
      {% endfor %}
    </div>
    {% endif %}
  </a>
{% endfor %}
</div>
