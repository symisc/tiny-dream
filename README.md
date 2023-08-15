<h1 align="center">TINY DREAM<br><br>An embedded, Header Only, Stable Diffusion implementation in C++<br><a href="https://pixlab.io/tiny-dream">pixlab.io/tiny-dream</a></h1>

![td_screen_website](https://github.com/symisc/tiny-dream/assets/4615920/b4e9f6b3-4019-4d48-9e3e-879a071213a5)

[![API documentation](https://img.shields.io/badge/API%20documentation-Ready-green.svg)](https://pixlab.io/tiny-dream)
[![dependency](https://img.shields.io/badge/dependency-none-ff96b4.svg)](https://pixlab.io/tiny-dream#downloads)
[![license](https://img.shields.io/badge/License-dual--licensed-blue.svg)](https://pixlab.io/tiny-dream#license)

* [Introduction](#tiny-dream)
* [Features](#td-features)
* [Getting Started](#td-start)
* [Downloads](https://pixlab.io/tiny-dream#downloads)
* [Project Roadmap](https://pixlab.io/tiny-dream#roadmap)
* [License](https://pixlab.io/tiny-dream#license)
* [C++ API Reference Guide](https://pixlab.io/tiny-dream#cpp-api)
* [Issue Tracker](https://github.com/symisc/tiny-dream/issues)
* [Related Projects](#td-projects)

<h2 id="tiny-dream">Introducing PixLab's Tiny Dream</h2>
<p><a href="https://pixlab.io/tiny-dream" target="_blank">Tiny Dream</a> is a header only, dependency free, <strong>partially uncensored, Stable Diffusion implementation written in C++</strong> with primary focus on CPU efficiency, and smaller memory footprint. <strong>Tiny Dream</strong> runs reasonably <a href="https://pixlab.io/tiny-dream#features">fast</a> on the average consumer hardware, <a href="https://pixlab.io/tiny-dream#features">require</a> <strong>only 5.5 GB of RAM</strong> to execute, does not enforce Nvidia GPUs presence, and is designed to be <a href="https://pixlab.io/tiny-dream#getting-started">embedded</a> on larger codebases (host programs) with an easy to use <a href="https://pixlab.io/tiny-dream#cpp-api">C++ API</a>. The possibilities are literally endless, or at least extend to the boundaries of Stable Diffusion's latent manifold.</p>
<h2 id="td-features">Features</h2>
<em>For the extensive list of features, please refer to the officical documentation <a href="https://pixlab.io/tiny-dream#features" target="_blank"><strong>here</strong></a>.</em>
<br><br>
<ul>
  <li><strong>OpenCV Dependecny Free</strong>: Only <font face="courier"><a href="https://github.com/nothings/stb/blob/master/stb_image_write.h" target="_blank">stb_image_write.h</a></font> from the excellent <a href="https://github.com/nothings/stb/" target="_blank">stb <em class="ti ti-new-window"></em></a> single-header, public domain C library is required for saving images to disk.</li>
  <li><strong>Lowest Run-Time <a href="https://pixlab.io/tiny-dream#features" target="_blank">Memory Footprint</a> Recorded for a Stable Diffusion Implementation</strong>.</li>
  <li><strong>Straightforward to <a href="https://pixlab.io/tiny-dream#getting-started" target="_blank">Integrate on Existing Codebases</strong>: Just drop <font face="courier"><em>tinydream.hpp</em></font> and <font face="courier"><em>stb_image_write.h</em></font> on your source tree with the <a href="https://pixlab.io/tiny-dream#downloads"><strong>Pre-trained Models & Assets</strong></a>.</li>
    <li><strong>Reasonably fast on Intel/AMD CPUs (<a href="https://pixlab.io/tiny-dream#bench">Benchmarks</a>)</strong>: With TBB threading and SSE/AVX vectorization.</li>
    <li><strong>Support <a href="https://github.com/xinntao/Real-ESRGAN" target="_blank">Real-ESRGAN</a>, A Super Resolution Network Upscaler</strong>.</li>
    <li><strong>Full Support for Words Priority</strong>: Instruct the model to pay attention, and <strong>give higher priority</strong> to word (<em>keywords</em>) surrounded by parenthesis <em><strong>()</strong></em>.</li>
    <li><strong>Support for Output Metadata</strong>: Link meta information to your output images such as <em>copyright notice</em>, <em>comments</em>, or any other meta data you would like to see linked to your image.</li>
    <li><strong>Support for Stable Diffusion Extra Parameters</strong>: Adjust <a href="https://pixlab.io/tiny-dream#tiny-dream-method">Seed resizing</a> & <a href="https://pixlab.io/tiny-dream#tiny-dream-method">Guidance Scale</a>.</li>
</ul>
<h2 id="td-start">Getting Started with Tiny-Dream</h2>
<p>Integrating <span style="font-family:Montserrat"><strong>Tiny Dream</strong></span> on your existing code base is straightforward. Here is what to do without having to do a lot of tedious reading and configuration:</p>
<ol>
  <li></li>
</ol>
<h2>Official Docs & Resources</h2>
<table class="table">
    <tbody>
      <tr>
        <th><a href="https://pixlab.io/tiny-dream#downloads">Pre-Trained Models & Assets Downloads</a></th>
        <td><a href="https://pixlab.io/tiny-dream#getting-started">Getting Started Guide</a></th>
        <td><a href="https://pixlab.io/tiny-dream#license">Licensing</a></td>
        <td><a href="https://pixlab.io/tiny-dream#cpp-api">C++ API Reference Guide</a></td>
        <td><a href="https://pixlab.io/tiny-dream#roadmap">Project Roadmap</a></td>
        <td><a href="https://pixlab.io/tiny-dream#features">Features</a></td>
      </tr>
    </tbody>
  </table>
  <h2 id="td-projects">Related Projects</h2>
