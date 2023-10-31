<h1 align="center">TINY DREAM<br><br>An embedded, Header Only, Stable Diffusion Inference C++ Library<br><a href="https://pixlab.io/tiny-dream">pixlab.io/tiny-dream</a></h1>

![td_screen_website](https://github.com/symisc/tiny-dream/assets/4615920/b4e9f6b3-4019-4d48-9e3e-879a071213a5)

<h5><em>Latest News</em> 🔥</h5>
<ul>
	<li><strong>Tiny Dream 1.7.5 <a href="https://github.com/symisc/tiny-dream/releases/tag/1.7.5">Released</a> - <a href="https://pixlab.io/tiny-dream">Get Started</a></strong>.</li>
</ul>

[![API documentation](https://img.shields.io/badge/API%20documentation-Ready-green.svg)](https://pixlab.io/tiny-dream)
[![dependency](https://img.shields.io/badge/dependency-none-ff96b4.svg)](https://pixlab.io/tiny-dream#downloads)
[![license](https://img.shields.io/badge/License-dual--licensed-blue.svg)](https://pixlab.io/tiny-dream#license)

* [Introduction](#tiny-dream)
* [Features](#td-features)
* [Getting Started](#td-start)
* [Downloads](https://pixlab.io/tiny-dream#downloads)
* [Project Roadmap](#roadmap)
* [License](https://pixlab.io/tiny-dream#license)
* [C++ API Reference Guide](https://pixlab.io/tiny-dream#cpp-api)
* [Issues Tracker](https://github.com/symisc/tiny-dream/issues)
* [Related Projects](#td-projects)

<h2 id="tiny-dream">Introducing PixLab's Tiny Dream</h2>
<p><a href="https://pixlab.io/tiny-dream" target="_blank">Tiny Dream</a> is a header only, dependency free, <strong>partially uncensored, Stable Diffusion implementation written in C++</strong> with primary focus on CPU efficiency, and smaller memory footprint. <strong>Tiny Dream</strong> runs reasonably <a href="https://pixlab.io/tiny-dream#features">fast</a> on the average consumer hardware, <a href="https://pixlab.io/tiny-dream#features">require</a> <strong>only 1.7 ~ 5.5 GB of RAM</strong> to execute, does not enforce Nvidia GPUs presence, and <strong>is designed to be <a href="https://pixlab.io/tiny-dream#getting-started">embedded</a> on larger codebases (host programs) with an easy to use <a href="https://pixlab.io/tiny-dream#cpp-api">C++ API</a></strong>. The possibilities are literally endless, or at least extend to the boundaries of Stable Diffusion's latent manifold.</p>
<h2 id="td-features">Features 🔥</h2>
<em>For the extensive list of features, please refer to the official documentation <a href="https://pixlab.io/tiny-dream#features" target="_blank"><strong>here</strong></a>.</em>
<br><br>
<ul>
  <li><strong>OpenCV Dependency Free</strong>: Only <font face="courier"><a href="https://github.com/nothings/stb/blob/master/stb_image_write.h" target="_blank">stb_image_write.h</a></font> from the excellent <a href="https://github.com/nothings/stb/" target="_blank">stb <em class="ti ti-new-window"></em></a> single-header, public domain C library is required for saving images to disk.</li>
  <li><strong>Smallest, Run-Time <a href="https://pixlab.io/tiny-dream#features" target="_blank">Memory Footprint</a> for Running Stable Diffusion in Inference</strong>.</li>
  <li><strong>Straightforward to <a href="https://pixlab.io/tiny-dream#getting-started" target="_blank">Integrate on Existing Codebases</strong>: Just drop <font face="courier"><em>tinydream.hpp</em></font> and <font face="courier"><em>stb_image_write.h</em></font> on your source tree with the <a href="https://pixlab.io/tiny-dream#downloads"><strong>Pre-trained Models & Assets</strong></a>.</li>
    <li><strong>Reasonably fast on Intel/AMD CPUs (<a href="https://pixlab.io/tiny-dream#bench">Benchmarks</a>)</strong>: With TBB threading and SSE/AVX vectorization.</li>
    <li><strong>Support <a href="https://github.com/xinntao/Real-ESRGAN" target="_blank">Real-ESRGAN</a>, A Super Resolution Network Upscaler</strong>.</li>
    <li><strong>Full Support for Words Priority</strong>: Instruct the model to pay attention, and <strong>give higher priority</strong> to word (<em>keywords</em>) surrounded by parenthesis <em><strong>()</strong></em>.</li>
    <li><strong>Support for Output Metadata</strong>: Link meta information to your output images such as <em>copyright notice</em>, <em>comments</em>, or any other meta data you would like to see linked to your image.</li>
    <li><strong>Support for Stable Diffusion Extra Parameters</strong>: Adjust <a href="https://pixlab.io/tiny-dream#tiny-dream-method">Seed resizing</a> & <a href="https://pixlab.io/tiny-dream#tiny-dream-method">Guidance Scale</a>.</li>
</ul>
<h2 id="td-start">Getting Started with Tiny-Dream 🔥</h2>
<p><strong>Integrating Tiny Dream on your existing code base is straightforward</strong>. Here is what to do without having to do a lot of tedious reading and configuration:</p>
<h4>Download Tiny-Dream</h4>
<ul>
  <li><a href="https://github.com/symisc/tiny-dream/releases">Download</a> the latest public release of Tiny Dream, and extract the package on a directory of your choice.</li>
  <li>Refer to the <a href="https://pixlab.io/tiny-dream#downloads">downloads section</a> to get a copy of the Tiny Dream source code as well as the <strong>Pre-Trained Models & Assets</strong>.</li>
</ul>
<h4>Embedding Tiny-Dream</h4>
    <ul>
      <li>The Tiny Dream source code <a href="https://pixlab.io/tiny-dream#downloads">comprise</a> <strong>only two header files</strong> that is <font face="courier"><strong>tinydream.hpp</strong></font> and <font face="courier"><strong>stb_image_write.h</strong></font>.</li>
      <li>All you have to do is drop these two C/C++ header files on your source tree, and <a href="https://pixlab.io/tiny-dream#tiny-dream-constructor">instantiate</a> a new <font face="courier">tinyDream</font> object as shown on the pseudo C++ code below:</li>
    </ul>
    
```
#include "tinydream.hpp"
/*
* Main Entry Point. The only required argument is the Positive Prompt.
* Passing a Negative Prompt (words separated by commas) is highly recommended though.
* 
* We recommend that you experiment with different seed & step values
* in order to achieve a desirable result.
* 
* ./tinydream "positive prompt" ["negative prompt"] [seed] [step]
*/
int main(int argc, char *argv[]) 
{
	tinyDream td; // stack allocated tinyDream object

	// Display the library current inference engine, version number, and copyright notice
	std::cout << tinyDream::about() << std::endl;
	
	// At least a positive prompt must be supplied via command line
	if (argc < 2) {
		std::cout << "Missing Positive (and potentially Negative) Prompt: Describe something you'd like to see generated..." << std::endl;
		std::cout << "Example of Prompts:" << std::endl;
		// Example of built-in Positive/Negative Prompts
		auto prompts = tinyDream::promptExample();
		std::cout << "\tPositive Prompt: " << prompts.first << std::endl;
		std::cout << "\tNegative Prompt: " << prompts.second << std::endl;
		return -1;
	}

	// Register a log handler callback responsible of 
	// consuming log messages generated during inference.
	td.setLogCallback(logCallback, nullptr);
	
	// Optionally, set the assets path if the pre-trained models
	// are not extracted on the same directory as your executable
	// The Tiny-Dream assets can be downloaded from: https://pixlab.io/tiny-dream#downloads
	td.setAssetsPath("/path/to/tinydream/assets"); // Remove or comment this if your assets are located on the same directory as your executable
	
	// Optionally, set a prefix of your choice to each freshly generated image name
	td.setImageOutputPrefix("tinydream-");
	
	// Optionally, set the directory where you want
	// the generated images to be stored
	td.setImageOutputPath("/home/photos/");
	
	int seedMax = 90;
	if (argc > 3) {
		/*
		* Seed in Stable Diffusion is a number used to initialize the generation. 
		* Controlling the seed can help you generate reproducible images, experiment
		* with other parameters, or prompt variations.
		*/
		seedMax = std::atoi(argv[3]);
	}
	int step = 30;
	if (argc > 4) {
		/*
		* adjusting the inference steps in Stable Diffusion: The more steps you use,
		* the better quality you'll achieve but you shouldn't set steps as high
		* as possible. Around 30 sampling steps (default value) are usually enough
		* to achieve high-quality images.
		*/
		step = std::atoi(argv[4]);
	}

	/*
	* User Supplied Prompts - Generate an image that matches the input criteria.
	* 
	* Positive Prompt (required): Describe something you'd like to see generated (comma separated words).
	* Negative Prompt (optional): Describe something you don't like to see generated (comma separated words).
	*/
	std::string positivePrompt{ argv[1] };
	std::string negativePrompt{ "" };
	if (argc > 2) {
		negativePrompt = std::string{ argv[2] };
	}

	/*
	* Finally, run Stable Diffusion in inference
	* 
	* The supplied log consumer callback registered previously should shortly receive
	* all generated log messages (including errors if any) during inference.
	* 
	* Refer to the official documentation at: https://pixlab.io/tiny-dream#tiny-dream-method
	* for the expected parameters the tinyDream::dream() method takes.
	*/
	for (int seed = 1; seed < seedMax; seed++) {
		std::string outputImagePath;

		td.dream(
			positivePrompt, 
			negativePrompt, 
			outputImagePath, 
			true, /* Set to false if you want 512x512 pixels output instead of 2048x2048 output */
			seed,
			step
		);

		// You do not need to display the generated image path manually each time via std::cout
		// as the supplied log callback should have already done that.
		std::cout << "Output Image location: " << outputImagePath << std::endl; // uncomment this if too intrusive
	}
	return 0;
}
```
<h4>Learn the Fundamentals (C++ API)</h4>
<ul>
	<li>The above code should be self-explanatory, and easy to understand for the average C++ programmer. The <strong>full C++ integration code</strong> for a typical application embedding Tiny Dream is located at: <a href="https://pixlab.io/tiny-dream#code-gist">pixlab.io/tiny-dream#code-gist</a>.</li>
	<li>As of this release, the library exposes a single class named <code>tinyDream</code> with the following exported methods:
		<ul>
			<li><a href="https://pixlab.io/tiny-dream#tiny-dream-constructor">tinyDream::tinyDream()</a> - <em>Constructor</em></li>
                        <li><a href="https://pixlab.io/tiny-dream#tiny-dream-method"><strong>tinyDream::dream()</strong></a> - <em>Stable Diffusion Inference</em></li>
                        <li><a href="https://pixlab.io/tiny-dream#set-img-output-method">tinyDream::setImageOutputPath()</a></li>
                        <li><a href="https://pixlab.io/tiny-dream#set-img-output-prefix">tinyDream::setImageOutputPrefix()</a></li>
			<li><a href="https://pixlab.io/tiny-dream#set-log-callback">tinyDream::setLogCallback()</a></li>
                        <li><a href="https://pixlab.io/tiny-dream#set-assets-path-method">tinyDream::setAssetsPath()</a></li>
                        <li><a href="https://pixlab.io/tiny-dream#prompt-example-method">tinyDream::promptExample()</a></li>
                        <li><a href="https://pixlab.io/tiny-dream#about-method">tinyDream::about()</a></li>
		</ul>
	</li>
	<li><strong>A step-by-step, detailed integration guide, and call logic of the above methods is located at: <a href="https://pixlab.io/tiny-dream#step-by-step-cpp">pixlab.io/tiny-dream#step-by-step-cpp</a></strong>.</li>
</ul>
<h4>Building Tiny-Dream</h4>
<ul>
	<li>Building Tiny-Dream from source require a modern C++17 compiler such as GCC 7 or later, Clang or Microsoft Visual Studio (MSVC).</li>
	<li>You also <strong>need to link to the default backend Tensor library</strong> in order to generate the executable.</li>
	<li>As of this release, <a href="https://github.com/Tencent/ncnn/wiki/how-to-build" target="_blank">NCNN <em class="ti ti-new-window"></em></a> is the default tensor library. On our <a href="#roadmap">Roadmap</a>, we plan to ditch <font face="courier">ncnn</font> to a less bloated tensor library such as <a href="https://sod.pixlab.io" target="_blank">SOD</a> or <a href="https://github.com/ggerganov/ggml " target="_blank">GGML</a> with focus on CPU efficiency</strong>.</li>
	<li>Alternatively, you can rely on a build manager such as CMAKE to build the executable for you. The Tiny-Dream repository repository already contain the necessarily CMAKE template to build the executable from source.</li>
	<li>An example of generating a heavy optimized executable without relying on a external build manager is shown just below:</li>
</ul>

```
git clone https://github.com/symisc/tiny-dream.git
cd tiny-dream
g++ -o tinydream boilerplate.cpp -funsafe-math-optimizations -Ofast -flto=auto  -funroll-all-loops -pipe -march=native -std=c++17 -Wall -Wextra `pkg-config --cflags --libs ncnn` -lstdc++ -pthread -Wl -flto -fopt-info-vec-optimized
./tinydream "pyramid, desert, palm trees, river, (landscape), (high quality)"
```
<h4>Get the Pre-Trained Models & Assets</h4>
<ul>
	<li>Once your executable built, <strong>you will need the Tiny Dream <a href="https://pixlab.io/tiny-dream#downloads">Pre-Trained Models & Assets</a> path accessible to your executable</strong>.</li>
	<li>The Tiny Dream assets comprise all pre-trained models (<strong>over 2GB as of this release</strong>) required by the <a href="https://pixlab.io/tiny-dream#tiny-dream-method"><font face="courier">tinyDream::dream()</font></a> method in order to run stable diffusion in inference.</li>
	<li>You can download the pre-trained models from the <a href="https://pixlab.io/tiny-dream#downloads">Download</a> section on the <a href="https://pixlab.io/">PixLab</a> website.</li>
	<li>Once downloaded, extract the assets ZIP archive in a directory of your choice (usually the directory where your executable is located), and set the full path via <font face="courier"><a href="https://pixlab.io/tiny-dream#set-assets-path-method">tinyDream::setAssetsPath()</a></font> or from the Tiny Dream <a href="https://pixlab.io/tiny-dream#tiny-dream-constructor">constructor</a>.</li>
</ul>
<h4>Continue with The C++ API Reference Guide</h4>
<p>The Tiny Dream <a href="https://pixlab.io/tiny-dream#cpp-api">C++ Interface</a>, provides detailed specifications for all of the various methods the Tiny Dream class exports. Once the reader understands the basic principles of operation for <strong>Tiny Dream</strong>, that <a href="https://pixlab.io/tiny-dream##cpp-api">document</a> should serve as a reference guide.</p>
<h2 id="roadmap">TODOs & Roadmap 🔥</h2>
<p>As we continue to develop and improve Tiny Dream, we have an exciting roadmap of future addons and enhancements planned. Refer to the Roadmap page at <a href="https://pixlab.io/tiny-dream#roadmap">pixlab.io/tiny-dream</a> or the <a href="https://blog.pixlab.io">PixLab Blog</a> for the exhaustive list of todos & ongoing progress...</p>
<ul>
	<li><strong>Move the Tensor library to a non bloated one such as <a href="https://sod.pixlab.io/">SOD</a> or <a href="https://github.com/ggerganov/ggml">GGML</a> with focus on CPU performance</strong>.</li>
	<li><strong>Provide a Cross-Platform GUI to Tiny Dream implemented in <a href="https://github.com/ocornut/imgui">Dear imGUI</a></strong>.</li>
	<li>Provide a Web-Assembly port to the library once the future Tensor library (SOD or GGML) ported to WASM.</li>
	<li>Output SVG, and easy to alter formats (potentially PSD) rather than static PNGs.</li>
	<li> Provide an Android, proof of concept, show-case APK.</li>
</ul>
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
  <h2 id="td-projects">Related Projects 🔥</h2>
  <p>You may find useful the following production-ready projects developed & maintained by <a href="https://pixlab.io">PixLab</a> | <a href="https://symisc.net">Symisc Systems</a>:</p>
  <ul>
	  <li><a href="https://sod.pixlab.io">SOD</a> - An Embedded, Dependency-Free, Computer Vision C/C++ Library.</li>
	  <li><a href="https://faceio.net">FACEIO</a> - Cross Browser, Passwordless Facial Authentication Framework.</li>
	  <li><a href="https://annotate.pixlab.io/">PixLab Annotate</a> - Online Image Annotation, Labeling & Segmentation Tool.</li>
	  <li><a href="https://pixlab.io/art">ASCII Art</a> - Real-Time ASCII Art Rendering C Library.</li>
	  <li><a href="https://unqlite.org">UnQLite</a> - An Embedded, Transactional Key/Value Database Engine.</li>
  </ul>
