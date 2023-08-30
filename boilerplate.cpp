/*
* Tiny Dream Integration Boilerplate.
* 
* Compile this file together with the Tiny Dream header (tinydream.hpp) to generate
* the executable. Example of heavy optimized compilation under g++:
*
*  g++ -o tinydream boileplate.cpp -funsafe-math-optimizations -Ofast -flto=auto  -funroll-all-loops -pipe -march=native -std=c++17 -Wall -Wextra `pkg-config --cflags --libs ncnn` -lstdc++ -pthread -Wl -flto -fopt-info-vec-optimized
* 
* To run the program simply type:
* 
* ./tinydream "pyramid, desert, palm trees, river, (landscape), (high quality)"
*
* Under Microsoft Visual Studio (>= 2019), just drop `tinydream.hpp` on your source
* tree and you're done.
*
* Do not forget to link to the current backend tensor library (TINY_DREAM_INFERENCE_ENGINE).
* You will need to Pre-trained Models from https://pixlab.io/tiny-dream#downloads in order
* to start generating images (Stable Diffusion Inference).
* 
* If you have any trouble integrating Tiny-Dream on your project, please submit a support
* ticket at: https://pixlab.io/tiny-dream
*/
/*
* This simple program is a quick introduction on how to embed
* and start experimenting with Tiny Dream (Stable Diffusion inference)
* without having to do a lot of tedious reading and configuration.
*
* Make sure you have the latest release of Tiny-Dream
* plus the Pre-Trained Models from:
* 
*  https://pixlab.io/tiny-dream#downloads
*
* The Tiny Dream C++ documentation is available to consult on:
*  https://pixlab.io/tiny-dream
*  https://github.com/symisc/tiny-dream
*/
#include "tinydream.hpp"
#include <iostream>
/*
* Register a log consumer callback first
* 
* The main task of the supplied callback is to consume log messages
* generated during Stable Diffusion inference.
* Inference may take some time to execute depending on the available
* resources so it make sense to log everything to the terminal or
* text file for example.
* 
* The supplied callback must have the following signature:
*    void(const char *zLogMsg,int msgLen void *pUserData)
* 
* Refer to the setLogCallback() API documentation at: https://pixlab.io/tiny-dream#set-log-callback
* for additional information.
*/
#if defined (_WIN32) || defined(_WIN64) || defined (_MSC_VER)
#include <Windows.h>
#define TD_WIN
#else
/* Assume POSIX compatible */
#include <unistd.h>
#endif
void logCallback(const char* zLogMsg, int msgLen, [[maybe_unused]] void* pCookie)
{
	// All this log consumer callback does, is just redirecting
	// the generated log messages by the inference engine
	// to the default standard output (STDOUT)
#ifdef TD_WIN
	WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), static_cast<LPCVOID>(zLogMsg), static_cast<DWORD>(msgLen), 0, 0);
#else
	write(STDOUT_FILENO, static_cast<const void*>(zLogMsg), static_cast<size_t>(msgLen));
#endif /* __WINT__ */
}
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
	// td.setImageOutputPrefix("tinydream-");
	
	// Optionally, set the directory where you want
	// the generated images to be stored
	//td.setImageOutputPath("/home/photos/");
	
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