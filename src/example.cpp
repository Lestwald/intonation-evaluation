#include <iostream>
#include <vector>

#include "intonation-evaluation.hpp"

int main(int argc, char* argv[]) {
	std::vector<Score> score = evaluate_intonation("../example_audio/t04.wav",
												   "../example_audio/t04s04.wav",
												   "would you like to join me for dinner");
	for (int i = 0; i < score.size(); i++) {
		std::cout << score[i].segment << " " << score[i].score << std::endl;
	}

	return 0;
}