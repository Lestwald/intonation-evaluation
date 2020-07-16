#ifndef INTONATION_EVALUATION_H
#define INTONATION_EVALUATION_H

#include <vector>
#include <string>

struct Segment {
	float start_time = -1.0;
	float end_time;
	std::string text;
	std::vector<float> pitch;
	std::vector<float> intensity;
	bool accented;
};

struct Score {
	std::string segment;
	float score;
};

std::vector<Score> evaluate_intonation(const std::string& reference_filename,
									   const std::string& student_filename,
									   const std::string& text,
									   const std::string& segmentation = "foot",
									   const std::string& distance_metric = "mahalanobis");

#endif