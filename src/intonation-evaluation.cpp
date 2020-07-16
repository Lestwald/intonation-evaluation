#include "intonation-evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Cholesky"
#include "Eigen/Dense"
#include "essentia/essentiamath.h"
#include "essentia/pool.h"
#define uint64 algorithmfactory_uint64
#include "essentia/algorithmfactory.h"
#undef uint64
#define uint64 pocketsphinx_uint64
#include "pocketsphinx.h"
#undef uint64

std::string make_jsgf(const std::string& utterance) {
	std::string result = "#JSGF V1.0; grammar utterance; public <utterance> =";
	std::istringstream iss(utterance);
	do {
		std::string word;
		iss >> word;
		result += " [sil] ";
		result += word;
	} while (iss);
	result += ";";
	return result;
}

std::vector<Segment> word_segmentaion(const std::string& filename, const std::string& text) {
	ps_decoder_t* ps;
	cmd_ln_t* config;
	FILE* fh;
	char const *hyp, *uttid;
	int16 buf[512];
	int rv;

	config = cmd_ln_init(NULL,
						 ps_args(), TRUE,
						 "-hmm", "../model/en-us/en-us",
						 "-lm", "../model/en-us/en-us.lm.bin",
						 "-dict", "../model/en-us/cmudict-en-us.dict",
						 "-fsgusefiller", "no",
						 "-logfn", "/dev/null",
						 NULL);

	if (config == NULL) {
		std::cerr << "Failed to create config object" << std::endl;
		return {};
	}

	ps = ps_init(config);
	if (ps == NULL) {
		std::cerr << "Failed to create recognizer" << std::endl;
		return {};
	}

	fh = fopen(filename.c_str(), "rb");
	if (fh == NULL) {
		std::cerr << "Failed to open input file" << std::endl;
		return {};
	}

	ps_set_jsgf_string(ps, "search", make_jsgf(text).c_str());
	ps_set_search(ps, "search");
	rv = ps_start_utt(ps);

	while (!feof(fh)) {
		size_t nsamp;
		nsamp = fread(buf, 2, 512, fh);
		rv = ps_process_raw(ps, buf, nsamp, FALSE, FALSE);
	}

	std::vector<Segment> words;
	ps_seg_t* iter = ps_seg_iter(ps);
	while (iter != NULL) {
		int sf, ef;
		ps_seg_frames(iter, &sf, &ef);
		std::string word = std::string(ps_seg_word(iter));
		if (word != "sil" && word != "(NULL)") {
			Segment segment;
			segment.start_time = sf * 0.01;
			segment.end_time = ef * 0.01;
			int pos = word.find('(');
			if (pos != std::string::npos) {
				word = word.substr(0, pos);
			}
			segment.text = word;
			words.push_back(segment);
		}
		iter = ps_seg_next(iter);
	}
	rv = ps_end_utt(ps);
	fclose(fh);
	ps_unset_search(ps, "search");
	ps_free(ps);
	cmd_ln_free_r(config);
	return words;
}

void get_features(const std::string& filename,
				  std::vector<float>& time,
				  std::vector<float>& pitch,
				  std::vector<float>& intensity) {
	int framesize = 2048;
	int hopsize = 128;
	int sample_rate = 16000;

	essentia::init();
	essentia::standard::AlgorithmFactory& factory = essentia::standard::AlgorithmFactory::instance();
	essentia::standard::Algorithm* audioload = factory.create("MonoLoader",
															  "filename", filename,
															  "sampleRate", sample_rate,
															  "downmix", "mix");
	std::vector<float> audio;
	audioload->output("audio").set(audio);
	audioload->compute();

	essentia::standard::Algorithm* frameCutter = factory.create("FrameCutter",
																"frameSize", framesize,
																"hopSize", hopsize,
																"startFromZero", false);

	essentia::standard::Algorithm* window = factory.create("Windowing",
														   "type", "hann",
														   "zeroPadding", 0);

	essentia::standard::Algorithm* spectrum = factory.create("Spectrum",
															 "size", framesize);

	essentia::standard::Algorithm* pitchDetect = factory.create("PitchYinFFT",
																"frameSize", framesize,
																"sampleRate", sample_rate,
																"minFrequency", 80,
																"maxFrequency", 400);

	std::vector<float> frame;
	frameCutter->input("signal").set(audio);
	frameCutter->output("frame").set(frame);

	std::vector<float> windowedframe;
	window->input("frame").set(frame);
	window->output("frame").set(windowedframe);

	std::vector<float> spec;
	spectrum->input("frame").set(windowedframe);
	spectrum->output("spectrum").set(spec);

	float thisPitch = 0.0, thisConf = 0.0;
	float localTime = 0.0;
	std::vector<float> allConf;
	pitchDetect->input("spectrum").set(spec);
	pitchDetect->output("pitch").set(thisPitch);
	pitchDetect->output("pitchConfidence").set(thisConf);

	while (true) {
		frameCutter->compute();
		if (!frame.size())
			break;
		if (essentia::isSilent(frame))
			continue;
		window->compute();
		spectrum->compute();
		pitchDetect->compute();
		pitch.push_back(thisPitch);
		localTime += float(hopsize) / float(sample_rate);
		time.push_back(localTime);
		allConf.push_back(thisConf);
	}

	essentia::standard::Algorithm* audioload_le = factory.create("AudioLoader",
																 "filename", filename);

	essentia::standard::Algorithm* le = factory.create("LoudnessEBUR128",
													   "hopSize", float(hopsize) / float(sample_rate),
													   "sampleRate", sample_rate,
													   "startAtZero", true);

	float sr;
	int ch, br;
	std::string md5, cod;

	std::vector<essentia::StereoSample> audioBuffer;
	audioload_le->output("audio").set(audioBuffer);
	audioload_le->output("sampleRate").set(sr);
	audioload_le->output("numberChannels").set(ch);
	audioload_le->output("md5").set(md5);
	audioload_le->output("bit_rate").set(br);
	audioload_le->output("codec").set(cod);

	std::vector<float> shortTermLoudness;
	float integratedLoudness, loudnessRange;

	le->input("signal").set(audioBuffer);
	le->output("momentaryLoudness").set(intensity);
	le->output("shortTermLoudness").set(shortTermLoudness);

	le->output("integratedLoudness").set(integratedLoudness);
	le->output("loudnessRange").set(loudnessRange);

	audioload_le->compute();
	le->compute();

	delete le;
	delete audioload_le;
	delete pitchDetect;
	delete spectrum;
	delete window;
	delete frameCutter;
	delete audioload;
	essentia::shutdown();

	int max_pitch = *std::max_element(pitch.begin(), pitch.end());
	int min_pitch = *std::min_element(pitch.begin(), pitch.end());
	for (int i = 0; i < pitch.size(); i++) {
		pitch[i] = (pitch[i] - min_pitch) / (max_pitch - min_pitch);
	}

	int max_intensity = *std::max_element(intensity.begin(), intensity.end());
	int min_intensity = *std::min_element(intensity.begin(), intensity.end());
	for (int i = 0; i < intensity.size(); i++) {
		intensity[i] = (intensity[i] - min_intensity) / (max_intensity - min_intensity);
	}
}

void set_segments_features(std::vector<Segment>& segments,
						   const std::vector<float>& time,
						   const std::vector<float>& pitch,
						   const std::vector<float>& intensity) {
	for (int i = 0; i < segments.size(); i++) {
		for (int j = 0; j < time.size(); j++) {
			if (segments[i].start_time <= time[j] && time[j] < segments[i].end_time) {
				segments[i].pitch.push_back(pitch[j]);
				segments[i].intensity.push_back(intensity[j]);
			}
			if (time[j] >= segments[i].end_time) break;
		}
	}
}

bool is_accented(const Segment segment) {
	float coeff[] = {-4.76685688, 4.90904681, -1.48913831, 0.29636509, 0.45219536, -2.99525358, 1.77449659, 4.66016809};
	float duration = segment.end_time - segment.start_time;
	float max_pitch = *std::max_element(segment.pitch.begin(), segment.pitch.end());
	float min_pitch = *std::min_element(segment.pitch.begin(), segment.pitch.end());
	float mean_pitch = std::accumulate(segment.pitch.begin(), segment.pitch.end(), 0.0) / segment.pitch.size();
	float max_intensity = *std::max_element(segment.intensity.begin(), segment.intensity.end());
	float min_intensity = *std::min_element(segment.intensity.begin(), segment.intensity.end());
	float mean_intensity = std::accumulate(segment.intensity.begin(), segment.intensity.end(), 0.0) / segment.intensity.size();
	float z = coeff[0] + coeff[1] * duration + coeff[2] * min_pitch + coeff[3] * max_pitch +
			  coeff[4] * mean_pitch + coeff[5] * min_intensity + coeff[6] * max_intensity + coeff[7] * mean_intensity;
	float f = exp(z) / (exp(z) + 1);
	return f > 0.5;
}

std::vector<Segment> make_foots(const std::vector<Segment>& words) {
	std::vector<Segment> result;
	Segment foot;
	for (int i = 0; i < words.size(); i++) {
		if (words[i].accented) {
			if (foot.start_time != -1.0) {
				result.push_back(foot);
			}
			foot = {words[i].start_time, words[i].end_time, words[i].text, {}, {}};
			foot.pitch = words[i].pitch;
			foot.intensity = words[i].intensity;
		} else {
			if (foot.start_time == -1.0) {
				foot.start_time = words[i].start_time;
				foot.text = words[i].text;
				foot.pitch = words[i].pitch;
				foot.intensity = words[i].intensity;
			} else {
				foot.text = foot.text + " " + words[i].text;
				foot.pitch.insert(foot.pitch.end(), words[i].pitch.begin(), words[i].pitch.end());
				foot.intensity.insert(foot.intensity.end(), words[i].intensity.begin(), words[i].intensity.end());
			}
			foot.end_time = words[i].end_time;
		}
	}
	result.push_back(foot);
	return result;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& x) {
	std::vector<std::vector<float>> result;
	int size = x[0].size();
	for (int i = 1; i < x.size(); i++) {
		if (x[i].size() != size) {
			std::cerr << "Vectors must be the same length" << std::endl;
			return {};
		}
	}
	for (int i = 0; i < size; i++) {
		std::vector<float> tmp;
		for (int j = 0; j < x.size(); j++) {
			tmp.push_back(x[j][i]);
		}
		result.push_back(tmp);
	}
	return result;
}

Eigen::MatrixXf convert_to_eigen_matrix(const std::vector<std::vector<float>>& x) {
	Eigen::MatrixXf matrix(x.size(), x[0].size());
	for (int i = 0; i < x.size(); i++)
		matrix.row(i) = Eigen::VectorXf::Map(&x[i][0], x[0].size());
	return matrix;
}

Eigen::MatrixXf covariance_matrix(const Eigen::MatrixXf& x) {
	Eigen::MatrixXf centered = x.rowwise() - x.colwise().mean();
	Eigen::MatrixXf cov = (centered.adjoint() * centered) / float(x.rows() - 1);
	return cov;
}

float euclidean_distance(const std::vector<float>& x, const std::vector<float>& y) {
	Eigen::VectorXf x_ = Eigen::VectorXf::Map(&x[0], x.size());
	Eigen::VectorXf y_ = Eigen::VectorXf::Map(&y[0], y.size());
	float result = std::sqrt((x_ - y_).transpose() * (x_ - y_));
	return result;
}

float mahalanobis_distance(const std::vector<float>& x, const std::vector<float>& y, const Eigen::MatrixXf& cov) {
	Eigen::VectorXf x_ = Eigen::VectorXf::Map(&x[0], x.size());
	Eigen::VectorXf y_ = Eigen::VectorXf::Map(&y[0], y.size());
	Eigen::MatrixXf cov_inverse = cov.inverse();
	float result = std::sqrt((x_ - y_).transpose() * cov_inverse * (x_ - y_));
	return result;
}

float dtw(const std::vector<std::vector<float>>& x, const std::vector<std::vector<float>>& y, const std::string& distance) {
	std ::vector<std ::vector<float>> matrix(x.size());
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i] = std ::vector<float>(y.size(), 0);
	}
	Eigen::MatrixXf cov = covariance_matrix(convert_to_eigen_matrix(x));
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			if (distance == "euclidean") {
				matrix[i][j] = euclidean_distance(x[i], y[j]);
			} else if (distance == "mahalanobis") {
				matrix[i][j] = mahalanobis_distance(x[i], y[j], cov);
			}
		}
	}
	matrix[0][0] = 0;
	for (int i = 1; i < matrix.size(); i++) {
		matrix[i][0] = matrix[i - 1][0] + matrix[i][0];
	}
	for (int i = 1; i < matrix[0].size(); i++) {
		matrix[0][i] = matrix[0][i - 1] + matrix[0][i];
	}
	for (int i = 1; i < matrix.size(); i++) {
		for (int j = 1; j < matrix[i].size(); j++) {
			matrix[i][j] = matrix[i][j] + std::min(std::min(matrix[i][j - 1], matrix[i - 1][j]), matrix[i - 1][j - 1]);
		}
	}
	int i = matrix.size() - 1;
	int j = matrix[0].size() - 1;
	float dtwSum = matrix[i][j];
	int sumOfConnections = 0;
	while (i > 0 && j > 0) {
		if (i == 0) {
			j--;
		} else if (j == 0) {
			i--;
		} else {
			float minNeighbour = std ::min(
				std ::min(matrix[i][j - 1], matrix[i - 1][j]), matrix[i - 1][j - 1]);
			if (matrix[i - 1][j - 1] - minNeighbour < float(10e-6)) {
				i--;
				j--;
			} else if (matrix[i - 1][j] - minNeighbour < float(10e-6)) {
				i--;
			} else if (matrix[i][j - 1] - minNeighbour < float(10e-6)) {
				j--;
			}
		}
		dtwSum += matrix[i][j];
		sumOfConnections++;
	}
	return dtwSum / sumOfConnections;
}

// audio files should be .wav 16kHz 16bit mono
std::vector<Score> evaluate_intonation(const std::string& reference_filename,
									   const std::string& student_filename,
									   const std::string& text,
									   const std::string& segmentation,
									   const std::string& distance_metric) {
	std::vector<Segment> reference_words = word_segmentaion(reference_filename, text);
	std::vector<Segment> student_words = word_segmentaion(student_filename, text);

	std::vector<float> reference_pitch, student_pitch,
		reference_time, student_time,
		reference_intensity, student_intensity;

	get_features(reference_filename, reference_time, reference_pitch, reference_intensity);
	get_features(student_filename, student_time, student_pitch, student_intensity);

	set_segments_features(reference_words, reference_time, reference_pitch, reference_intensity);
	set_segments_features(student_words, student_time, student_pitch, student_intensity);

	for (int i = 0; i < reference_words.size(); i++) {
		bool accented = is_accented(reference_words[i]);
		reference_words[i].accented = accented;
		student_words[i].accented = accented;
	}

	std::vector<Segment> reference_foots = make_foots(reference_words);
	std::vector<Segment> student_foots = make_foots(student_words);

	std::vector<Score> result;
	float total_score = 0.0;
	int total_length = 0;
	std::vector<Segment> reference_segments, student_segments;

	if (segmentation == "word") {
		reference_segments = reference_words;
		student_segments = student_words;
	} else if (segmentation == "foot") {
		reference_segments = reference_foots;
		student_segments = student_foots;
	} else {
		std::cout << "Wrong argument" << std::endl;
		return {};
	}

	for (int i = 0; i < reference_segments.size(); i++) {
		Score score;
		float distance = dtw(transpose({reference_segments[i].pitch, reference_segments[i].intensity}),
							 transpose({student_segments[i].pitch, student_segments[i].intensity}), distance_metric);
		float max_dist;
		if (distance_metric == "mahalanobis") {
			Eigen::MatrixXf cov = covariance_matrix(convert_to_eigen_matrix(transpose({reference_segments[i].pitch,
																					   reference_segments[i].intensity})));
			max_dist = mahalanobis_distance({1.0, 1.0}, {0.0, 0.0}, cov);
		} else if (distance_metric == "euclidean") {
			max_dist = std::sqrt(2);
		} else {
			std::cout << "Wrong argument" << std::endl;
			return {};
		}
		int length = reference_segments[i].pitch.size();
		float n = length * max_dist;
		score.score = (n - distance) / n;
		score.segment = reference_segments[i].text;
		total_score += score.score * length;
		total_length += length;
		result.push_back(score);
	}
	total_score = total_score / total_length;
	result.push_back(Score{"total", total_score});
	return result;
}
