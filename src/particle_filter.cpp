/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void Particle::prediction(double delta_t, double noise[], double velocity, double yaw_rate) {

	//cout << id << "Before: " << x << "," << y << "," << theta << endl;

	if (yaw_rate == 0) {
		x += delta_t * velocity * sin(theta);
		y += delta_t * velocity * cos(theta);
	} else {
		double theta_new = theta + yaw_rate * delta_t;
		x += (sin(theta_new) - sin(theta)) * velocity / yaw_rate;
		y += (cos(theta) - cos(theta_new)) * velocity / yaw_rate;
		theta = theta_new;
	}

	//cout << id << "Before noise: " << x << "," << y << "," << theta << endl;

	x += noise[0];
	y += noise[1];
	theta += noise[2];

	//cout << id << "After: " << x << "," << y << "," << theta << endl;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized) return;

	// Create normal distributions around known coordinates
	dist_x = normal_distribution<double>(0, std[0]);
	dist_y = normal_distribution<double>(0, std[1]);
	dist_theta = normal_distribution<double>(0, std[2]);

	num_particles = 1000;

	for (int i = 0; i < num_particles; i++) {
		double sample_x = x + dist_x(gen);
		double sample_y = y + dist_y(gen);
		double sample_theta = theta + dist_theta(gen);

		Particle particle = Particle(i, sample_x, sample_y, sample_theta);
		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//cout << "Predict: " << delta_t << ", " << velocity << ", " << yaw_rate << endl;

	for (int i = 0; i < num_particles; i++) {
		double noise[] = { dist_x(gen), dist_y(gen), dist_theta(gen) };
		particles[i].prediction(delta_t, noise, velocity, yaw_rate);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double gauss_norm = (1. / (2 * M_PI * sig_x * sig_y));

	for (Particle& particle : particles) {

		particle.weight = 1;

		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		// Go through all the landmarks within range
		for (auto landmark : map_landmarks.landmark_list) {
			if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
				// Go through all observations and find the closest
				double min_distance = numeric_limits<double>::max();
				double closest_x, closest_y;
				for (auto observation : observations) {
					// Transform from car to map
					double x_t = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
					double y_t = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
					double distance = dist(x_t, y_t, landmark.x_f, landmark.y_f);
					if (distance < min_distance) {
						min_distance = distance;
						closest_x = x_t;
						closest_y = y_t;
					}
				}
				if (min_distance < numeric_limits<double>::max()) {
					double dist_x = closest_x - landmark.x_f;
					double dist_y = closest_y - landmark.y_f;
					double exponent = (dist_x * dist_x) / (2 * sig_x * sig_x) + (dist_y * dist_y) / (2 * sig_y * sig_y);
					double weight = gauss_norm * exp(-exponent);

					//cout << "Particle " << particle.id << " +weight " << weight << endl;

					particle.weight *= weight;

					associations.push_back(landmark.id_i);
					sense_x.push_back(closest_x);
					sense_y.push_back(closest_y);
				}
			}
			SetAssociations(particle, associations, sense_x, sense_y);
		}
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//cout << "Reshuffle" << endl;

	// Construct weights
	vector<double> weights;
	//cout << "Weights:";
	for (auto particle : particles) {
		//cout << " " << particle.id << "/" << particle.weight;
		weights.push_back(particle.weight);
	}
	//cout << endl;

	discrete_distribution<int> dist(weights.begin(), weights.end());
	vector<Particle> new_particles;
	for (int i = 0; i < particles.size(); i++) {
		int index = dist(gen);
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
