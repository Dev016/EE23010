#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function to generate a random sample from a normal distribution
double generateNormalSample(double mean, double variance) {
    double sum = 0.0;
    int num_samples = 100;  // You can adjust the number of samples

    for (int i = 0; i < num_samples; i++) {
        sum += (double)rand() / RAND_MAX;
    }

    double normal_sample = mean + sqrt(variance) * (sum - num_samples / 2.0) / (num_samples / 2.0);

    return normal_sample;
}

int main() {

    // Parameters of the normal distribution
    double mean = 0.0;  // Mean
    double theta = 1;  // Variance = theta^2
	
    // Generate and print random samples from the normal distribution
    
    int numSamples = 1000;  // You can change the number of samples as needed
    double variables[10][numSamples];
    for (int i = 0; i < 10; i++){
    	// Set the seed for the random number generator (you can use any seed)
    	srand(time(NULL));
    	for (int j = 0; j < numSamples; j++) {
        	double sample = generateNormalSample(mean, theta*theta);
        	//Converting X to X^2
        	variables[i][j] = sample*sample;
    	}
    }
    double T[numSamples];
    for(int i = 0; i < numSamples; i++){
    	T[i] = 0;
    	for (int j = 0; j < 10; j++){
    	T[i] += variables[j][i];
    } 		
    }
    //Finding value of C which minimizes E((cT - theta^2)^2)
    double min = 1000;
    double c_min = 0;
    for(double c = 0.0; c < 1; c += 0.01){
    	double sum = 0.0;
    	for(int i = 0; i < numSamples; i++){
    		sum += pow(c*T[i] - theta*theta, 2);	
    	}
    	sum = sum/numSamples;
    	if (sum < min){
    		min = sum;
    		c_min = c;
    	}
    }
    printf("The value of c which minimizes the mean square error is %lf \n", c_min);
    return 0;
}

