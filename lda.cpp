#include "stdio.h"
#include "stdlib.h"
#include <omp.h>
#include "string.h"
#include <string>
#include <iostream>
#include <fstream>
#include "sstream"
#include "sys/time.h"
#include <cmath>


// get real computer time
double clock_()
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

class Gibbs {
	int** examples;
	double** pTopicToWord;
	double** pDocToTopic;

	// model parameters
	int nTopics;
	int nDocs;
	int nWords;
	int nVar; // 3 variable
	int nSamples;

	// counts for computation
	int** cTopicToWord;
	int* cTopic;
	int** cDocToTopic;
	int* cDoc;
	int iter;

	// Dirichlet prior
	const double alpha;
	const double beta;

	std::string dataset;

	enum {DOC, TOPIC, WORD};

	void normalize(double* param, int n) {
		double Z = 0.0;
		for(int i = 0; i < n; i++) 
			Z += param[i];
		for(int i = 0; i < n; i++)
			param[i] /= Z;
	}

	int sampleTopic(int* example) {
		double* p = new double[nTopics];
		double Z = 0.0;
		for (int i = 0; i < nTopics; i++) {
			if (cDoc[example[DOC]] == 0) {
				printf("doc %d is empty\n",example[DOC]);
				exit(0);
			}
			if (cTopic[i] == 0) {
				printf("topic %d is empty\n", i);
				exit(0);
			}
			p[i] = (cDocToTopic[example[DOC]][i] + alpha) // / (cDoc[example[DOC]] + alpha*nTopics) * 
 			 * (cTopicToWord[i][example[WORD]] + beta) / (cTopic[i] + beta*nWords);

			// p[i] = pDocToTopic[example[DOC]][i] * pTopicToWord[i][example[WORD]];
			Z += p[i];
		}
		for (int i = 0; i < nTopics; i++)
			p[i] /= Z;
		int newTopic = sampleFromDist(p, nTopics);

		delete [] p;
		return newTopic;
	}

	void sampleAndUpdate() {
		iter++;

		#pragma omp parallel for
		for(int i = 0; i < nSamples; i++) {
			int topic;
			int* example = examples[i];

			cTopic[example[TOPIC]]--;
			cDocToTopic[example[DOC]][example[TOPIC]]--;
			cTopicToWord[example[TOPIC]][example[WORD]]--;

			topic = sampleTopic(example);
			
			cTopic[topic]++;
			cDocToTopic[example[DOC]][topic]++;
			cTopicToWord[topic][example[WORD]]--;

			example[TOPIC] = topic;
		}
		update();
	}


	void update() {
		
		#pragma omp parallel for	
		for(int i = 0; i < nDocs; i++) 
			updateParameter(pDocToTopic[i], cDocToTopic[i], nTopics, alpha);

		#pragma omp parallel for
		for(int i = 0; i < nTopics; i++) 
			updateParameter(pTopicToWord[i], cTopicToWord[i], nWords, beta);
	}

	double delta;
	void updateParameter(double* param, int* count, int n, double alpha) {
		int Z = 0;
		for (int i = 0; i < n; i++) {
			Z += count[i];
		}
		double temp;
		for (int i = 0; i < n; i++) {
		  temp = (alpha + count[i]) / (n*alpha + Z);

			if (std::abs(temp - param[i]) > delta) {
				delta = std::abs(temp-param[i]);
				// printf("change %3f\n", delta);
			}
			if (std::abs(temp - param[i]) > 2) {
				printf("%f %f\n", temp, param[i]);
				exit(0);
			}
			param[i] = temp;
		}
		// printf("\n");
	}

	void randomDist(double* p, int n) {
		for (int i = 0; i < n; i++) {
			p[i] = rand() * 1.0 / (1.0 + RAND_MAX);
		}
		normalize(p, n);
	}

	int sampleUniform(int n) {
		double x = rand() * 1.0 / (1.0 + RAND_MAX);
		int i = 0;
		while(i < n-1) {
			x -= 1.0/n;
			if (x < 0) break;
			i++;
		}
		return i;
	}

	int sampleFromDist(double* p, int n){
        double x = rand() * 1.0 / (1.0 + RAND_MAX);
        // printf("%f sampled number\n", x);
        int i = 0;
        while (i < n-1)
        {
          x -= p[i];
          if (x < 0)
            break;
          i++;
        }
        return i;
	}


	public: Gibbs(std::string data, int states[], const int n) : 
	alpha(0.1), beta(0.01)	{
		nSamples = n;
		nVar = 3;
		nDocs = states[0]; 
		nTopics = states[1]; nWords = states[2];
		iter = 0;
		dataset = data;
		std::string filename = dataset+".em";

		printf("initilizing models\n");
		printf("=================\n");
		printf("nDocs: %d, nTopics: %d, nSamples: %d\n", nDocs, nTopics, nSamples);
		printf("=================\n");

		cDoc = new int[nDocs];
		cTopic = new int[nTopics];

		cDocToTopic = new int*[nDocs];
		pDocToTopic = new double*[nDocs];
		#pragma omp parallel for
		for (int d = 0; d < nDocs; d++) {
			cDocToTopic[d] = new int[nTopics];
			pDocToTopic[d] = new double[nTopics];
		}

		cTopicToWord = new int*[nTopics];
		pTopicToWord = new double*[nTopics];		
		#pragma omp parallel for
		for (int i = 0; i < nTopics; i++) {
			cTopicToWord[i] = new int[nWords];
			pTopicToWord[i] = new double[nWords];
		}

		// read in the dataset
		std::ifstream myfile(filename.c_str());
        std::string line;

        int count = 0;
        examples = new int*[nSamples];
        #pragma omp parallel for
        for (int i = 0; i < nSamples; i++)
                examples[i] = new int[nVar];

        while (std::getline(myfile, line)) {
                std::stringstream ss(line);
                ss >> examples[count][0] >> examples[count][1] >>
                        examples[count][2];
                count++;
        }
        myfile.close();

        printf("finish reading files\n");

        // filling missing values by random sampling
        for (int i = 0; i < nSamples; i++) {
        	examples[i][TOPIC] = sampleUniform(nTopics);
        }
        printf("finish imputing\n");

        // erasing counts
        for (int t = 0; t < nTopics; t++) {
        	cTopic[t] = 0;
        	for (int w = 0; w < nWords; w++)
        		cTopicToWord[t][w] = 0;
        }
        for (int d = 0; d < nDocs; d++) {
        	cDoc[d] = 0;
        	for (int t = 0; t < nTopics; t++)
        		cDocToTopic[d][t] = 0;
        }
        printf("finish initializing to 0\n");

        // collect counting statistics
        for (int i = 0; i < nSamples; i++) {
        	cDoc[examples[i][DOC]]++;
        	cDocToTopic[examples[i][DOC]][examples[i][TOPIC]]++;
        	cTopic[examples[i][TOPIC]]++;
        	cTopicToWord[examples[i][TOPIC]][examples[i][WORD]]++;
        }
        for (int t = 0; t < nTopics; t++) 
        	printf("topic %d, count %d\n", t, cTopic[t]);
        printf("finish initializing statistics\n");

        // update parameters
        update();
        }

    	void train() {
    		double ll = 0.0;
    		double t1, t2;
            while(iter < 2000) {
            	delta = 0;
            	t1 = clock_();
                sampleAndUpdate();
                t2 = clock_();
                if (iter % 500 == 0) {
                	std::ostringstream convert;   // stream used for the conversion
                	convert << iter;
                	writeResults(convert.str());
                }
                ll = computeLogLikelihood();
                printf("iter %d, delta %f, ll %f, time %f\n", iter, delta, ll, t2-t1);
            }		
        }

        double computeLogLikelihood() {
        	double ll = 0.0;

        	double t1,t2;
        	t1 = clock_();
        	#pragma omp parallel for reduction(+:ll)
        	for (int n = 0; n < nSamples; n++) {
        		double ll_example = 0.0;
        		int* example = examples[n];

        		for (int t = 0; t < nTopics; t++) 
        			ll_example += pTopicToWord[t][example[WORD]] * pDocToTopic[example[DOC]][t];

        		ll += std::log(ll_example);
        	}
        	t2 = clock_();
        	printf("computing likelihood takes %f\n", t2-t1);
        	return ll;
        }

        void writeResults(std::string prefix) {
        	std::ofstream fTopicWord;
        	fTopicWord.open((dataset+"_tw_"+prefix).c_str());
        	for(int i = 0; i < nTopics; i++) {
        		for (int w = 0; w < nWords; w++) {
        			fTopicWord << pTopicToWord[i][w] << " ";
        		}
        		fTopicWord << "\n";
        	}
        	fTopicWord.close();

        	std::ofstream fUser;
        	fUser.open((dataset+"_dt_"+prefix).c_str());
        	for(int i = 0; i < nDocs; i++) {
        		for (int w = 0; w < nTopics; w++) {
        			fUser << pDocToTopic[i][w] << " ";
        		}
        		fUser << "\n";
        	}
        	fUser.close();
        }

        void writeFinal() {
        	std::ofstream output;
        	output.open((dataset+".model").c_str());
        	output << nDocs << " " << nTopics << " " << nDocs << "\n";
        	for(int i = 0; i < nTopics; i++) {
        		for (int w = 0; w < nWords; w++) 
        			output << pTopicToWord[i][w] << " ";
        		output << "\n";
        	}
        	for(int i = 0; i < nDocs; i++) {
        		for (int w = 0; w < nTopics; w++) 
        			output << pDocToTopic[i][w] << " ";
        		output << "\n";
        	}
        	output.close();
        }

        void readModel(char* filename) {
        	// TODO: finish reading models for initializing
        }

        ~Gibbs() {
        	for (int i = 0; i < nSamples; i++) {
        		delete [] examples[i];
        	}
        	delete [] examples;

        	for (int d = 0; d < nDocs; d++) {
        		delete [] pDocToTopic[d];
        		delete [] cDocToTopic[d];
        	}
        	delete [] pDocToTopic;
        	delete [] cDocToTopic;

        	delete [] cTopic;
        	delete [] cDoc;

        	for (int i = 0; i < nTopics; i++) {
        		delete [] pTopicToWord[i];
        		delete [] cTopicToWord[i];
        	}
        	delete [] cTopicToWord;
        	delete [] pTopicToWord;

        }

};


int main(int argc, char** argv) {
	if (argc < 2)
	{
		printf("An input file is required\n");
		exit(0);
	}
	
	std::string dataset = std::string(argv[1]);
	if (dataset == "arts") {
		int states[] = {27980, 20, 5000};
		int n = 930376;
		Gibbs b = Gibbs(dataset, states, n);
		b.train();
	}
	else if (dataset == "foods") {
		int states[] = {256058, 74257, 10, 10, 20, 5000};
		int n = 21006617;
		Gibbs b = Gibbs(dataset, states, n);
		b.train();
	}
	else {
		printf("not recognized dataset: %s\n", dataset.c_str());
		exit(0);
	}
	return 1;
}


