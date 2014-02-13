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

#define MAX_ITER 2000

// get real computer time
double clock_()
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

class Gibbs {
	int** examples;
	double*** pUserItemToWord;
	double** pItem;
	double** pUser;

	// model parameters
	int nWords;
	int nUserType;
	int nItemType;
	int nTopics;
	int nUser;
	int nItem;
	int nVar; // 6 variable
	int nSamples;

	// counts for computation
	int*** cUserItemToWord;
	int** cUserItem;

	int** cItem;
	int** cUser;
	int iter;
	double* lls;

	// Dirichlet prior
	double alpha_tw;
	double alpha_uiw;
	double alpha_u;
	double alpha_i;

	std::string dataset;

	// skip the topic here
	enum {USER, ITEM, USER_TYPE, ITEM_TYPE, TOPIC, WORD};

	void normalize(double* param, int n) {
		double Z = 0.0;
		for(int i = 0; i < n; i++) 
			Z += param[i];
		for(int i = 0; i < n; i++)
			param[i] /= Z;
	}

	
	int sampleUserAndItem(int* example) {
		double* p = new double[nUserType*nItemType];
		double Z = 0.0;
		for (int u = 0; u < nUserType; u++) {
			for (int i = 0; i < nItemType; i++) {
				p[u*nItemType+i] = (cUserItemToWord[u][i][example[WORD]] + alpha_uiw) / 
				(cUserItem[u][i] + alpha_uiw * nWords)
				* (cUser[example[USER]][u] + alpha_u) * (cItem[example[ITEM]][i] + alpha_i);
				Z += p[u*nItemType+i];
			}
		}
		for (int i = 0; i < nUserType*nItemType; i++) 
			p[i] /= Z;
		int newUserItem = sampleFromDist(p, nUserType*nItemType);
		delete p;
		return newUserItem;
	}
	
	void sampleFullBlockAndUpdate() {
		iter++;

		#pragma omp parallel for 
	for(int i = 0; i < nSamples; i++) {
			int item, user;
			int* example = examples[i];

			int prev_item = example[ITEM_TYPE];
			int prev_user = example[USER_TYPE];
			int word = example[WORD];

			// update user and item

			#pragma omp critical
			{
				cUserItem[prev_user][prev_item]--;
				cUserItemToWord[prev_user][prev_item][word]--;
				cUser[example[USER]][prev_user]--;
				cItem[example[ITEM]][prev_item]--;
			}
	
			
			int userAndItem = sampleUserAndItem(example);
			item = userAndItem % nItemType;
			user = userAndItem / nItemType;
			example[USER_TYPE] = user;
			example[ITEM_TYPE] = item;

			#pragma omp critical
			{
				cUserItem[user][item]++;
				cUserItemToWord[user][item][word]++;
				cUser[example[USER]][user]++;
				cItem[example[ITEM]][item]++;
			}			
		}
		update();

	}
	
	void update() {
		#pragma omp parallel for        
		for(int i = 0; i < nItem; i++) 
			updateParameter(pItem[i], cItem[i], nItemType, alpha_i);
		
		#pragma omp parallel for    
		for(int i = 0; i < nUser; i++) 
			updateParameter(pUser[i], cUser[i], nUserType, alpha_u);
		
		#pragma omp parallel for collapse(2)
		for (int u = 0; u < nUserType; u++)
			for (int i = 0; i < nItemType; i++)
				updateParameter(pUserItemToWord[u][i], cUserItemToWord[u][i], nWords, alpha_uiw);	
		printf("finish updating param\n");
	}

	double delta;
	void updateParameter(double* param, int* count, int n, double alpha) {
		double Z = 0;
		for (int i = 0; i < n; i++) {
			Z += (alpha + count[i]);
		}
		double temp;
		for (int i = 0; i < n; i++) {
			temp = (alpha + count[i]) / Z;

			if (std::abs(temp - param[i]) > delta) {
				delta = std::abs(temp-param[i]);
				// printf("change %3f\n", delta);
			}
			if (std::abs(temp - param[i]) > 2) {
				printf("%f %f %f\n", temp, param[i], Z);
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

	public: Gibbs(std::string data, int states[], const int n) {
		nSamples = n;
		nVar = 6;
		nUser = states[0]; nItem = states[1];
		nUserType = states[2]; nItemType = states[3];
		nTopics = states[4]; nWords = states[5];
		iter = 0;
		dataset = data;
		std::string filename = dataset+".data";

		// read settings from setting.txt
		std::ifstream setting_file("settings.txt");
		std::string line;
		std::getline(setting_file, line);
		std::stringstream alpha_line(line);
		alpha_line >> alpha_u >> alpha_i >> alpha_uiw >> alpha_tw;

		// TODO: lazy
		alpha_uiw = alpha_tw;

		printf("initilizing models\n");
		printf("=================\n");
		printf("nUser: %d, nItem: %d, nSamples: %d, nWords: %d\n", nUser, nItem, nSamples, nWords);
		printf("nUserType: %d, nItemType: %d, nTopics: %d\n", nUserType, nItemType, nTopics);
		printf("alpha_u: %f, alpha_i: %f, alpha_uit: %f, alpha_tw: %f\n", 
			alpha_u, alpha_i, alpha_uiw, alpha_tw);
		printf("=================\n");

		lls = new double[MAX_ITER+1];
		cUser = new int*[nUser];
		pUser = new double*[nUser];
		#pragma omp parallel for
		for (int i = 0; i < nUser; i++) {
			cUser[i] = new int[nUserType];
			pUser[i] = new double[nUserType];

		}
		cItem = new int*[nItem];
		pItem = new double*[nItem];
		#pragma omp parallel for
		for (int i = 0; i < nItem; i++) {
			cItem[i] = new int[nItemType];
			pItem[i] = new double[nItemType];
		}

		cUserItem = new int*[nUserType];
		cUserItemToWord = new int**[nUserType];
		pUserItemToWord = new double**[nUserType];
		#pragma omp parallel for
		for (int u = 0; u < nUserType; u++) {
			cUserItem[u] = new int[nItemType];
			cUserItemToWord[u] = new int*[nItemType];
			pUserItemToWord[u] = new double*[nItemType];
			for (int i = 0; i < nItemType; i++) {
				cUserItemToWord[u][i] = new int[nWords];
				pUserItemToWord[u][i] = new double[nWords];
			}
		}

		// read in the dataset
		std::ifstream myfile(filename.c_str());

		int count = 0;
		examples = new int*[nSamples];
		#pragma omp parallel for
		for (int i = 0; i < nSamples; i++)
				examples[i] = new int[nVar];

		while (std::getline(myfile, line)) {
				std::stringstream ss(line);
				ss >> examples[count][0] >> examples[count][1] >>
						examples[count][2] >> examples[count][3] >>
						examples[count][4] >> examples[count][5];
				count++;
		}
		myfile.close();

		printf("finish reading files\n");

		
		// filling missing values by random sampling
		for (int i = 0; i < nSamples; i++) {
				examples[i][ITEM_TYPE] = sampleUniform(nItemType);
				examples[i][USER_TYPE] = sampleUniform(nUserType);
		}
		printf("finish imputing\n");

		// erasing counts

		for (int i = 0; i < nUser; i++) 
				for (int j = 0; j < nUserType; j++) 
						cUser[i][j] = 0;
		for (int i = 0; i < nItem; i++) 
				for (int j = 0; j < nItemType; j++)
						cItem[i][j] = 0;
			
		for (int i = 0; i < nUserType; i++) {
			for (int j = 0; j < nItemType; j++) {
				cUserItem[i][j] = 0;
				for (int k = 0; k < nWords; k++)
					cUserItemToWord[i][j][k] = 0;
			}
		}
		printf("finish initializing to 0\n");

		// collect counting statistics
		for (int i = 0; i < nSamples; i++) {
			cUser[examples[i][USER]][examples[i][USER_TYPE]]++;
			cItem[examples[i][ITEM]][examples[i][ITEM_TYPE]]++;

			cUserItemToWord[examples[i][USER_TYPE]][examples[i][ITEM_TYPE]][examples[i][WORD]]++;
			cUserItem[examples[i][USER_TYPE]][examples[i][ITEM_TYPE]]++;
		}
		printf("finish initializing statistics\n");

		// update parameters
		update();
		printf("finish constructor\n");
		}

		void train() {
			double ll = 0.0;
			double t1, t2;
			std::ofstream ll_file;
			ll_file.open((dataset+".ll").c_str());

			while(iter < MAX_ITER) {
				delta = 0;
				t1 = clock_();
				sampleFullBlockAndUpdate();
//				sampleAndUpdate();
				t2 = clock_();
				if (iter % 50 == 0) {
					std::ostringstream convert;   // stream used for the conversion
					convert << iter;
					writeResults(convert.str());
				}
				if (iter % 1 == 0 || iter == 1) {
					ll = computeLogLikelihood();
					printf("iter %d, delta %f, ll %f, time %f\n", iter, delta, ll, t2-t1);
				}
				else
					printf("iter %d, delta %f, ll %f, time %f\n", iter, delta, ll, t2-t1);
				lls[iter] = ll;
				ll_file << ll << "\n";
			}       
			ll_file.close();
		}

		double computeLogLikelihood() {
			double ll = 0.0;

			double t1,t2;
			t1 = clock_();
			#pragma omp parallel for reduction(+:ll)
			for (int n = 0; n < nSamples; n++) {
				double ll_example = 0.0;
				int* example = examples[n];
				for (int u = 0; u < nUserType; u++) {
					for (int i = 0; i < nItemType; i++) {
						ll_example += pUserItemToWord[u][i][example[WORD]]
						* pUser[example[USER]][u] * pItem[example[ITEM]][i];
					}
				}
				if (ll_example < 1e-100) ll += -100; // naively avoid NaN
				else ll += std::log(ll_example);
			}
			t2 = clock_();
			printf("computing likelihood takes %f\n", t2-t1);
			return ll;
		}

		void writeResults(std::string prefix) {
			std::ofstream fUser;
			fUser.open((dataset+"_joint_u_"+prefix).c_str());
			for(int i = 0; i < nUser; i++) {
				for (int w = 0; w < nUserType; w++) {
					fUser << pUser[i][w] << " ";
				}
				fUser << "\n";
			}
			fUser.close();

			std::ofstream fItem;
			fItem.open((dataset+"_joint_i_"+prefix).c_str());
			for(int i = 0; i < nItem; i++) {
				for (int w = 0; w < nItemType; w++) {
					fItem << pItem[i][w] << " ";
				}
				fItem << "\n";
			}
			fItem.close();

			std::ofstream fTUserItemWord;
			fTUserItemWord.open((dataset+"_joint_uiw_"+prefix).c_str());
			for(int i = 0; i < nUserType; i++) {
				for (int j = 0; j < nItemType; j++) {
					for (int t = 0; t < nWords; t++) {
						fTUserItemWord << pUserItemToWord[i][j][t] << " ";
					}
					fTUserItemWord << "\n";
				}
			}
			
			fTUserItemWord.close();
	
		}

		~Gibbs() {
			delete [] lls;

			for (int i = 0; i < nSamples; i++) 
				delete [] examples[i];
			delete [] examples;

			for (int i = 0; i < nUser; i++) {
				delete [] cUser[i];
				delete [] pUser[i];
			}
			delete [] cUser;
			delete [] pUser;
		
			for (int i = 0; i < nItem; i++) {
				delete [] cItem[i];
				delete [] pItem[i];
			}
			delete [] cItem;
			delete [] pItem;

			for (int u = 0; u < nUserType; u++) {
				for (int i = 0; i < nItemType; i++) {
					delete [] pUserItemToWord[u][i];
					delete [] cUserItemToWord[u][i];
				}
				delete [] cUserItem[u];
				delete [] pUserItemToWord[u];
				delete [] cUserItemToWord[u];
			}
			delete [] cUserItem;
			delete [] cUserItemToWord;
			delete [] pUserItemToWord;
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
		int states[] = {24071, 4211, 5, 5, 20, 5000};
		int n = 1965602;
		Gibbs b = Gibbs(dataset, states, n);
		b.train();
	}
	else if (dataset == "foods") {
		int states[] = {256058, 74257, 10, 10, 20, 5000};
		int n = 21006617;
		Gibbs b = Gibbs(dataset, states, n);
		b.train();
	}
	else if (dataset == "yelp") {
		int states[] = {45981, 11537, 5, 5, 10, 5000};
		int n = 13474641;
		Gibbs b = Gibbs(dataset, states, n);
		b.train();
	}
	else if (dataset == "small") {
		int states[] = {1859, 354, 10, 10, 10, 5000};
		int n = 61503;
		Gibbs b = Gibbs(dataset, states, n);
		b.train();
	}
	else {
		printf("not recognized dataset: %s\n", dataset.c_str());
		exit(0);
	}
	return 1;
}