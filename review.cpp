#include "stdio.h"
#include "stdlib.h"
#include <omp.h>
#include "string.h"
#include <string>
#include <iostream>
#include <fstream>
#include "sstream"
#include <cmath>

class Gibbs {
	int** examples;
	double** pTopicToWord;
	double*** pUserItemToTopic;
	double** pItem;
	double** pUser;

	// model parameters
	int nTopics;
	int nWords;
	int nUserType;
	int nItemType;
	int nUser;
	int nItem;
	int nVar; // 6 variable
	int nSamples;

	// counts for computation
	int** cTopicToWord;
	int*** cUserItemToTopic;
	int** cItem;
	int** cUser;
	int iter;

	// temp distributions
	double* postItem;
	double* postUser;
	double* postTopic;

	enum {USER, ITEM, USER_TYPE, ITEM_TYPE, TOPIC, WORD};

	void normalize(double* param, int n) {
		double Z = 0.0;
		for(int i = 0; i < n; i++) 
			Z += param[i];
		for(int i = 0; i < n; i++)
			param[i] /= Z;
	}

	int sampleItem(int* example) {
		// reuse the postItem
		double* p = new double[nItemType];
		double Z = 0.0;
		for (int i = 0; i < nItemType; i++) {
			p[i] = pUserItemToTopic[example[USER_TYPE]][i][example[TOPIC]]
				* pItem[example[ITEM]][i];
			Z += p[i];
		}
		for (int i = 0; i < nItemType; i++)
			p[i] /= Z;
		int newItemType = sampleFromDist(p, nItemType);
		delete [] p;
		return newItemType;
	}

	int sampleUser(int* example) {
		double* p = new double[nUserType];
		double Z = 0.0;
		// double* p = new double[nUserType];
		for (int i = 0; i < nUserType; i++) {
			p[i] = pUserItemToTopic[i][example[ITEM_TYPE]][example[TOPIC]]
				* pUser[example[USER]][i];
			Z += p[i];
		}
		for (int i = 0; i < nUserType; i++)
			p[i] /= Z;
		int newUserType = sampleFromDist(p, nUserType);
		delete [] p;
		return newUserType;
	}

	int sampleTopic(int* example) {
		double* p = new double[nTopics];
		double Z = 0.0;
		for (int i = 0; i < nTopics; i++) {
			p[i] = pUserItemToTopic[example[USER_TYPE]][example[ITEM_TYPE]][i]
				* pTopicToWord[i][example[WORD]];
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
			int item, user, topic;
			int* example = examples[i];
			item = sampleItem(example);
			user = sampleUser(example);
			topic = sampleTopic(example);

			cUser[example[USER]][example[USER_TYPE]]--;
			cUser[example[USER]][user]++;

			cItem[example[ITEM]][example[ITEM_TYPE]]--;
			cItem[example[ITEM]][item]++;

			cTopicToWord[example[TOPIC]][example[WORD]]--;
			cTopicToWord[topic][example[WORD]]++;

			cUserItemToTopic[example[USER_TYPE]][example[ITEM_TYPE]][example[TOPIC]]--;
			cUserItemToTopic[user][item][topic]++;

			example[USER_TYPE] = user;
			example[ITEM_TYPE] = item;
			example[TOPIC] = topic;
		}
		update();
	}

	void update() {
		#pragma omp parallel for		
		for(int i = 0; i < nItem; i++) 
			updateParameter(pItem[i], cItem[i], nItemType);

		#pragma omp parallel for	
		for(int i = 0; i < nUser; i++) 
			updateParameter(pUser[i], cUser[i], nUserType);

		#pragma omp parallel for
		for(int i = 0; i < nTopics; i++) 
			updateParameter(pTopicToWord[i], cTopicToWord[i], nWords);

		#pragma omp parallel for collapse(2)
		for (int u = 0; u < nUserType; u++)
			for (int i = 0; i < nItemType; i++)
				updateParameter(pUserItemToTopic[u][i], cUserItemToTopic[u][i], nTopics);
	}

	double delta;
	void updateParameter(double* param, int* count, int n) {
		int Z = 0;
		double alpha = 0.5 / n;
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


	public: Gibbs(char* filename, int states[], const int n) {
		nSamples = n;
		nVar = 6;
		nUser = states[0]; nItem = states[1];
		nUserType = states[2]; nItemType = states[3];
		nTopics = states[4]; nWords = states[5];
		iter = 0;
		printf("initilizing models\n");
		printf("=================\n");
		printf("nUser: %d, nItem: %d, nSamples: %d\n", nUser, nItem, nSamples);
		printf("nUserType: %d, nItemType: %d, nTopics: %d\n", nUserType, nItemType, nTopics);
		printf("=================\n");

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

		cUserItemToTopic = new int**[nUserType];
		pUserItemToTopic = new double**[nUserType];
		#pragma omp parallel for
		for (int u = 0; u < nUserType; u++) {
			cUserItemToTopic[u] = new int*[nItemType];
			pUserItemToTopic[u] = new double*[nItemType];
			for (int i = 0; i < nItemType; i++) {
				cUserItemToTopic[u][i] = new int[nTopics];
				pUserItemToTopic[u][i] = new double[nTopics];
			}
		}

		cTopicToWord = new int*[nTopics];
		pTopicToWord = new double*[nTopics];		
		#pragma omp parallel for
		for (int i = 0; i < nTopics; i++) {
			cTopicToWord[i] = new int[nWords];
			pTopicToWord[i] = new double[nWords];
		}
		

		// initialize posteriors
		postItem = new double[nItemType];
		postUser = new double[nUserType];
		postTopic = new double[nTopics];


		// read in the dataset
		std::ifstream myfile(filename);
        std::string line;

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
                examples[i][TOPIC] = sampleUniform(nTopics);
        }
        printf("finish imputing\n");

        // erasing counts
        for (int i = 0; i < nUser; i++) 
                for (int j = 0; j < nUserType; j++) 
                        cUser[i][j] = 0;
        for (int i = 0; i < nItem; i++) 
                for (int j = 0; j < nItemType; j++)
                        cItem[i][j] = 0;
        for (int i = 0; i < nTopics; i++) 
                for (int j = 0; j < nWords; j++) 
                        cTopicToWord[i][j] = 0;
        for (int i = 0; i < nUserType; i++)
                for (int j = 0; j < nItemType; j++)
                        for (int k = 0; k < nTopics; k++)
                                cUserItemToTopic[i][j][k] = 0;
        printf("finish initializing to 0\n");

        // collect counting statistics
        for (int i = 0; i < nSamples; i++) {
                cUser[examples[i][USER]][examples[i][USER_TYPE]]++;
                cItem[examples[i][ITEM]][examples[i][ITEM_TYPE]]++;
                cTopicToWord[examples[i][TOPIC]][examples[i][WORD]]++;
                cUserItemToTopic[examples[i][USER_TYPE]][examples[i][ITEM_TYPE]][examples[i][TOPIC]]++;
        }
        printf("finish initializing statistics\n");

        // update parameters
        update();
        }

    	void train() {
            while(iter < 2000) {
            	delta = 0;
                sampleAndUpdate();
                if (iter % 500 == 0) {
                	std::ostringstream convert;   // stream used for the conversion
                	convert << iter;
                	writeResults(convert.str());
                }
                printf("finish iteration %d, delta %f\n", iter, delta);
            }		
        }

        void writeResults(std::string prefix) {
        	std::ofstream fTopicWord;
        	fTopicWord.open(("tw_"+prefix).c_str());
        	for(int i = 0; i < nTopics; i++) {
        		for (int w = 0; w < nWords; w++) {
        			fTopicWord << pTopicToWord[i][w] << " ";
        		}
        		fTopicWord << "\n";
        	}
        	fTopicWord.close();

        	std::ofstream fUser;
        	fUser.open(("u_"+prefix).c_str());
        	for(int i = 0; i < nUser; i++) {
        		for (int w = 0; w < nUserType; w++) {
        			fUser << pUser[i][w] << " ";
        		}
        		fUser << "\n";
        	}
        	fUser.close();

        	std::ofstream fItem;
        	fItem.open(("i_"+prefix).c_str());
        	for(int i = 0; i < nItem; i++) {
        		for (int w = 0; w < nItemType; w++) {
        			fItem << pItem[i][w] << " ";
        		}
        		fItem << "\n";
        	}
        	fItem.close();

        	std::ofstream fTUserItemTopic;
        	fTUserItemTopic.open(("uit_"+prefix).c_str());
        	for(int i = 0; i < nUserType; i++) {
        		for (int j = 0; j < nItemType; j++) {
        			for (int t = 0; t < nTopics; t++) {
        				fTUserItemTopic << pUserItemToTopic[i][j][t] << " ";
        			}
        			fTUserItemTopic << "\n";
        		}
        	}
        	
        	fTUserItemTopic.close();
    
        }

        ~Gibbs() {
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

        	for (int i = 0; i < nTopics; i++) {
        		delete [] pTopicToWord[i];
        		delete [] cTopicToWord[i];
        	}
        	delete [] cTopicToWord;
        	delete [] pTopicToWord;

        	for (int u = 0; u < nUserType; u++) {
        		for (int i = 0; i < nItemType; i++) {
        			delete [] pUserItemToTopic[u][i];
        			delete [] cUserItemToTopic[u][i];
        		}
        		delete [] pUserItemToTopic[u];
        		delete [] cUserItemToTopic[u];
        	}
        	delete [] cUserItemToTopic;
        	delete [] pUserItemToTopic;

        	delete [] postTopic;
        	delete [] postItem;
        	delete [] postUser;
        }

};

int main() {
	int states[] = {24071, 4211, 10, 10, 20, 5000};
	int n = 1965602;
	Gibbs b = Gibbs("arts.data", states, n);
	b.train();
	return 1;
}



