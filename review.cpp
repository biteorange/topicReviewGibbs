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
	int* cTopic;
	int*** cUserItemToTopic;
	int** cUserItem;

	int** cItem;
	int** cUser;
	int iter;

	// Dirichlet prior
	const double alpha_uit;
	const double alpha_u;
	const double alpha_i;
	const double alpha_tw;

	// temp distributions
	double* postItem;
	double* postUser;
	double* postTopic;
	double* lls;

	std::string dataset;

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
			p[i] = (cUserItemToTopic[example[USER_TYPE]][i][example[TOPIC]] + alpha_uit) / (cUserItem[example[USER_TYPE]][i] + nTopics*alpha_uit)
			* (cItem[example[ITEM]][i] + alpha_i);

			// p[i] = pUserItemToTopic[example[USER_TYPE]][i][example[TOPIC]]
			//	* pItem[example[ITEM]][i];
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
			p[i] = (cUserItemToTopic[i][example[ITEM_TYPE]][example[TOPIC]] + alpha_uit) / (cUserItem[i][example[ITEM_TYPE]] + nTopics*alpha_uit)
			* (cUser[example[USER]][i] + alpha_u);

			// p[i] = pUserItemToTopic[i][example[ITEM_TYPE]][example[TOPIC]]
			// 	* pUser[example[USER]][i];
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
			p[i] = (cUserItemToTopic[example[USER_TYPE]][example[ITEM_TYPE]][i] + alpha_uit) *
			(cTopicToWord[i][example[WORD]] + alpha_tw) / (cTopic[i] + alpha_tw * nWords);
// 			p[i] = pUserItemToTopic[example[USER_TYPE]][example[ITEM_TYPE]][i]
//				* pTopicToWord[i][example[WORD]];
			Z += p[i];
		}
		for (int i = 0; i < nTopics; i++)
			p[i] /= Z;
		int newTopic = sampleFromDist(p, nTopics);

		delete [] p;
		return newTopic;
	}

	int sampleUserAndItem(int* example) {
		double* p = new double[nUserType*nItemType];
		double Z = 0.0;
		for (int u = 0; u < nUserType; u++) {
			for (int i = 0; i < nItemType; i++) {
				p[u*nItemType+i] = (cUserItemToTopic[u][i][example[TOPIC]] + alpha_uit) / 
				(cUserItem[u][i] + alpha_uit * nTopics)
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
	int sampleItemAndTopic(int* example) {
		double* p = new double[nItemType*nTopics];
		double Z = 0.0;
		for (int i = 0; i < nItemType; i++) {
			for (int t = 0; t < nTopics; t++) {
				p[i*nTopics+t] = (cUserItemToTopic[example[USER_TYPE]][i][t] + alpha_uit) /
				(cUserItem[example[USER_TYPE]][i] + alpha_uit * nTopics)
				* (cItem[example[ITEM]][i] + alpha_i) 
				* (cTopicToWord[t][example[WORD]] + alpha_tw) / (cTopic[t] + alpha_tw * nWords);

				Z += p[i*nTopics+t];
			}
		}
		for (int i = 0; i < nItemType*nTopics; i++) {
			p[i] /= Z;
		}
		int newItemTopic = sampleFromDist(p, nItemType*nTopics);
		delete p;
		return newItemTopic;
	}

	int sampleUserAndTopic(int* example) {
		double* p = new double[nUserType*nTopics];
		double Z = 0.0;
		for (int u = 0; u < nUserType; u++) {
			for (int t = 0; t < nTopics; t++) {
				p[u*nTopics+t] = (cUserItemToTopic[u][example[ITEM_TYPE]][t] + alpha_uit) /
				(cUserItem[u][example[ITEM_TYPE]] + alpha_uit * nTopics)
				* (cUser[example[USER]][u] + alpha_u) 
				* (cTopicToWord[t][example[WORD]] + alpha_tw) / (cTopic[t] + alpha_tw * nWords);

				Z += p[u*nTopics+t];
			}
		}
		for (int i = 0; i < nUserType*nTopics; i++) {
			p[i] /= Z;
		}
		int newUserTopic = sampleFromDist(p, nUserType*nTopics);
		delete p;
		return newUserTopic;
	}
	void sampleFullBlockAndUpdate() {
		iter++;

		#pragma omp parallel for 
		for(int i = 0; i < nSamples; i++) {
			int item, user, topic;
			int* example = examples[i];

			int prev_item = example[ITEM_TYPE];
			int prev_user = example[USER_TYPE];
			int prev_topic = example[TOPIC];

			// update user and topic
			cUserItemToTopic[prev_user][prev_item][prev_topic]--;
			cUserItem[prev_user][prev_item]--;
			cUser[example[USER]][prev_user]--;
			cTopicToWord[prev_topic][example[WORD]]--;
			cTopic[prev_topic]--;

			int userAndTopic = sampleUserAndTopic(example);
			user = userAndTopic / nTopics;
			topic = userAndTopic % nTopics;

			example[TOPIC] = topic;
			example[USER_TYPE] = user;

			cUserItemToTopic[user][prev_item][topic]++;
			cUserItem[user][prev_item]++;
			cUser[example[USER]][user]++;
			cTopicToWord[topic][example[WORD]]++;
			cTopic[topic]++;

			// update item and topic
			cUserItemToTopic[user][prev_item][topic]--;
			cUserItem[user][prev_item]--;
			cItem[example[ITEM]][prev_item]--;
			cTopicToWord[topic][example[WORD]]--;
			cTopic[topic]--;

			int itemAndTopic = sampleItemAndTopic(example);
			item = itemAndTopic / nTopics;
			topic = itemAndTopic % nTopics;
			example[TOPIC] = topic;
			example[ITEM_TYPE] = item; // will be overwritten

			cUserItemToTopic[user][item][topic]++;
			cUserItem[user][item]++;
			cItem[example[ITEM]][item]++;
			cTopicToWord[topic][example[WORD]]++;
			cTopic[topic]++;

			// update user and item

			cUserItem[user][item]--;
			cUserItemToTopic[user][item][topic]--;
			cUser[example[USER]][user]--;
			cItem[example[ITEM]][item]--;
			int userAndItem = sampleUserAndItem(example);
			item = userAndItem % nItemType;
			user = userAndItem / nItemType;
			example[USER_TYPE] = user;
			example[ITEM_TYPE] = item;
			cUserItem[user][item]++;
			cUserItemToTopic[user][item][topic]++;
			cUser[example[USER]][user]++;
			cItem[example[ITEM]][item]++;
		}
		update();

	}

	void sampleBlockAndUpdate() {
		iter++;

		#pragma omp parallel for
		for(int i = 0; i < nSamples; i++) {
			int item, user, topic;
			int* example = examples[i];

			int prev_item = example[ITEM_TYPE];
			int prev_user = example[USER_TYPE];
			int prev_topic = example[TOPIC];

			cTopic[prev_topic]--;
			cUserItemToTopic[prev_user][prev_item][prev_topic]--;
			cTopicToWord[prev_topic][example[WORD]]--;
			topic = sampleTopic(example);
			example[TOPIC] = topic;
			cUserItemToTopic[prev_user][prev_item][topic]++;
			cTopicToWord[topic][example[WORD]]++;
			cTopic[topic]++;
			// printf("finish sampling topic %d -> %d\n", prev_topic, topic);

			cUserItem[prev_user][prev_item]--;
			cUserItemToTopic[prev_user][prev_item][topic]--;
			cUser[example[USER]][prev_user]--;
			cItem[example[ITEM]][prev_item]--;
			int userAndItem = sampleUserAndItem(example);
			item = userAndItem % nItemType;
			user = userAndItem / nItemType;
			example[USER_TYPE] = user;
			example[ITEM_TYPE] = item;
			cUserItem[user][item]++;
			cUserItemToTopic[user][item][topic]++;
			cUser[example[USER]][user]++;
			cItem[example[ITEM]][item]++;
			// printf("finish sampling item %d -> %d\n", prev_item, item);

		}
		update();
	}

	void sampleAndUpdate() {
		iter++;

		#pragma omp parallel for
		for(int i = 0; i < nSamples; i++) {
			int item, user, topic;
			int* example = examples[i];

			int prev_item = example[ITEM_TYPE];
			int prev_user = example[USER_TYPE];
			int prev_topic = example[TOPIC];

			cTopic[prev_topic]--;
			cUserItemToTopic[prev_user][prev_item][prev_topic]--;
			cTopicToWord[prev_topic][example[WORD]]--;
			topic = sampleTopic(example);
			example[TOPIC] = topic;
			cUserItemToTopic[prev_user][prev_item][topic]++;
			cTopicToWord[topic][example[WORD]]++;
			cTopic[topic]++;
			// printf("finish sampling topic %d -> %d\n", prev_topic, topic);

			cUser[example[USER]][prev_user]--;
			cUserItem[prev_user][prev_item]--;
			cUserItemToTopic[prev_user][prev_item][topic]--;
			user = sampleUser(example);
			example[USER_TYPE] = user;
			cUser[example[USER]][user]++;
			cUserItem[user][prev_item]++;
			cUserItemToTopic[user][prev_item][topic]++;
			// printf("finish sampling user %d -> %d\n", prev_user, user);

			cItem[example[ITEM]][prev_item]--;
			cUserItem[user][prev_item]--;
			cUserItemToTopic[user][prev_item][topic]--;
			item = sampleItem(example);
			example[ITEM_TYPE] = item;
			cItem[example[ITEM]][item]++;
			cUserItem[user][item]++;
			cUserItemToTopic[user][item][topic]++;
			// printf("finish sampling item %d -> %d\n", prev_item, item);

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
		
		#pragma omp parallel for
		for(int i = 0; i < nTopics; i++) 
			updateParameter(pTopicToWord[i], cTopicToWord[i], nWords, alpha_tw);
		
		#pragma omp parallel for collapse(2)
		for (int u = 0; u < nUserType; u++)
			for (int i = 0; i < nItemType; i++)
				updateParameter(pUserItemToTopic[u][i], cUserItemToTopic[u][i], nTopics, alpha_uit);
		
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


	public: Gibbs(std::string data, int states[], const int n) : 
	alpha_u(1), alpha_i(1), alpha_uit(1), alpha_tw(1) {
		nSamples = n;
		nVar = 6;
		nUser = states[0]; nItem = states[1];
		nUserType = states[2]; nItemType = states[3];
		nTopics = states[4]; nWords = states[5];
		iter = 0;
		dataset = data;
		std::string filename = dataset+".data";

		printf("initilizing models\n");
		printf("=================\n");
		printf("nUser: %d, nItem: %d, nSamples: %d\n", nUser, nItem, nSamples);
		printf("nUserType: %d, nItemType: %d, nTopics: %d\n", nUserType, nItemType, nTopics);
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
		cUserItemToTopic = new int**[nUserType];
		pUserItemToTopic = new double**[nUserType];
		#pragma omp parallel for
		for (int u = 0; u < nUserType; u++) {
			cUserItem[u] = new int[nItemType];
			cUserItemToTopic[u] = new int*[nItemType];
			pUserItemToTopic[u] = new double*[nItemType];
			for (int i = 0; i < nItemType; i++) {
				cUserItemToTopic[u][i] = new int[nTopics];
				pUserItemToTopic[u][i] = new double[nTopics];
			}
		}

		cTopic = new int[nTopics];
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
        for (int i = 0; i < nTopics; i++) {
        	cTopic[i] = 0;
        	for (int j = 0; j < nWords; j++) 
        		cTopicToWord[i][j] = 0;
        }
        	
        for (int i = 0; i < nUserType; i++) {
        	for (int j = 0; j < nItemType; j++) {
        		cUserItem[i][j] = 0;
        		for (int k = 0; k < nTopics; k++)
        			cUserItemToTopic[i][j][k] = 0;
        	}
        }
        printf("finish initializing to 0\n");

        // collect counting statistics
        for (int i = 0; i < nSamples; i++) {
        	cUser[examples[i][USER]][examples[i][USER_TYPE]]++;
        	cItem[examples[i][ITEM]][examples[i][ITEM_TYPE]]++;
        	cTopicToWord[examples[i][TOPIC]][examples[i][WORD]]++;
        	cTopic[examples[i][TOPIC]]++;
        	cUserItemToTopic[examples[i][USER_TYPE]][examples[i][ITEM_TYPE]][examples[i][TOPIC]]++;
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
            while(iter < MAX_ITER) {
            	delta = 0;
            	t1 = clock_();
           	sampleFullBlockAndUpdate();
		//           	sampleBlockAndUpdate();
//            	sampleAndUpdate();
                t2 = clock_();
                if (iter % 500 == 0) {
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
        		for (int u = 0; u < nUserType; u++) {
        			for (int i = 0; i < nItemType; i++) {
        				for (int t = 0; t < nTopics; t++) {
        					ll_example += pTopicToWord[t][example[WORD]] * pUserItemToTopic[u][i][t]
        					* pUser[example[USER]][u] * pItem[example[ITEM]][i];
        				}
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
        	fUser.open((dataset+"_u_"+prefix).c_str());
        	for(int i = 0; i < nUser; i++) {
        		for (int w = 0; w < nUserType; w++) {
        			fUser << pUser[i][w] << " ";
        		}
        		fUser << "\n";
        	}
        	fUser.close();

        	std::ofstream fItem;
        	fItem.open((dataset+"_i_"+prefix).c_str());
        	for(int i = 0; i < nItem; i++) {
        		for (int w = 0; w < nItemType; w++) {
        			fItem << pItem[i][w] << " ";
        		}
        		fItem << "\n";
        	}
        	fItem.close();

        	std::ofstream fTUserItemTopic;
        	fTUserItemTopic.open((dataset+"_uit_"+prefix).c_str());
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

        	delete [] cTopic;
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
        		delete [] cUserItem[u];
        		delete [] pUserItemToTopic[u];
        		delete [] cUserItemToTopic[u];
        	}
        	delete [] cUserItem;
        	delete [] cUserItemToTopic;
        	delete [] pUserItemToTopic;

        	delete [] postTopic;
        	delete [] postItem;
        	delete [] postUser;
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
		int states[] = {24071, 4211, 10, 10, 20, 5000};
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



