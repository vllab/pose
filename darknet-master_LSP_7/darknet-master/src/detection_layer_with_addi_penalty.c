#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore, int miss, int numgt)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));
	l.miss = miss;
	l.numgt = numgt;
#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

/*
void forward_detection_layer(const detection_layer l, network_state state)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    int b;
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax_array(l.output + index + offset, l.classes,
                        l.output + index + offset);
            }
            int offset = locations*l.classes;
            activate_array(l.output + index + offset, locations*l.n*(1+l.coords), LOGISTIC);
        }
    }
    if(state.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = state.truth[truth_index];
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                if (!is_obj){
                    continue;
                }

                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    l.delta[class_index+j] = l.class_scale * (state.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(state.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(state.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    avg_allcat += l.output[class_index+j];
                }

                box truth = float_to_box(state.truth + truth_index + 1 + l.classes);
                truth.x /= l.side;
                truth.y /= l.side;

                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                l.delta[box_index+0] = l.coord_scale*(state.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(state.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(state.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(state.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(state.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(state.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }
            if(l.softmax){
                gradient_array(l.output + index + locations*l.classes, locations*l.n*(1+l.coords), 
                        LOGISTIC, l.delta + index + locations*l.classes);
            }
        }
        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
    }
}
*/

void forward_detection_layer(const detection_layer l, network_state state)
{
	int locations = l.side*l.side;
	int i, j, u, v;
	memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
	int b;
	if (l.softmax){
		for (b = 0; b < l.batch; ++b){
			int index = b*l.inputs;
			for (i = 0; i < locations; ++i) {
				int offset = i*l.classes;
				softmax_array(l.output + index + offset, l.classes,
					l.output + index + offset);
			}
			int offset = locations*l.classes;
			activate_array(l.output + index + offset, locations*l.n*(1 + l.coords), LOGISTIC);
		}
	}
	if (state.train){
		//float avg_iou = 0;
		//float avg_cat = 0;
		//float avg_allcat = 0;
		//float avg_obj = 0;
		//float avg_anyobj = 0;
		int count = 0;
		*(l.cost) = 0;
		int size = l.inputs * l.batch;
		memset(l.delta, 0, size * sizeof(float));
		//float cost_test = 0;

		int *index_best = calloc(l.numgt, sizeof(int));
		int *index_current = calloc(l.numgt, sizeof(int));
		int *index_picked = calloc(l.n, sizeof(int));
		double **array_dis, *array_array;
		array_dis = (double **)calloc(l.numgt, sizeof(double *));
		array_array = (double *)calloc(l.numgt*l.n, sizeof(double));
		for (j = 0; j < l.numgt; j++){
			array_dis[j] = &array_array[j*l.n];
		}
		keypoint *pred = calloc(l.n, sizeof(keypoint));

		// parameters to memorize each gt chooses which pred
		int *gt_showup = calloc(l.numgt, sizeof(int));
		int gt_showup_index = 0;
		int *correspondence = malloc(l.classes*sizeof(int));
		keypoint *truth = calloc(l.classes, sizeof(keypoint));
		keypoint *truth_whole_img = calloc(l.classes, sizeof(keypoint));
		double *limb_length = calloc(9, sizeof(double));
		int limb_pick[2*9] = {1,2,2,3,4,5,5,6,8,9,9,10,11,12,12,13,15,16};
		double *additional_penalty = calloc(l.classes, sizeof(double));

		for (b = 0; b < l.batch; ++b){
			printf("batch %d\n", b);
			int index = b*l.inputs;

			for (i = 0; i < l.classes; i++){
				correspondence[i] = -1;
			}
			//memset(truth, 0, l.classes*sizeof(keypoint));
			for (i = 0; i < 9; i++){
				limb_length[i] = 0;
			}
			//memset(limb_length, 0, 9 * sizeof(double));

			for (i = 0; i < locations; ++i) {
				int truth_index = (b*locations + i)*(1 + l.classes + l.coords*l.numgt);
				int numKeypoints = (int)state.truth[truth_index];
				//cost_test = *l.cost;
				for (j = 0; j < l.n; ++j) {
					int p_index = index + locations*l.classes + i*l.n + j;
					l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
					*(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
					/*if (fabs(l.output[p_index]) > 10){
						printf("confidence:l.output[%d][%d]=%f\n", i, j, l.output[p_index]);
					}*/
					//avg_anyobj += l.output[p_index];
				}
				//printf("cost(noobj): %f\n", *l.cost);
				/*if (fabs(cost_test - (*l.cost)) > 10){
					printf("%d,%d:noobj:%f\n", b, i, fabs(cost_test - (*l.cost)));
				}*/
				if (!numKeypoints){
					continue;
				}
				else if (numKeypoints != 1 && numKeypoints != 2 && numKeypoints != 3){
					printf("numKeypoints = %d\n", numKeypoints);
					for (j = 0; j <= (1 + l.classes + l.coords*l.numgt); j++){
						printf("%.2f ", state.truth[truth_index + j]);
					}
					printf("\n");
					continue;
				}


				int class_index = index + i*l.classes;
				gt_showup_index = 0;
				//cost_test = *l.cost;
				printf("gt_showup: ");
				for (j = 0; j < l.classes; ++j) {
					l.delta[class_index + j] = l.class_scale * (state.truth[truth_index + 1 + j] - l.output[class_index + j]);
					*(l.cost) += l.class_scale * pow(state.truth[truth_index + 1 + j] - l.output[class_index + j], 2);
					if ((int)state.truth[truth_index + 1 + j] == 1){
						gt_showup[gt_showup_index] = j;
						printf("%d ", gt_showup[gt_showup_index]);
						gt_showup_index++;
					}
					/*if (fabs(l.output[class_index + j]) > 10){
						printf("class:l.output[%d][%d]=%f, state.truth=%f\n", i, j, l.output[class_index + j], state.truth[truth_index + 1 + j]);
					}*/
					//if (state.truth[truth_index + 1 + j]) avg_cat += l.output[class_index + j];
					//avg_allcat += l.output[class_index + j];
				}
				printf("\n");
				//printf("cost(+class): %f\n", *l.cost);
				/*if (fabs(cost_test - (*l.cost)) > 10){
					printf("%d,%d:class:%f\n", b, i, fabs(cost_test - (*l.cost)));
				}*/
				//find best_index

				/*int *index_best = calloc(numKeypoints, sizeof(int));
				int *index_current = calloc(numKeypoints, sizeof(int));
				int *index_picked = calloc(l.n, sizeof(int));*/
				for (j = 0; j < l.numgt; j++){
					index_best[j] = 0;
					index_current[j] = 0;
				}
				for (j = 0; j < l.n; j++){
					index_picked[j] = 0;
				}
				int numgt_current = 0;
				double dis_shortest = 100;
				double dis_current = 0;

				/*double **array_dis;
				array_dis = (double **)calloc(numKeypoints, sizeof(double *));
				for (j = 0; j < numKeypoints; j++){
					array_dis[j] = calloc(l.n, sizeof(double));
				}*/

				/*double **array_dis, *array_array;
				array_dis = (double **)calloc(numKeypoints, sizeof(double *));
				array_array = (double *)calloc(numKeypoints*l.n, sizeof(double));
				for (j = 0; j < numKeypoints; j++){
					array_dis[j] = &array_array[j*l.n];
				}*/

				/*double *pData;
				pData = (double *)calloc(numKeypoints*l.n, sizeof(double));
				for (j = 0; j < numKeypoints; j++, pData += l.n){
					array_dis[j] = pData;
				}*/

				// read output of predictions in advance
				for (j = 0; j < l.n; j++){
					int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
					pred[j] = float_to_keypoint(l.output + box_index);
				}

				// assign distance to array[l.numgt][l.n]
				for (u = 0; u < numKeypoints; u++){
					truth[gt_showup[u]] = float_to_keypoint(state.truth + truth_index + 1 + l.classes + u*l.coords);
					for (v = 0; v < l.n; v++){
						array_dis[u][v] = keypoint_rmse(pred[v], truth[gt_showup[u]]);
					}
				}

				find_smallest_cost(array_dis, &dis_shortest, &dis_current, index_best, index_current, index_picked, numKeypoints, l.n, numgt_current);

				/*for (u = 0; u < numKeypoints; u++){
					for (v = 0; v < l.n; v++){
						printf("%.5f  ", array_dis[u][v]);
					}
					printf("\n");
				}

				printf("index_best: ");
				for (j = 0; j < numKeypoints; j++){
					printf("%d ", index_best[j]);
				}
				printf("\n");
				getchar();*/

				for (j = 0; j < numKeypoints; j++){

					int keypoint_index = index + locations*(l.classes + l.n) + (i*l.n + index_best[j]) * l.coords;
					int tkeypoint_index = truth_index + 1 + l.classes + j*l.coords;

					// confidence
					int p_index = index + locations*l.classes + i*l.n + index_best[j];
					*(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
					*(l.cost) += l.object_scale * pow(1 - l.output[p_index], 2);
					l.delta[p_index] = l.object_scale * (1. - l.output[p_index]);
					//printf("cost(+confidence): %f\n", *l.cost);
					/*if (fabs(cost_test - (*l.cost)) > 10){
					printf("%d,%d:confidence:%f\n", b, i, fabs(cost_test - (*l.cost)));
					}*/

					*(l.cost) += l.coord_scale*pow((state.truth[tkeypoint_index + 0] - l.output[keypoint_index + 0]), 2);
					*(l.cost) += l.coord_scale*pow((state.truth[tkeypoint_index + 1] - l.output[keypoint_index + 1]), 2);

					l.delta[keypoint_index + 0] = l.coord_scale*(state.truth[tkeypoint_index + 0] - l.output[keypoint_index + 0]);
					l.delta[keypoint_index + 1] = l.coord_scale*(state.truth[tkeypoint_index + 1] - l.output[keypoint_index + 1]);

					if (l.miss == 0 || l.miss == 3){
						*(l.cost) += l.coord_scale*pow((state.truth[tkeypoint_index + 2] - l.output[keypoint_index + 2]), 2);
						l.delta[keypoint_index + 2] = l.coord_scale*(state.truth[tkeypoint_index + 2] - l.output[keypoint_index + 2]);
					}
					if (l.miss == 0 || l.miss == 2){
						*(l.cost) += l.coord_scale*pow((state.truth[tkeypoint_index + 3] - l.output[keypoint_index + 3]), 2);
						l.delta[keypoint_index + 3] = l.coord_scale*(state.truth[tkeypoint_index + 3] - l.output[keypoint_index + 3]);
					}

					/*if (fabs(l.output[keypoint_index + 0]) > 10){
					printf("x:l.output[%d]=%f, state.truth=%f\n", i, l.output[keypoint_index + 0], state.truth[tkeypoint_index + 0]);
					printf("y:l.output[%d]=%f, state.truth=%f\n", i, l.output[keypoint_index + 1], state.truth[tkeypoint_index + 1]);
					printf("z:l.output[%d]=%f, state.truth=%f\n", i, l.output[keypoint_index + 2], state.truth[tkeypoint_index + 2]);
					printf("v:l.output[%d]=%f, state.truth=%f\n", i, l.output[keypoint_index + 3], state.truth[tkeypoint_index + 3]);
					}*/
					/*
					if (l.sqrt){
					l.delta[box_index + 2] = l.coord_scale*(sqrt(state.truth[tbox_index + 2]) - l.output[box_index + 2]);
					l.delta[box_index + 3] = l.coord_scale*(sqrt(state.truth[tbox_index + 3]) - l.output[box_index + 3]);
					}*/

					//*(l.cost) += pow(1 - iou, 2);
					//avg_iou += iou;

					correspondence[gt_showup[j]] = i*l.n + index_best[j];
					if (correspondence[gt_showup[j]] < 0 || correspondence[gt_showup[j]] >= (l.n*locations)){
						printf("correspondence failed: %d", correspondence[gt_showup[j]]);
					}

					++count;
				}

				
			}
			/*
			if (l.softmax){
				gradient_array(l.output + index + locations*l.classes, locations*l.n*(1 + l.coords),
					LOGISTIC, l.delta + index + locations*l.classes);
			}*/

			// additional penalty
			// transform coordinate from grid cell to whole image
			printf("correspendence: ");
			for (i = 0; i < l.classes; i++){
				printf("%d ", correspondence[i]);
			}
			printf("\n");
			for(i = 0; i < l.classes; i++){
				if (correspondence[i] == -1){
					continue;
				}
				int gridcell_belong = correspondence[i] / l.n;
				int row = gridcell_belong / l.side;
				int col = gridcell_belong % l.side;
				truth_whole_img[i].x = (truth[i].x + col) / l.side;
				truth_whole_img[i].y = (truth[i].y + row) / l.side;
			}

			// evaluate the length of every limb and penalty
			//int limb_pick[2*9] = {1,2,2,3,4,5,5,6,8,9,9,10,11,12,12,13,15,16};
			for (i = 0; i < l.classes; i++){
				additional_penalty[i] = 1;
			}
			for (i = 0; i < 9; i++){
				if (correspondence[limb_pick[2*i]-1] != -1 &&correspondence[limb_pick[2*i+1]-1] != -1){
					limb_length[i] = keypoint_rmse(truth_whole_img[limb_pick[2*i]-1], truth_whole_img[limb_pick[2*i+1]-1]);
					double penalty = (2 / (1 + limb_length[i])) - 1;
					additional_penalty[limb_pick[2*i]-1] += penalty;
					additional_penalty[limb_pick[2*i+1]-1] += penalty;
				}
				else{
					limb_length[i] = -2;
				}
			}
			printf("limb_length: ");
			for (i = 0; i < 9; i++){
				printf("%.2f ", limb_length[i]);
			}
			printf("\n");

			printf("additional_penalty: ");
			for (i = 0; i < l.classes; i++){
				printf("%.2f ", additional_penalty[i]);
			}
			printf("\n");

			//penalize
			for (i = 0; i < l.classes; i++){
				if (additional_penalty[i] != 1 && correspondence[i] != -1){
					//printf("[%d]: %.2f\n", i, additional_penalty[i]);
					int keypoint_index = index + locations*(l.classes + l.n) + correspondence[i] * l.coords;
					
					*(l.cost) += (additional_penalty[i] - 1)*l.coord_scale*pow((truth[i].x - l.output[keypoint_index + 0]), 2);
					*(l.cost) += (additional_penalty[i] - 1)*l.coord_scale*pow((truth[i].y - l.output[keypoint_index + 1]), 2);

					l.delta[keypoint_index + 0] = additional_penalty[i] * l.coord_scale*(truth[i].x - l.output[keypoint_index + 0]);
					l.delta[keypoint_index + 1] = additional_penalty[i] * l.coord_scale*(truth[i].y - l.output[keypoint_index + 1]);

					if (l.miss == 0 || l.miss == 3){
						*(l.cost) += (additional_penalty[i] - 1)*l.coord_scale*pow((truth[i].z - l.output[keypoint_index + 2]), 2);
						l.delta[keypoint_index + 2] = additional_penalty[i] * l.coord_scale*(truth[i].z - l.output[keypoint_index + 2]);
					}
					if (l.miss == 0 || l.miss == 2){
						*(l.cost) += (additional_penalty[i] - 1)*l.coord_scale*pow((truth[i].v - l.output[keypoint_index + 3]), 2);
						l.delta[keypoint_index + 3] = additional_penalty[i] * l.coord_scale*(truth[i].v - l.output[keypoint_index + 3]);
					}

				}
			}
			


		}
		//printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou / count, avg_cat / count, avg_allcat / (count*l.classes), avg_obj / count, avg_anyobj / (l.batch*locations*l.n), count);
		//printf("Detection Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_cat / count, avg_allcat / (count*l.classes), avg_obj / count, avg_anyobj / (l.batch*locations*l.n), count);
		//printf("count: %d\n", count);
		free(index_best);
		free(index_current);
		free(index_picked);
		free(array_array);
		free(array_dis);
		free(pred);
		free(gt_showup);
		free(correspondence);
		free(truth);
		free(truth_whole_img);
		free(limb_length);
		free(limb_pick);
		free(additional_penalty);
	}
}

void backward_detection_layer(const detection_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void find_smallest_cost(double **array_dis, double *dis_shortest, double *dis_current, int *index_best, int *index_current, int *index_picked, int numKeypoints, int numpred, int numgt_current){
	int i, j, k;

	if(numgt_current < (numKeypoints-1)){
		for(i = 0; i < numpred; i++){
			if(index_picked[i] == 0){
				index_picked[i] = 1;
				index_current[numgt_current] = i;
				*dis_current += array_dis[numgt_current][i];
				find_smallest_cost(array_dis, dis_shortest, dis_current, index_best, index_current, index_picked, numKeypoints, numpred, numgt_current+1);

				// initial index_picked
				for(j = numgt_current; j < numKeypoints; j++){
					index_picked[index_current[j]] = 0;
				}
				// initial dis_current
				*dis_current = 0;
				for(j = 0; j <numgt_current; j++){
					*dis_current += array_dis[j][index_current[j]];
				}

			}
		}
	}
	else if(numgt_current == (numKeypoints-1)){ //the last one
		for(i = 0; i < numpred; i++){
			if(index_picked[i] == 0){
				//index_picked[i] = 1;
				*dis_current += array_dis[numgt_current][i];
				if(*dis_current < *dis_shortest){
					index_current[numgt_current] = i;
					for(k = 0; k < numKeypoints; k++){
						index_best[k] = index_current[k];
					}
					*dis_shortest = *dis_current;
				}

				//index_picked[index_current[numgt_current]] = 0;
				*dis_current -= array_dis[numgt_current][i];
			}
		}
	}
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network_state state)
{
    if(!state.train){
        copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
        return;
    }

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.side*l.side*(1+l.coords+l.classes);
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_detection_layer(l, cpu_state);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_detection_layer_gpu(detection_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

