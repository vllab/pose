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

	l.miss = miss;
	int coords_decided;
	if (miss == 0){ coords_decided = 4; } //xyzv
	else if (miss == 1){ coords_decided = 2; } //xy
	else if (miss == 2 || miss == 3){ coords_decided = 3; } //xyv xyz
	else{
		printf("miss should be integer 0 to 3.\n");
		exit(0);
	}
	assert(coords == coords_decided);
	l.coords = coords;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.rescore = rescore;
    l.side = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords*numgt+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));
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

		// parameters to record the matchings(each gt chooses which pred)
		int *index_best = calloc(l.numgt, sizeof(int));
		int *index_current = calloc(l.numgt, sizeof(int));
		int *index_picked = calloc(l.n, sizeof(int));
		float *array_dis = calloc(l.numgt*l.n, sizeof(float));
		keypoint *pred = calloc(l.n, sizeof(keypoint));

		int *gt_showup = calloc(l.numgt, sizeof(int));
		int gt_showup_index = 0;
		int *correspondence = malloc(l.classes*sizeof(int));
		keypoint *truth = calloc(l.classes, sizeof(keypoint));
		keypoint *truth_whole_img = calloc(l.classes, sizeof(keypoint));
		float *limb_length = calloc(9, sizeof(float));
		int limb_pick[2*9] = {1,2,2,3,4,5,5,6,8,9,9,10,11,12,12,13,15,16};
		float *additional_penalty = calloc(l.classes, sizeof(float));

		for (b = 0; b < l.batch; ++b){
			int index = b*l.inputs;

			for (i = 0; i < l.classes; i++){
				correspondence[i] = -1;
			}
			for (i = 0; i < 9; i++){
				limb_length[i] = 0;
			}

			for (i = 0; i < locations; ++i) {
				int truth_index = (b*locations + i)*(1 + l.classes + l.coords*l.numgt);
				int numKeypoints = (int)(state.truth[truth_index]+0.5);

				//update the confidence scores of the no-keypoint grid cell
				//(First, it will be updated all the confidence scores.
				// Then, it will be updated by another way later if there are keypoints in this grid cell.)
				for (j = 0; j < l.n; ++j) {
					int p_index = index + locations*l.classes + i*l.n + j;
					l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
					*(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
				}
				if (!numKeypoints){
					continue;
				}

				//front_back label
				if ((i == 0 || i == locations - 1) && (int)(state.truth[truth_index + l.classes] + 0.5) == 1){
					correspondence[l.classes-1] = i;
					numKeypoints--; // front_back label is not a keypoint
					truth[l.classes-1] = float_to_keypoint(state.truth + truth_index + 1 + l.classes + numKeypoints*l.coords);
					if (numKeypoints == 0){
						continue;
					}
				}

				////update classes
				int class_index = index + i*l.classes;
				gt_showup_index = 0;
				//cost_test = *l.cost;
				//printf("gt_showup: ");
				for (j = 0; j < l.classes-1; ++j) { //update front_back label in another way
					l.delta[class_index + j] = l.class_scale * (state.truth[truth_index + 1 + j] - l.output[class_index + j]);
					*(l.cost) += l.class_scale * pow(state.truth[truth_index + 1 + j] - l.output[class_index + j], 2);
					//remember which keypoint is in this grid cell
					if ((int)(state.truth[truth_index + 1 + j]+0.5) == 1){
						gt_showup[gt_showup_index] = j;
						//printf("%d ", gt_showup[gt_showup_index]);
						gt_showup_index++;
					}
				}

				//find best_index
				for (j = 0; j < l.numgt; j++){
					index_best[j] = 0;
					index_current[j] = 0;
				}
				for (j = 0; j < l.n; j++){
					index_picked[j] = 0;
				}
				for (j = 0; j < l.numgt*l.n; j++){
					array_dis[j] = 100;
				}
				int numgt_current = 0;
				float dis_shortest = 100;
				float dis_current = 0;

				// read output of predictions in advance
				for (j = 0; j < l.n; j++){
					int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
					pred[j] = float_to_keypoint(l.output + box_index);
				}

				// assign distance to array_dis[l.numgt][l.n]
				for (u = 0; u < numKeypoints; u++){
					truth[gt_showup[u]] = float_to_keypoint(state.truth + truth_index + 1 + l.classes + u*l.coords);
					for (v = 0; v < l.n; v++){
						array_dis[u*l.n+v] = keypoint_rmse(pred[v], truth[gt_showup[u]]);
					}
				}

				find_smallest_cost(array_dis, &dis_shortest, &dis_current, index_best, index_current, index_picked, numKeypoints, l.n, numgt_current);

				/*for (u = 0; u < numKeypoints; u++){
					for (v = 0; v < l.n; v++){
						printf("%.5f  ", array_dis[u*l.n+v]);
					}
					printf("\n");
				}

				printf("index_best: ");
				for (j = 0; j < numKeypoints; j++){
					printf("%d ", index_best[j]);
				}
				printf("\n");
				fgetc(stdin);*/

				//update confidence score and coordinate
				for (j = 0; j < numKeypoints; j++){
					//confidence score
					int p_index = index + locations*l.classes + i*l.n + index_best[j];
					*(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
					*(l.cost) += l.object_scale * pow(1 - l.output[p_index], 2);
					l.delta[p_index] = l.object_scale * (1. - l.output[p_index]);

					//coordinate
					int keypoint_index = index + locations*(l.classes + l.n) + (i*l.n + index_best[j]) * l.coords;
					int tkeypoint_index = truth_index + 1 + l.classes + j*l.coords;
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

					correspondence[gt_showup[j]] = i*l.n + index_best[j];

					++count;
				}

			}//for (i = 0; i < locations; ++i)

			//// additional penalty
			// transform coordinate from grid cell to whole image
			/*printf("correspendence: ");
			for (i = 0; i < l.classes; i++){
				printf("%d ", correspondence[i]);
			}
			printf("\n");*/
			for (i = 0; i < l.classes-1; i++){
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
				if (correspondence[limb_pick[2*i]-1] != -1 && correspondence[limb_pick[2* i+1]-1] != -1){
					limb_length[i] = keypoint_rmse(truth_whole_img[limb_pick[2*i]-1], truth_whole_img[limb_pick[2*i+1]-1]);
					float penalty = (2 / (1 + limb_length[i])) - 1;
					additional_penalty[limb_pick[2*i]-1] += penalty;
					additional_penalty[limb_pick[2*i+1]-1] += penalty;
				}
				else{
					limb_length[i] = -2;
				}
			}
			/*printf("limb_length: ");
			for (i = 0; i < 9; i++){
				printf("%.2f ", limb_length[i]);
			}
			printf("\n");

			printf("additional_penalty: ");
			for (i = 0; i < l.classes; i++){
				printf("%.2f ", additional_penalty[i]);
			}
			printf("\n");*/

			//penalize
			for (i = 0; i < l.classes; i++){
				if ((int)(additional_penalty[i]+0.5) != 1 && correspondence[i] != -1){
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

			//handle the front_back label
			if (correspondence[l.classes-1] != -1){
				//update the value by new approach
				float f_b_determinant = ((float)correspondence[l.classes - 1]) / (float)locations;
				if ((int)(f_b_determinant + 0.5) == 1){
					for (i = 0; i < l.classes; i++){
						if (correspondence[i] == -1) continue;
						int gridcell_belong = correspondence[i] / l.n;
						l.delta[index + (gridcell_belong+1)*l.classes-1] = l.class_scale * (1.0 - l.output[index + (gridcell_belong+1)*l.classes-1]);
						*(l.cost) += l.class_scale * pow(1.0 - l.output[index + (gridcell_belong+1)*l.classes-1], 2);
					}
				}
				else if ((int)(f_b_determinant + 0.5) == 0){
					for (i = 0; i < l.classes; i++){
						if (correspondence[i] == -1) continue;
						int gridcell_belong = correspondence[i] / l.n;
						l.delta[index + (gridcell_belong+1)*l.classes-1] = l.class_scale * (0.0 - l.output[index + (gridcell_belong+1)*l.classes-1]);
						*(l.cost) += l.class_scale * pow(0.0 - l.output[index + (gridcell_belong+1)*l.classes-1], 2);
					}
				}
			}
			if (l.softmax){
				gradient_array(l.output + index + locations*l.classes, locations*l.n*(1 + l.coords),
					LOGISTIC, l.delta + index + locations*l.classes);
			}
		}//for (b = 0; b < l.batch; ++b)
		free(index_best);
		free(index_current);
		free(index_picked);
		free(array_dis);
		free(pred);
		free(gt_showup);
		free(correspondence);
		free(truth);
		free(truth_whole_img);
		free(limb_length);
		free(additional_penalty);
	}
}

void backward_detection_layer(const detection_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void find_smallest_cost(float *array_dis, float *dis_shortest, float *dis_current, int *index_best, int *index_current, int *index_picked, int numKeypoints, int numpred, int numgt_current){
	int i, j, k;
	float dis_current_temp = 0;

	if(numgt_current < (numKeypoints-1)){
		for(i = 0; i < numpred; i++){
			if(index_picked[i] == 0){
				index_picked[i] = 1;
				index_current[numgt_current] = i;
				*dis_current += array_dis[numgt_current*numpred+i];
				find_smallest_cost(array_dis, dis_shortest, dis_current, index_best, index_current, index_picked, numKeypoints, numpred, numgt_current+1);

				// initial index_picked
				for(j = numgt_current; j < numKeypoints; j++){
					index_picked[index_current[j]] = 0;
				}
				// initial dis_current
				dis_current_temp = 0;
				*dis_current = dis_current_temp;
				for(j = 0; j <numgt_current; j++){
					*dis_current += array_dis[j*numpred+index_current[j]];
				}

			}
		}
	}
	else if(numgt_current == (numKeypoints-1)){ //the last one
		for(i = 0; i < numpred; i++){
			if(index_picked[i] == 0){
				//index_picked[i] = 1;
				*dis_current += array_dis[numgt_current*numpred+i];
				if(*dis_current < *dis_shortest){
					index_current[numgt_current] = i;
					for(k = 0; k < numKeypoints; k++){
						index_best[k] = index_current[k];
					}
					*dis_shortest = *dis_current;
				}

				//index_picked[index_current[numgt_current]] = 0;
				*dis_current -= array_dis[numgt_current*numpred+i];
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
        int num_truth = l.batch*l.side*l.side*(1+l.coords*l.numgt+l.classes);//
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

