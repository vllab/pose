#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "params.h"
#include "layer.h"

typedef layer detection_layer;

detection_layer make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore, int miss, int numgt);
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);
void find_smallest_cost(float *array_dis, float *dis_shortest, float *dis_current, int *index_best, int *index_current, int *index_picked, int numKeypoints, int numpred, int numgt_current);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network_state state);
void backward_detection_layer_gpu(detection_layer l, network_state state);
#endif

#endif
