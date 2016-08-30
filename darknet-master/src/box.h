#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} box;

//
typedef struct{
	float x, y, z, v;
} keypoint;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f);
keypoint float_to_keypoint(float *f);//
float box_iou(box a, box b);
float box_rmse(box a, box b);
float keypoint_rmse(keypoint a, keypoint b);//
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
//void do_nms(keypoint *keypoints, float **probs, int total, int classes, float thresh);
//void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(keypoint *keypoints, float **probs, int total, int classes, float thresh);
void do_nms_sort_file(keypoint *keypoints, float **probs, int total, int classes, float thresh, char *filename, int miss);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
