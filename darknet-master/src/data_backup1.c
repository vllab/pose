#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned int data_seed;

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        int index = rand_r(&data_seed)%m;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char *replaced = find_replace(paths[i], find, replace);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

box_label *read_boxes(char *filename, int *n)
{
    box_label *boxes = calloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = realloc(boxes, (count+1)*sizeof(box_label));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

//
keypoint_label *read_keypoints(char *filename, int *n, int miss, int flip)
{
	keypoint_label *keypoints = calloc(1, sizeof(keypoint_label));
	FILE *file = fopen(filename, "r");
	if (!file) file_error(filename);
	float x, y, z, v;
	int id;
	int count = 0;
	if (miss == 0){ //xyzv
		while (fscanf(file, "%d %f %f %f %f", &id, &x, &y, &z, &v) == 5){
			keypoints = realloc(keypoints, (count + 1)*sizeof(keypoint_label));
			if (flip){
				if (id <= 7){
					id = id + 7;
				}
				else if (id <= 14){
					id = id - 7;
				}
			}
			keypoints[count].id = id - 1;
			keypoints[count].x = x;
			keypoints[count].y = y;
			keypoints[count].z = z;
			keypoints[count].v = v;
			++count;
		}
	}
	else if (miss == 1){ //xy
		while (fscanf(file, "%d %f %f", &id, &x, &y) == 3){
			keypoints = realloc(keypoints, (count + 1)*sizeof(keypoint_label));
			if (flip){
				if (id <= 7){
					id = id + 7;
				}
				else if (id <= 14){
					id = id - 7;
				}
			}
			keypoints[count].id = id - 1;
			keypoints[count].x = x;
			keypoints[count].y = y;
			//keypoints[count].z = z;
			//keypoints[count].v = v;
			++count;
		}
	}
	else if (miss == 2){ //xyv
		while (fscanf(file, "%d %f %f %f", &id, &x, &y, &v) == 4){
			keypoints = realloc(keypoints, (count + 1)*sizeof(keypoint_label));
			if (flip){
				if (id <= 7){
					id = id + 7;
				}
				else if (id <= 14){
					id = id - 7;
				}
			}
			keypoints[count].id = id - 1;
			keypoints[count].x = x;
			keypoints[count].y = y;
			//keypoints[count].z = z;
			keypoints[count].v = v;
			++count;
		}
	}
	else if (miss == 3){ //xyz
		while (fscanf(file, "%d %f %f %f", &id, &x, &y, &z) == 4){
			keypoints = realloc(keypoints, (count + 1)*sizeof(keypoint_label));
			if (flip){
				if (id <= 7){
					id = id + 7;
				}
				else if (id <= 14){
					id = id - 7;
				}
			}
			keypoints[count].id = id - 1;
			keypoints[count].x = x;
			keypoints[count].y = y;
			keypoints[count].z = z;
			//keypoints[count].v = v;
			++count;
		}
	}
	
	fclose(file);
	*n = count;
	return keypoints;
}


void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = rand_r(&data_seed)%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

//
void randomize_keypoints(keypoint_label *b, int n)
{
	int i;
	for (i = 0; i < n; ++i){
		keypoint_label swap = b[i];
		int index = rand_r(&data_seed) % n;
		b[i] = b[index];
		b[index] = swap;
	}
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

//
void correct_keypoints(keypoint_label *keypoints, int n, float dx, float dy, float sx, float sy, int flip, float rad)
{
	int i;
	for (i = 0; i < n; ++i){

		//rotate
		keypoints[i].x = cos(rad)*(keypoints[i].x - 0.5) - sin(rad)*(keypoints[i].y - 0.5) + 0.5;
		keypoints[i].y = sin(rad)*(keypoints[i].x - 0.5) + cos(rad)*(keypoints[i].y - 0.5) + 0.5;

		keypoints[i].x = keypoints[i].x  * sx - dx;
		keypoints[i].y = keypoints[i].y  * sy - dy;

		if (flip){
			if (keypoints[i].id != 19){
				keypoints[i].x = 1. - keypoints[i].x;
			}
		}
		keypoints[i].x = constrain(0, 0.999, keypoints[i].x);
		keypoints[i].y = constrain(0, 0.999, keypoints[i].y);
	}
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy, int miss, int numgt, int coords, float rad, image sized_test)
{
    char *labelpath = find_replace(path, "crop/crop", "labels/labels");

    labelpath = find_replace(labelpath, ".jpg", ".txt");
    labelpath = find_replace(labelpath, ".JPG", ".txt");
    labelpath = find_replace(labelpath, ".JPEG", ".txt");
    int count = 0;
	//need to rearrange the order of the ground truth, or we will make mistakes in detection layer.
    //box_label *boxes = read_boxes(labelpath, &count);
	keypoint_label *keypoints = read_keypoints(labelpath, &count, miss, flip);
    //randomize_boxes(boxes, count);
	randomize_keypoints(keypoints, count);
    //correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	correct_keypoints(keypoints, count, dx, dy, sx, sy, flip, rad);
    float x,y;
    int id;
    int i,j,k;
	int *index_gc_count = calloc(count, sizeof(int));
	int *index_gc_classes = calloc(classes, sizeof(int));
	int *order_gc = calloc(classes, sizeof(int));
	int *which_keypoint = calloc(classes, sizeof(int));

	// draw ground truth bounding box to test the rotation, flip, and jitter
	for (i = 0; i < count; i++){
		float red = 1. - 0.02*i;
		float green = 1.;
		float blue = 1.;
		int x_start = keypoints[i].x*sized_test.w - 1;
		int y_start = keypoints[i].y*sized_test.h - 1;

		for (j = x_start; j <= x_start + 3; j++){
			for (k = y_start; k <= y_start + 3; k++){
				sized_test.data[j + y_start*sized_test.w + 0 * sized_test.w*sized_test.h] = red;
				sized_test.data[j + y_start*sized_test.w + 1 * sized_test.w*sized_test.h] = green;
				sized_test.data[j + y_start*sized_test.w + 2 * sized_test.w*sized_test.h] = blue;
			}
		}
	}

	for (i = 0; i < classes; i++){
		index_gc_classes[i] = -1; // won't put the information of this keypoint into ground truth
		which_keypoint[i] = -1;
	}

	//change the coords for the grid cells
	//memorize the beginning of that grid cell
	//eliminate those keypoints can not be put in ground truth
	for (i = 0; i < count; i++){

		index_gc_count[i] = -1; // won't put the information of this keypoint into ground truth

		x = keypoints[i].x;
		y = keypoints[i].y;
		id = keypoints[i].id;

		int col = (int)(x*num_boxes);
		int row = (int)(y*num_boxes);

		int index = (col + row*num_boxes)*(1 + classes + coords*numgt);
		int num_same_class = 1;
		for (j = i-1; j >= 0; j--){
			if (index_gc_count[j] == index){
				num_same_class += 1;
			}
		}
		if (num_same_class > numgt){
			//printf("!!\n");
			continue;
		}

		index_gc_count[i] = index;
		index_gc_classes[id] = index;
		which_keypoint[id] = i;

		keypoints[i].x = x*num_boxes - col;
		keypoints[i].y = y*num_boxes - row;
		
	}

	/*for (i = 0; i < count; i++){
		printf("%d %.2f %.2f %.2f\n", keypoints[i].id, keypoints[i].x, keypoints[i].y, keypoints[i].v);
	}

	for (i = 0; i < count; i++){
		printf("%d ", index_gc_count[i]);
	}
	printf("\n");

	for (i = 0; i < classes; i++){
		printf("%d ", index_gc_classes[i]);
	}
	printf("\n");

	for (i = 0; i < classes; i++){
		printf("%d ", which_keypoint[i]);
	}
	printf("\n");*/

	//decide the order in a grid cell and fill the ground truth
	for (i = 0; i < classes; i++){
		int index = index_gc_classes[i];
		if (index == -1){
			order_gc[i] = -1;
			continue;
		}
		else order_gc[i] = 0;

		for (j = i - 1; j >= 0; j--){
			if (index_gc_classes[j] == index){
				order_gc[i] = order_gc[j] + 1;
				break;
			}
		}
		if (order_gc[i] >= numgt) continue;

		int nkpt = (int)(truth[index]+0.5);
		/*if (nkpt == order_gc[i]){
			//error
			printf("!!\n");
			fgetc(stdin);
		}*/

		int index_ini = index;
		truth[index] = nkpt + 1.001;
		index++;

		truth[index + i] = 1;
		index += classes;

		index += coords * order_gc[i];
		truth[index++] = keypoints[which_keypoint[i]].x;
		truth[index++] = keypoints[which_keypoint[i]].y;

		if (miss == 0 || miss == 3){
			truth[index] = keypoints[which_keypoint[i]].z;
		}
		if (miss == 0 || miss == 2){
			index++;
			truth[index] = keypoints[which_keypoint[i]].v;
		}

		/*for (j = 0; j < (1 + classes + coords*numgt); j++){
			printf("%.2f ", truth[index_gc_classes[i]+j]);
		}
		printf("\n");*/

		/*if (nkpt != 0 && nkpt != 1 && nkpt != 2){
		printf("nkpt = %d\n", nkpt);
		/*for (j = 1; j <= (1 + classes + coords*numgt); j++){
		printf("%.2f ", truth[index_ini++]);
		}
		printf("\n");
		}*/
	}

	/*for (i = 0; i < classes; i++){
		printf("%d ", order_gc[i]);
	}
	printf("\n");

	for (i = 0; i < classes; i++){
		if (index_gc_classes[i] != -1){
			for (j = 0; j < (1 + classes + coords*numgt); j++){
				printf("%.2f ", truth[index_gc_classes[i]+j]);
			}
			printf("\n");
		}
	}
	fgetc(stdin);*/

    free(keypoints);
	free(index_gc_count);
	free(index_gc_classes);
	free(order_gc);
	free(which_keypoint);
	//printf("\n");
}

void fill_truth_detection(char *path, float *truth, int classes, int num_boxes, int flip, int background, float dx, float dy, float sx, float sy, int miss)
{
    char *labelpath = find_replace(path, "crop", "labels");
    labelpath = find_replace(labelpath, ".jpg", ".txt");
    labelpath = find_replace(labelpath, ".JPEG", ".txt");
    int count = 0;
    //box_label *boxes = read_boxes(labelpath, &count);
	keypoint_label *keypoints = read_keypoints(labelpath, &count, miss, flip);
    //randomize_boxes(boxes, count);
	randomize_keypoints(keypoints, count);
    //float x,y,w,h;
	float x, y, z, v;
    //float left, top, right, bot;
    int id;
    int i;
    if(background){
        for(i = 0; i < num_boxes*num_boxes*(4+classes+background); i += 4+classes+background){
            truth[i] = 1;
        }
    }
    for(i = 0; i < count; ++i){
		x = keypoints[i].x*sx - dx;
		y = keypoints[i].y*sy - dy;
		z = keypoints[i].z;
		v = keypoints[i].v;
		/*
        left  = boxes[i].left  * sx - dx;
        right = boxes[i].right * sx - dx;
        top   = boxes[i].top   * sy - dy;
        bot   = boxes[i].bottom* sy - dy;
		*/
        //id = boxes[i].id;
		id = keypoints[i].id;
		
		/*
        if(flip){
            float swap = left;
            left = 1. - right;
            right = 1. - swap;
        }
		if (flip){
			x = 1. - x;
		}
		x = constrain(0, 1, x);
		y = constrain(0, 1, y);
		
        left =  constrain(0, 1, left);
        right = constrain(0, 1, right);
        top =   constrain(0, 1, top);
        bot =   constrain(0, 1, bot);

        x = (left+right)/2;
        y = (top+bot)/2;
        w = (right - left);
        h = (bot - top);
		*/
        if (x <= 0 || x >= 1 || y <= 0 || y >= 1) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);
		//transfer to grid coordinate
        x = x*num_boxes - col;
        y = y*num_boxes - row;

        /*
           float maxwidth = distance_from_edge(i, num_boxes);
           float maxheight = distance_from_edge(j, num_boxes);
           w = w/maxwidth;
           h = h/maxheight;
         */
		/*
        w = constrain(0, 1, w);
        h = constrain(0, 1, h);
        if (w < .01 || h < .01) continue;
        if(1){
            w = pow(w, 1./2.);
            h = pow(h, 1./2.);
        }
		*/
        int index = (col+row*num_boxes)*(4+classes+background);
        if(truth[index+classes+background+3]) continue; //check visible
        if(background) truth[index++] = 0;
        truth[index+id] = 1;
        index += classes;
        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = z;
        truth[index++] = v;
    }
    free(keypoints);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if(count != 1) printf("Too many or too few labels: %d, %s\n", count, path);
}

matrix load_labels_paths(char **paths, int n, char **labels, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
    }
    return y;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

#define TWO_PI 6.2831853071795864769252866
data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, int miss, int coords, int numgt)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d;
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = size*size*(1+classes+coords*numgt);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

		// randomly rotate images
		float rad = (rand() / ((double)RAND_MAX)) * TWO_PI;
		image rotated = rotate_image(orig, rad);

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = (rand_uniform() * 2*dw - dw);
        int pright = (rand_uniform() * 2*dw - dw);
        int ptop   = (rand_uniform() * 2*dh - dh);
        int pbot   = (rand_uniform() * 2*dh - dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = rand_r(&data_seed)%2;
        image cropped = crop_image(rotated, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;

		image sized_test;
		sized_test.h = sized.h;
		sized_test.w = sized.w;
		sized_test.c = sized.c;
		sized_test.data = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy, miss, numgt, coords, rad, sized_test);

		// draw ground truth bounding box to test the rotation, flip, and jitter


		char *test_image_path = find_replace(random_paths[i], "/mnt/data/yfchen_data/dataset/LSP/lsp_dataset_original/crop/crop_v1/", "/home/yfchen/Dropbox/darknet/darknet-master_LSP_5/darknet-master/test_rotated_image/");
		test_image_path = find_replace(random_paths[i], "/mnt/data/yfchen_data/dataset/LSP/lsp_dataset_extended_training/crop/crop_v1/", "/home/yfchen/Dropbox/darknet/darknet-master_LSP_5/darknet-master/test_rotated_image/");
		test_image_path = find_replace(test_image_path, ".jpg", "");
		test_image_path = find_replace(test_image_path, ".JPG", "");
		test_image_path = find_replace(test_image_path, ".JPEG", "");
		save_image(sized_test, test_image_path);
		free_image(sized_test);
        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d;
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = calloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char *imlabel1 = find_replace(paths[i*2],   "imgs", "labels");
        imlabel1 = find_replace(imlabel1, "jpg", "txt");
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        char *imlabel2 = find_replace(paths[i*2+1], "imgs", "labels");
        imlabel2 = find_replace(imlabel2, "jpg", "txt");
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }
        
        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_detection(int n, char **paths, int m, int classes, int w, int h, int num_boxes, int background, int miss)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d;
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = num_boxes*num_boxes*(4+classes+background);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = ow/10;
        int dh = oh/10;

        int pleft  = (rand_uniform() * 2*dw - dw);
        int pright = (rand_uniform() * 2*dw - dw);
        int ptop   = (rand_uniform() * 2*dh - dh);
        int pbot   = (rand_uniform() * 2*dh - dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        /*
           float angle = rand_uniform()*.1 - .05;
           image rot = rotate_image(orig, angle);
           free_image(orig);
           orig = rot;
         */

        int flip = rand_r(&data_seed)%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        d.X.vals[i] = sized.data;

        fill_truth_detection(random_paths[i], d.y.vals[i], classes, num_boxes, flip, background, dx, dy, 1./sx, 1./sy, miss);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

void *load_thread(void *ptr)
{

#ifdef GPU
    cudaError_t status = cudaSetDevice(gpu_index);
    check_error(status);
#endif

    //printf("Loading data: %d\n", rand_r(&data_seed));
    load_args a = *(struct load_args*)ptr;
    if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.m, a.classes, a.w, a.h, a.num_boxes, a.background, a.miss);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.miss, a.coords, a.numgt);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k);
    if(m) free(paths);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d;
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d;
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d;
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    translate_data_rows(d, -128);
    scale_data_rows(d, 1./128);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand_r(&data_seed)%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}


data load_all_cifar10()
{
    data d;
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar10/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    translate_data_rows(d, -128);
    scale_data_rows(d, 1./128);
    return d;
}

void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand_r(&data_seed)%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

