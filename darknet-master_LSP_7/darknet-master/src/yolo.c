#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "stdio.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *voc_names[] = { "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hip", "L_Knee", "L_Ankle", "L_Toes", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hip", "R_Knee", "R_Ankle", "R_Toes", "HeadTop", "Neck", "Nose", "Mean_Shoulder", "Mean_Hip", "Front_Back" };
image voc_labels[20];

void train_yolo(char *cfgfile, char *weightfile)
{
    char *train_images = "/mnt/data/yfchen_data/dataset/LSP/lsp_dataset_extended_training/list/LSP_train_cat.txt"; //
    char *backup_directory = "/mnt/data/yfchen_data/dataset/weights_backup/44/"; //
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
	// fine-tune the earlier model
	if (net.fine_tune == 1)	*net.seen = 0;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;
	int miss = l.miss;
	int coords = l.coords;
	int numgt = l.numgt;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;
	args.miss = miss;
	args.coords = coords;
	args.numgt = numgt;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;

	FILE *fout;
	char losspath[50];
	strcpy(losspath, backup_directory);
	strcat(losspath, "loss.txt");
	if (*net.seen == 0){ fout = fopen(losspath, "w"); }
	else{ fout = fopen(losspath, "a"); }

    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
		fprintf(fout, "%d: %f, %f avg, %f rate\n", i, loss, avg_loss, get_current_rate(net));
		if (i % 100 == 0){
			fclose(fout);
			fout = fopen(losspath, "a");
		}
        if(i%2500==0 || i == 600){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
		//printf("before free training data\n");
        free_data(train);
		//printf("after free training data\n");
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
	fclose(fout);
}

/*
void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}*/

//
void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, float thresh, float **probs, keypoint *keypoints, int only_objectness, int coords)
{
	int i, j, n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < side*side; ++i){
		int row = i / side;
		int col = i % side;
		for (n = 0; n < num; ++n){
			int kpt_ind = i*num + n;
			int coord_pred_ind = side*side*(classes + num) + (i*num + n) * coords;
			keypoints[kpt_ind].x = (predictions[coord_pred_ind + 0] + col) / side;
			keypoints[kpt_ind].y = (predictions[coord_pred_ind + 1] + row) / side;
			//keypoints[index].z = predictions[keypoint_index + 2];
			//keypoints[index].v = predictions[keypoint_index + 3];
			int confidence_pred_ind = side*side*classes + i*num + n;
			float scale = predictions[confidence_pred_ind];
			for (j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index + j];
				probs[kpt_ind][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness){
				probs[kpt_ind][0] = scale;
			}
		}
	}
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            //convert_yolo_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            //if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    int nms = 0;
    float iou_thresh = .5;
    float nms_thresh = .5;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
        //convert_yolo_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        //if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms_thresh);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int i,j;
    float nms=2;
    //box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	keypoint *keypoints = calloc(l.side*l.side*l.n, sizeof(keypoint));
	box *boxes = calloc(l.side*l.side*l.n, sizeof(keypoint));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

		/*FILE *fout;
		fout = fopen("confidence_score.txt", "a");
		fprintf(fout, "\n%s\n", filename);
		int index = l.side*l.side*l.classes;
		for (i = 0; i < l.side*l.side*l.n; i++){
			fprintf(fout, "predictions[%d][%d][%d]: %f\n", i/14+1, (i%14)/2+1, i%2+1, predictions[index+i]);
		}
		fclose(fout);*/

		//convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, thresh, probs, keypoints, 0, l.coords);
		char *labelpath = find_replace(input, ".jpg", ".txt");
        if (nms) do_nms_sort(keypoints, probs, l.side*l.side*l.n, l.classes, nms, labelpath);
		for (i = 0; i < l.side*l.side*l.n; i++){
			boxes[i].x = keypoints[i].x;
			boxes[i].y = keypoints[i].y;
			boxes[i].w = 0.07;
			boxes[i].h = 0.07;
		}
		float scale = 512/(float)im.w;
		printf("scale:%f\n", scale);
		image im_resized = resize_image(im, (int)im.w*scale, (int)im.h*scale);
        draw_detections_file(im_resized, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20, input, keypoints, weightfile);
		char *predictionpath = find_replace(input, ".jpg", "");
		save_image(im_resized, predictionpath);

		//scale = 224 / (float)im.w;
		//im_resized = resize_image(im_resized, (int)im.w*scale, (int)im.h*scale);
        show_image(im_resized, "predictions");
		//save_image(im_resized, "predictions");

        //show_image(sized, "resized");

        free_image(im);
        free_image(sized);
		free_image(im_resized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void test_list_yolo(char *cfgfile, char *weightfile, char *test_images, float thresh)
{
	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int i, j, k;
	float nms = 2; // 2: pick the highest prob, 0.14: eliminate the same detections around a little range
	//box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	keypoint *keypoints = calloc(l.side*l.side*l.n, sizeof(keypoint));
	box *boxes = calloc(l.side*l.side*l.n, sizeof(keypoint));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	list *plist = get_paths(test_images);
	int N = plist->size;
	char **paths = (char **)list_to_array(plist);

	//float loss = 0;
	for (i = 0; i < N; i++){

		if (i % 100 == 0){
			printf("%d\n", i);
		}

		strncpy(input, paths[i], 256);		
		image im = load_image_color(input, 0, 0);
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time = clock();
		float *predictions = network_predict(net, X);
		//printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));

		//evaluate loss
		/*char *labelpath = find_replace(input, "crop", "labels");
		int count = 0;
		keypoint_label *keypoints = read_keypoints(labelpath, &count, l.miss, 0);*/


		/*FILE *fout;
		fout = fopen("confidence_score.txt", "a");
		fprintf(fout, "\n%s\n", input);
		int index = l.side*l.side*l.classes;
		for (j = 0; j < l.side*l.side*l.n; j++){
		fprintf(fout, "predictions[%d][%d][%d]: %f\n", j/14+1, (j%14)/2+1, j%2+1, predictions[index+j]);
		}
		fclose(fout);*/

		//convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, thresh, probs, keypoints, 0, l.coords);
		char *labelpath = find_replace(input, "crop/crop_1.5_v5", "predictions/30/30_0.05_pc/labels");//
		//labelpath = find_replace(input, "crop/crop_v5", "predictions/44/labels");//
		labelpath = find_replace(input, "crop/crop_v6", "predictions/44/labels");
		labelpath = find_replace(labelpath, ".jpg", ".txt");
		//printf("\n%s\n", labelpath);
		if (nms) do_nms_sort(keypoints, probs, l.side*l.side*l.n, l.classes-1, nms, labelpath);

		for (j = 0; j < l.side*l.side*l.n; j++){
			boxes[j].x = keypoints[j].x;
			boxes[j].y = keypoints[j].y;
			boxes[j].w = 0.07;
			boxes[j].h = 0.07;
		}
		//draw_detections_file(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 15, input);
		float scale = 512 / (float)im.w;
		//printf("scale:%f\n", scale);
		image im_resized = resize_image(im, (int)im.w*scale, (int)im.h*scale);
		draw_detections_file(im_resized, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, l.classes, input, keypoints, weightfile);
		
		//write the results to txt
		/*FILE *fout;
		fout = fopen(labelpath, "w");
		for (j = 0; j < l.side*l.side*l.n; ++j){
			int class = max_index(probs[j], 20);
			float prob = probs[j][class];
			while (prob > thresh){
				fprintf(fout, "%d %.16f %.16f %.16f %.16f\n", class+1, keypoints[j].x, keypoints[j].y, keypoints[j].z, keypoints[j].v);
				probs[j][class] = 0;
				class = max_index(probs[j], 20);
				prob = probs[j][class];
			}
		}
		fclose(fout);*/
		
		//write all predictions to file
		char *np_path = find_replace(labelpath, "labels", "np");
		//printf("\n%s\n", np_path);
		/*FILE *fout_np;
		fout_np = fopen(np_path, "w");
		int index = 0;
		for (j = 0; j < l.side*l.side; j++){
			for (k = 0; k < l.classes; k++){
				fprintf(fout_np, "%.16f ", predictions[index]);
				index++;
			}
			fprintf(fout_np, "\n");
		}
		for (j = 0; j < l.side*l.side; j++){
			for (k = 0; k < l.n; k++){
				fprintf(fout_np, "%.16f ", predictions[index]);
				index++;
			}
			fprintf(fout_np, "\n");
		}
		for (j = 0; j < l.side*l.side; j++){
			for (k = 0; k < l.n*l.coords; k++){
				fprintf(fout_np, "%.16f ", predictions[index]);
				index++;
			}
			fprintf(fout_np, "\n");
		}
		fclose(fout_np);*/

		char *predictionpath = find_replace(np_path, "np/", "");//
		predictionpath = find_replace(predictionpath, ".txt", "");
		//printf("\n%s\n", predictionpath);
		save_image(im_resized, predictionpath);

		free_image(im);
		free_image(sized);
		free_image(im_resized);
		//getc(stdin);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
	}
}

void test_list_demo_yolo(char *cfgfile, char *weightfile, char *test_images, float thresh)
{
	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int i, j, k;
	float nms = 2; // 2: pick the highest prob, 0.14: eliminate the same detections around a little range
	//box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	keypoint *keypoints = calloc(l.side*l.side*l.n, sizeof(keypoint));
	box *boxes = calloc(l.side*l.side*l.n, sizeof(keypoint));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for (j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	//int index_predict = 0;
	//int count_predict_all = 0;

	while (1){
		list *plist = get_paths(test_images);
		int N = plist->size;
		char **paths = (char **)list_to_array(plist);

		for (i = 0; i < N; i++){
			strncpy(input, paths[i], 256);
			image im = load_image_color(input, 0, 0);
			image sized = resize_image(im, net.w, net.h);
			float *X = sized.data;
			time = clock();
			float *predictions = network_predict(net, X);
			printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));

			//convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
			convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, thresh, probs, keypoints, 0, l.coords);
			char *labelpath = find_replace(input, "crop/crop_1.5_v5", "predictions/30/30_0.05_pc/labels");//
			labelpath = find_replace(input, "crop/crop_v4", "predictions/42/42_0.05_pc/labels");//
			labelpath = find_replace(labelpath, ".jpg", ".txt");
			//printf("\n%s\n", labelpath);
			if (nms) do_nms_sort(keypoints, probs, l.side*l.side*l.n, l.classes - 1, nms, labelpath);

			for (j = 0; j < l.side*l.side*l.n; j++){
				boxes[j].x = keypoints[j].x;
				boxes[j].y = keypoints[j].y;
				boxes[j].w = 0.07;
				boxes[j].h = 0.07;
			}
			float scale = 512 / (float)im.w;
			//printf("scale:%f\n", scale);
			image im_resized = resize_image(im, (int)im.w*scale, (int)im.h*scale);
			draw_detections_file(im_resized, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, l.classes, input, keypoints, weightfile);

			char *predictionpath = find_replace(input, "/home/yfchen/Dropbox/Human_Pose_Estimation/demo/", "/mnt/data/yfchen_data/dataset/LSP/demo/images_26/");//
			predictionpath = find_replace(predictionpath, ".jpg", "");
			save_image(im_resized, predictionpath);

			free_image(im);
			free_image(sized);
			free_image(im_resized);
			//getc(stdin);
		}
		/*count_predict_all++;
		if (count_predict_all % 10 == 0){
			index_predict = 0;
		}
		else index_predict = N;*/

		time = clock();
		//printf("Wait 2 seconds.\n");
		while (sec((clock() - time)) < 5){}
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
	}
}

/*
#ifdef OPENCV
image ipl_to_image(IplImage* src);
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

void demo_swag(char *cfgfile, char *weightfile, float thresh)
{
network net = parse_network_cfg(cfgfile);
if(weightfile){
load_weights(&net, weightfile);
}
detection_layer layer = net.layers[net.n-1];
CvCapture *capture = cvCaptureFromCAM(-1);
set_batch_network(&net, 1);
srand(2222222);
while(1){
IplImage* frame = cvQueryFrame(capture);
image im = ipl_to_image(frame);
cvReleaseImage(&frame);
rgbgr_image(im);

image sized = resize_image(im, net.w, net.h);
float *X = sized.data;
float *predictions = network_predict(net, X);
draw_swag(im, predictions, layer.side, layer.n, "predictions", thresh);
free_image(im);
free_image(sized);
cvWaitKey(10);
}
}
#else
void demo_swag(char *cfgfile, char *weightfile, float thresh){}
#endif
 */

void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index);
#ifndef GPU
void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index)
{
    fprintf(stderr, "Darknet must be compiled with CUDA for YOLO demo.\n");
}
#endif

void run_yolo(int argc, char **argv)
{
    int i;
    for(i = 0; i < 20; ++i){
        char buff[256];
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
	else if(0==strcmp(argv[2], "test_list")) test_list_yolo(cfg, weights, filename, thresh);
	else if(0==strcmp(argv[2], "test_list_demo")) test_list_demo_yolo(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo_yolo(cfg, weights, thresh, cam_index);
}
