#include "od.hpp"
#include <stdio.h>
#include <turbojpeg.h>
#include <cstring>

int main(int argc, char *argv[]) {
  char *filename = argv[3];
  tjhandle tjInstance = NULL;
  if ((tjInstance = tj3Init(TJINIT_DECOMPRESS)) == NULL) {
    fprintf(stderr, "Could not initialize TurboJPEG\n");
    return -1;
  }

  FILE *jpegFile = fopen(filename, "rb");
  if (!jpegFile) {
    fprintf(stderr, "Could not open file %s\n", filename);
    return -1;
  }

  long size;
  if (fseek(jpegFile, 0, SEEK_END) < 0 || ((size = ftell(jpegFile)) < 0) ||
      fseek(jpegFile, 0, SEEK_SET) < 0) {
    fprintf(stderr, "Could not determine file size\n");
    return -1;
  }
  if (size == 0) {
    fprintf(stderr, "File is empty\n");
    return -1;
  }
  size_t jpegSize = size;

  unsigned char *jpegBuf;
  if ((jpegBuf = (unsigned char *)tj3Alloc(jpegSize)) == NULL) {
    fprintf(stderr, "Could not allocate memory\n");
    return -1;
  }

  if (fread(jpegBuf, jpegSize, 1, jpegFile) < 1) {
    fprintf(stderr, "Could not read file\n");
    return -1;
  }

  fclose(jpegFile);
  jpegFile = NULL;

  if (tj3DecompressHeader(tjInstance, jpegBuf, jpegSize) < 0) {
    fprintf(stderr, "Could not read JPEG header\n");
    return -1;
  }

  int width = tj3Get(tjInstance, TJPARAM_JPEGWIDTH);
  int height = tj3Get(tjInstance, TJPARAM_JPEGHEIGHT);
  printf("Image is %dx%d\n", width, height);

  unsigned char *imgBuf;

  if ((imgBuf = (unsigned char *)malloc(sizeof(unsigned char) * width * height *
                                        tjPixelSize[TJPF_RGB])) == NULL) {
    fprintf(stderr, "Could not allocate memory\n");
    return -1;
  }

  if (tj3Decompress8(tjInstance, jpegBuf, jpegSize, imgBuf, 0, TJPF_RGB) < 0) {
    fprintf(stderr, "Could not decode JPEG\n");
    return -1;
  }
  tj3Free(jpegBuf);
  tj3Destroy(tjInstance);

  printf("Image loaded\n");

  struct yolo_image img = {width, height, 3};
  img.data = std::vector<float>(width * height * 3);
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        int dst_index = i + width * j + width * height * k;
        int src_index = k + 3 * i + 3 * width * j;
        img.data[dst_index] = (float)imgBuf[src_index] / 255.;
      }
    }
  }
  free(imgBuf);

  object_detector *detector = nullptr;

  if (strcmp(argv[1], "yolov3_tiny") == 0) {
    printf("Loading model from %s\n", argv[2]);
    yolov3_tiny_detector *v3_detector = new yolov3_tiny_detector(argv[2]);

    if (v3_detector->model.width != width || v3_detector->model.height != height) {
      fprintf(stderr, "Model expects %dx%d image got %dx%d\n", v3_detector->model.width,
              v3_detector->model.height, width, height);
      return -1;
    }
    printf("yolov3_tiny loaded\n");
    detector = v3_detector;
  }

  if (strcmp(argv[1], "yolov8n") == 0) {
    printf("Loading model from %s\n", argv[2]);
    yolov8_detector *v8_detector = new yolov8_detector(argv[2]);

    if (v8_detector->model.width != width || v8_detector->model.height != height) {
      fprintf(stderr, "Model expects %dx%d image got %dx%d\n", v8_detector->model.width,
              v8_detector->model.height, width, height);
      return -1;
    }
    printf("yolov8 loaded\n");
    detector = v8_detector;
  }

  if (!detector) {
    fprintf(stderr, "Unknown model %s\n", argv[1]);
    return -1;
  }

  printf("Running model\n");
  std::vector<detection> detections = detector->detect(img);
}