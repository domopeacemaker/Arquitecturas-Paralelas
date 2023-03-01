#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void mandelbrot(float x0, float y0, float x1, float y1,
                int width, int height, int maxIterations,
                int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    int i, j, k;

    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; ++i)
        {
            float x = x0 + i * dx;
            float y = y0 + j * dy;
            int index = j*width + i;
            float zr = x, zi = y;

            for (k = 0; k < maxIterations; k++)
	    {
		if (zr*zr + zi*zi > 4.f)
		    break;

                float newr = zr*zr - zi*zi;
                float newi = 2.f*zr*zi;
                zr = x + newr;
                zi = y + newi;		
            }

            output[index] = k;
        }
    }
}

/* Write a PPM image file with the image of the Mandelbrot set */
static void writePPM(int *buf, int width, int height, const char *fn) 
{
    FILE *fp = fopen(fn, "wb");
    int i, j;

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");

    for (i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? 240 : 20;
        for (j = 0; j < 3; ++j)
            fputc(c, fp);
    }

    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

int main(int argc, char *argv[]) 
{
    unsigned int width = 768;
    unsigned int height = 512;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;
    struct timeval t1, t2;

    if (argc > 1) 
    {
        if (strncmp(argv[1], "--scale=", 8) == 0) 
        {
            float scale = atof(argv[1] + 8);
            width *= scale;
            height *= scale;
        }
    }

    int maxIterations = 256;
    int *buf = calloc(width*height,sizeof(int));

    gettimeofday(&t1,NULL);
    mandelbrot(x0, y0, x1, y1, width, height, maxIterations, buf);
    gettimeofday(&t2,NULL);

    printf("[mandelbrot serial]:\t\t[%.3f] sec.\n", (t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/(double)1000000);
    writePPM(buf, width, height, "mandelbrot-serial.ppm");
}
