#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <smmintrin.h>

void mandelbrot(float x0, float y0, float x1, float y1,
                int width, int height, int maxIterations,
                int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    int i, j, k;

    for (j = 0; j < height; j++)
    {
        for (i = 0; i < width; i+=4)
        {
            // float x = x0 + i * dx;
	    __m128 iv = _mm_set_ps(i+3,i+2,i+1,i);
	    __m128 x0v = _mm_set_ps1(x0);
	    __m128 dxv = _mm_set_ps1(dx);
	    __m128 xv = _mm_add_ps(x0v,_mm_mul_ps(iv,dxv));
            // float y = y0 + j * dy;
	    __m128 jv = _mm_set_ps1(j);
	    __m128 y0v = _mm_set_ps1(y0);
	    __m128 dyv = _mm_set_ps1(dy);
	    __m128 yv = _mm_add_ps(y0v,_mm_mul_ps(jv,dyv));
            int index = j*width + i;
            // float zr = x, zi = y;
	    __m128 zrv = xv;
	    __m128 ziv = yv;

	    __m128 kv = _mm_set_ps1(0);
            for (k = 0; k < maxIterations; k++)
	    {
		__m128 zrv2 = _mm_mul_ps(zrv,zrv);
		__m128 ziv2 = _mm_mul_ps(ziv,ziv);

		// if (zr*zr + zi*zi > 4.f)
		__m128 testv = _mm_add_ps(zrv2,ziv2);
		__m128 mask = _mm_cmple_ps(testv,_mm_set_ps1(4));
		if (_mm_movemask_ps(mask) == 0)
		    break;
		kv = _mm_add_ps(kv,_mm_and_ps(mask,_mm_set_ps1(1)));

                // float newr = zr*zr - zi*zi;
		__m128 newrv = _mm_sub_ps(zrv2,ziv2);
                // float newi = 2.f*zr*zi;
		__m128 zriv = _mm_mul_ps(zrv,ziv);
		__m128 newiv = _mm_mul_ps(_mm_set_ps1(2),zriv);
                // zr = x + newr;
		zrv = _mm_add_ps(xv,newrv);
                // zi = y + newi;		
		ziv = _mm_add_ps(yv,newiv);
            }

            // output[index] = k;
	    _mm_storeu_si128((__m128i*) &output[index],_mm_cvtps_epi32(kv));
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

    printf("[mandelbrot SIMD]:\t\t[%.3f] sec.\n", (t2.tv_sec-t1.tv_sec)+(t2.tv_usec-t1.tv_usec)/(double)1000000);
    writePPM(buf, width, height, "mandelbrot-SIMD.ppm");
}
