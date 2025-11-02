#include <SDL.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 800
#define N 20000  // You can raise this later
#define DT 0.02f
#define FRICTIONAL_HALFLIFE 0.40f
#define RMAX 0.04f
#define M 6
#define FORCE_FACTOR 100.0f
#define CELL_SIZE (RMAX * 2.5f)
#define GRID_SIZE_X (int)(2.0f / CELL_SIZE)
#define GRID_SIZE_Y (int)(2.0f / CELL_SIZE)

typedef struct Particle {
    int color;
    float x, y, z;
    float vx, vy, vz;
    struct Particle* next;
} Particle;

Particle particles[N];
Particle* grid[GRID_SIZE_X][GRID_SIZE_Y];

float matrix[M][M];
float frictionFactor;

float randFloat(float a, float b) {
    return a + (b - a) * ((float) rand() / RAND_MAX);
}

void makeRandomMatrix() {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
            matrix[i][j] = randFloat(-1.0f, 1.0f);
}

float force(float r, float a) {
    const float beta = 0.3f;
    if (r < beta) return r / beta - 1.0f;
    else if (r < 1.0f) return a * (1.0f - fabsf(2.0f * r - 1.0f - beta) / (1.0f - beta));
    else return 0.0f;
}

void initParticles() {
    for (int i = 0; i < N; i++) {
        particles[i].color = rand() % M;
        particles[i].x = randFloat(-1.0f, 1.0f);
        particles[i].y = randFloat(-1.0f, 1.0f);
        particles[i].z = randFloat(-1.0f, 1.0f);
        particles[i].vx = particles[i].vy = particles[i].vz = 0;
        particles[i].next = NULL;
    }
}

void clearGrid() {
    for (int i = 0; i < GRID_SIZE_X; i++)
        for (int j = 0; j < GRID_SIZE_Y; j++)
            grid[i][j] = NULL;
}

void insertIntoGrid(Particle* p) {
    int gx = (int)((p->x + 1.0f) / 2.0f * GRID_SIZE_X);
    int gy = (int)((p->y + 1.0f) / 2.0f * GRID_SIZE_Y);
    if (gx >= 0 && gx < GRID_SIZE_X && gy >= 0 && gy < GRID_SIZE_Y) {
        p->next = grid[gx][gy];
        grid[gx][gy] = p;
    }
}


void updateParticles() {
    clearGrid();
    for (int i = 0; i < N; i++) {
        insertIntoGrid(&particles[i]);
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Particle* p = &particles[i];
        float fx = 0, fy = 0, fz = 0;

        int gx = (int)((p->x + 1.0f) / 2.0f * GRID_SIZE_X);
        int gy = (int)((p->y + 1.0f) / 2.0f * GRID_SIZE_Y);

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = gx + dx;
                int ny = gy + dy;
                if (nx < 0 || nx >= GRID_SIZE_X || ny < 0 || ny >= GRID_SIZE_Y)
                    continue;

                Particle* neighbor = grid[nx][ny];
                while (neighbor) {
                    if (neighbor != p) {
                        float rx = neighbor->x - p->x;
                        float ry = neighbor->y - p->y;
                        float rz = neighbor->z - p->z;
                        float r2 = rx * rx + ry * ry + rz * rz;
                        float r = sqrtf(r2);
                        if (r > 0 && r < RMAX) {
                            float f = force(r / RMAX, matrix[p->color][neighbor->color]);
                            fx += (rx / r) * f;
                            fy += (ry / r) * f;
                            fz += (rz / r) * f;
                        }
                    }
                    neighbor = neighbor->next;
                }
            }
        }

        fx *= RMAX * FORCE_FACTOR;
        fy *= RMAX * FORCE_FACTOR;
        fz *= RMAX * FORCE_FACTOR;

        p->vx *= frictionFactor;
        p->vy *= frictionFactor;
        p->vz *= frictionFactor;

        p->vx += fx * DT;
        p->vy += fy * DT;
        p->vz += fz * DT;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        particles[i].x += particles[i].vx * DT;
        particles[i].y += particles[i].vy * DT;
        particles[i].z += particles[i].vz * DT;
    }
}

void hsvToRgb(float h, int* r, int* g, int* b) {
    float s = 1.0f, v = 1.0f;
    int i = (int)(h / 60.0f) % 6;
    float f = h / 60.0f - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    switch (i) {
        case 0: *r = v * 255; *g = t * 255; *b = p * 255; break;
        case 1: *r = q * 255; *g = v * 255; *b = p * 255; break;
        case 2: *r = p * 255; *g = v * 255; *b = t * 255; break;
        case 3: *r = p * 255; *g = q * 255; *b = v * 255; break;
        case 4: *r = t * 255; *g = p * 255; *b = v * 255; break;
        case 5: *r = v * 255; *g = p * 255; *b = q * 255; break;
    }
}

void drawParticles(SDL_Renderer* renderer, SDL_Texture* texture, uint32_t* pixels) {
    memset(pixels, 0, WIDTH * HEIGHT * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        Particle* p = &particles[i];
        float f = 1.0f / (p->z + 2.0f);
        int screenX = (int)((f * p->x + 1.0f) * 0.5f * WIDTH);
        int screenY = (int)((f * p->y + 1.0f) * 0.5f * HEIGHT);
        if (screenX >= 0 && screenX < WIDTH && screenY >= 0 && screenY < HEIGHT) {
            int r, g, b;
            hsvToRgb((360.0f * p->color) / M, &r, &g, &b);
            pixels[screenY * WIDTH + screenX] = (255 << 24) | (r << 16) | (g << 8) | b;
        }
    }

    SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

int main(int argc, char* argv[]) {
    srand((unsigned int) time(NULL));
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Optimized Particle Simulator", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    uint32_t* pixels = (uint32_t*) malloc(WIDTH * HEIGHT * sizeof(uint32_t));

    makeRandomMatrix();
    initParticles();
    frictionFactor = powf(0.5f, DT / FRICTIONAL_HALFLIFE);

    int running = 1;
    SDL_Event e;
    while (running) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
        }

        updateParticles();
        drawParticles(renderer, texture, pixels);
    }

    free(pixels);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

