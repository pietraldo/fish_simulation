#pragma once
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>


class Fish
{

private:
	float x;
	float y;

	float vx = 1;
	float vy = 0.4;

	float speed = 100.0f;
public:

	__host__ __device__ Fish() {
		x = 0;
		y = 0;
	}

	__host__ __device__ Fish(float x, float y) {
		this->x = x;
		this->y = y;
	}

	__device__  char CheckPointSide(float x1, float y1, float x2, float y2, float px, float py) const {
		float value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
		return (value > 0) ? 'L' : 'R';
	}

	__host__ __device__ void CalculateNewPosition(float dt) {
		x += vx * dt * speed;
		y += vy * dt * speed;

		if (x < 0) {
			x = 0;
			vx = -vx;
		}
		if (x > 800) {
			x = 800;
			vx = -vx;
		}
		if (y < 0) {
			y = 0;
			vy = -vy;
		}
		if (y > 600) {
			y = 600;
			vy = -vy;
		}
	}

	__host__ __device__ float obliczKatMiedzyWektorami(float x1, float y1, float x2, float y2) {
		float iloczynSkalarny = x1 * x2 + y1 * y2;

		float dlugoscA = sqrt(x1 * x1 + y1 * y1);
		float dlugoscB = sqrt(x2 * x2 + y2 * y2);

		float cosTheta = iloczynSkalarny / (dlugoscA * dlugoscB);
		float katRadiany = acos(cosTheta);
		float katStopnie = katRadiany * 180.0 / M_PI;

		return katStopnie;
	}

	__host__ __device__ void UpdatePositionKernel(Fish* fishes, int n, float dt) {

		float Speed = 60;
		float AvoidRange = 16;
		float AligmentRange = 150;
		float CohesionRange = 65;

		float AvoidFactor = 351;
		float AligmentFactor = 909;
		float CohasionFactor = 199;


		float MaxSpeedDifference =0.01;
		float ChangesOfVelocity = 0.1;

		float VisionDegree = 120;


		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= n) return;

		Fish* fish = &fishes[idx];
		/*float newVx = fish->vx;
		float newVy = fish->vy;*/

		//float closeDx = 0, closeDy = 0;
		//float vx_avg = 0, vy_avg = 0;
		//int neighborsAligment = 0;

		//float xpos_avg = 0, ypos_avg = 0;
		//int neghborsCohesion = 0;

		//for (int i = 0; i < n; i++) {
		//	if (i == idx) continue;

		//	Fish other = fishes[i];
		//	float dx = other.x - fish->x;
		//	float dy = other.y - fish->y;
		//	float dist = sqrtf(dx * dx + dy * dy);

		//	float degree = atan2f(fish->vy, fish->vx);
		//	float ddegree = M_PI / 180 * VisionDegree;

		//	// Avoidance
		//	if (dist < AvoidRange) {
		//		closeDx += (fish->x - other.x);
		//		closeDy += (fish->y - other.y);
		//	}

		//	// Alignment
		//	if (dist < AligmentRange) {
		//		vx_avg += other.vx;
		//		vy_avg += other.vy;
		//		neighborsAligment++;
		//	}

		//	// Cohesion
		//	if (dist < CohesionRange) {
		//		xpos_avg += other.x;
		//		ypos_avg += other.y;
		//		neghborsCohesion++;
		//	}
		//}

		//// Average calculations
		//if (neighborsAligment > 0) {
		//	vx_avg /= neighborsAligment;
		//	vy_avg /= neighborsAligment;
		//}
		//if (neghborsCohesion > 0) {
		//	xpos_avg /= neghborsCohesion;
		//	ypos_avg /= neghborsCohesion;
		//}

		//// Apply rules
		//newVx += (vx_avg - fish->vx) * AligmentFactor;
		//newVy += (vy_avg - fish->vy) * AligmentFactor;
		//newVx += (xpos_avg - fish->x) * CohasionFactor;
		//newVy += (ypos_avg - fish->y) * CohasionFactor;
		//newVx += closeDx * AvoidFactor;
		//newVy += closeDy * AvoidFactor;

		//if (x > 700 || x < 100 || y>500 || y < 100)
		//{
		//	newVx = (400-x)/10;
		//	newVy = (300-y)/10;
		//}
		float newVx = (400 - x) / 10;
		float newVy = (300-y)/10;

		normalize(newVx, newVy);
		newVx *= Speed;
		newVy *= Speed;

		if (abs(fish->vx - newVx) > MaxSpeedDifference)
			newVx = sign(newVx) * MaxSpeedDifference;
		if (abs(fish->vy - newVy) > MaxSpeedDifference)
			newVy = sign(newVy) * MaxSpeedDifference;

		normalize(newVx, newVy);
		newVx *= Speed;
		newVy *= Speed;

		fish->vx += (newVx - fish->vx) * ChangesOfVelocity;
		fish->vy += (newVy - fish->vy) * ChangesOfVelocity;

		
		
		fish->x += fish->vx*dt;
		fish->y += fish->vy*dt;
	}

	__host__ __device__ int sign(float x) {
		return (x > 0) - (x < 0);
	}

	__host__ __device__ void normalize(float& x, float& y) {
		float mag = sqrtf(x * x + y * y);
		if (mag > 0) {
			x = x / mag;
			y = y / mag;
		}
	}

	__host__ __device__ void SetVertexes(float* arr)
	{
		arr[0] = x + 20;
		arr[1] = y;

		arr[3] = x;
		arr[4] = y - 4;

		arr[6] = x;
		arr[7] = y + 4;

		ChangeCordinates(arr[0], arr[1]);
		ChangeCordinates(arr[3], arr[4]);
		ChangeCordinates(arr[6], arr[7]);

		float cx = 0, cy = 0;
		cx = (arr[0] + arr[3] + arr[6]) / 3;
		cy = (arr[1] + arr[4] + arr[7]) / 3;

		float degreeInRadians = atan2(-vy, vx);


		rotatePointAroundCenter(arr[0], arr[1], cx, cy, degreeInRadians);
		rotatePointAroundCenter(arr[3], arr[4], cx, cy, degreeInRadians);
		rotatePointAroundCenter(arr[6], arr[7], cx, cy, degreeInRadians);


	}

	__host__ __device__  void rotatePointAroundCenter(float& x, float& y, float cx, float cy, float radians) {
		float cosTheta = static_cast<float>(cos(radians));
		float sinTheta = static_cast<float>(sin(radians));

		// Translate point to origin
		float translatedX = x - cx;
		float translatedY = y - cy;

		// Rotate
		float rotatedX = translatedX * cosTheta + translatedY * sinTheta;
		float rotatedY = -translatedX * sinTheta + translatedY * cosTheta;

		// Translate back
		x = rotatedX + cx;
		y = rotatedY + cy;
	}

	__host__ __device__  void ChangeCordinates(float& x, float& y) {
		x = (x - 400) / 400;
		y = (y - 300) / 300;
	}

};

