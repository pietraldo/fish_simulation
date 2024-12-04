#pragma once
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>

class Fish
{
	
private:
	float x;
	float y;

	float vx = 1;
	float vy = 0.4;

	static float Speed;
	static float MaxChangeOfDegreePerSecond;
	float colorId = 2;
	static int FishId;
	int id;
public:

	Fish(): x(0), y(0){
		id = FishId++;
		colorId = (id == 0) ? 0 : 1;
		std::cout << colorId <<" "<<id << std::endl;
	}

	Fish(float x, float y):x(x), y(y){	
		id = FishId++;
		colorId = (id == 0) ? 0 : 1;
		std::cout << colorId << " "<<id<< std::endl;
	}

	void SetCordinates(float x, float y) {
		this->x = x;
		this->y = y;
	}

	__device__  char CheckPointSide(float x1, float y1, float x2, float y2, float px, float py) const {
		float value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
		return (value > 0) ? 'L' : 'R';
	}

	/*float Distance(const Fish& fish) const {
		return sqrt((x - fish.x) * (x - fish.x) + (y - fish.y) * (y - fish.y));
	}

	bool Angle(float angle, const Fish& fish) const {
		float direction = atan2(vy, vx) * 180 / M_PI;
		float angleLow = direction - angle / 2;
		float angleHigh = direction + angle / 2;
		float fishAngle = atan2(fish.y, fish.x) * 180 / M_PI;


	}

	vector<Fish> GetNeighbors(Fish* fishes,int n, int distance, float angle) {
		vector<Fish> neighbors;

		for (int i = 0; i < n; i++) {
			if (this == &fishes[i]) continue;
			if (Distance(fishes[i]) < distance && Angle(angle, fishes[i])) {
				neighbors.push_back(fishes[i]);
			}
		}
		return neighbors;
	}*/

	void CalculateAvoidVelocity(Fish* fishes, float& newVx, float& newVy)
	{

	}

	void CalculateDesiredVelocity(Fish* fishes, float& newVx, float& newVy, int mouseX, int mouseY) {
		
		
		newVx = mouseX-x;
		newVy = (600-mouseY)-y;
		normalize(newVx, newVy);
		newVx *= Speed;
		newVy *= Speed;
		//std::cout << x << " " << y << std::endl;
		
	}

	void ChangeVelocity(float newVx, float newVy, float dt)
	{
		float desiredDegree = atan2(newVy, newVx) * 180 / M_PI;
		float currentDegree = atan2(vy, vx) * 180 / M_PI;

		
		float degreeDifference = fabs(desiredDegree - currentDegree);
		int signOfDegree = sign(desiredDegree - currentDegree);
		
		if (degreeDifference > 180)
		{
			degreeDifference = 360 - degreeDifference;
			signOfDegree = -signOfDegree;
		}
			
		float maxChangeOfDegree = MaxChangeOfDegreePerSecond * dt;

		if (degreeDifference < maxChangeOfDegree)
		{
			vx = newVx;
			vy = newVy;
			return;
		}
		
		float newDegree = currentDegree + signOfDegree * maxChangeOfDegree;
		float newVx2 = cos(newDegree * M_PI / 180) * Speed;
		float newVy2 = sin(newDegree * M_PI / 180) * Speed;

		vx = newVx2;
		vy = newVy2;
	}

	void UpdatePositionKernel(Fish* fishes, int n, float dt, int mouseX, int mouseY) {

		float newVx;
		float newVy;
		CalculateDesiredVelocity(fishes,newVx, newVy, mouseX, mouseY);
		
		ChangeVelocity(newVx, newVy, dt);

		x += vx * dt;
		y += vy * dt;
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

		arr[3] = colorId;
		arr[7] = colorId;
		arr[11] = colorId;

		arr[4] = x;
		arr[5] = y - 4;

		arr[8] = x;
		arr[9] = y + 4;

		ChangeCordinates(arr[0], arr[1]);
		ChangeCordinates(arr[4], arr[5]);
		ChangeCordinates(arr[8], arr[9]);

		float cx = 0, cy = 0;
		cx = (arr[0] + arr[4] + arr[8]) / 3;
		cy = (arr[1] + arr[5] + arr[9]) / 3;

		float degreeInRadians = atan2(-vy, vx);


		rotatePointAroundCenter(arr[0], arr[1], cx, cy, degreeInRadians);
		rotatePointAroundCenter(arr[4], arr[5], cx, cy, degreeInRadians);
		rotatePointAroundCenter(arr[8], arr[9], cx, cy, degreeInRadians);


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

