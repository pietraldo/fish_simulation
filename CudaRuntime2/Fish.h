#pragma once
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cmath>

class Fish
{

private:
	float x;
	float y;

	float vx = 1;
	float vy = 0.4;

	static float Speed;
	static float MaxChangeOfDegreePerSecond;

public:

	Fish(): x(0), y(0){}

	Fish(float x, float y):x(x), y(y) {	}

	__device__  char CheckPointSide(float x1, float y1, float x2, float y2, float px, float py) const {
		float value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
		return (value > 0) ? 'L' : 'R';
	}

	int lastMouseX = 0;
	float lastVx = 100;
	void CalculateDesiredVelocity(float& newVx, float& newVy, int mouseX, int mouseY) {
		
		
		newVx = 400-x;
		newVy = 300-y;
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
			
		std::cout << desiredDegree - currentDegree << std::endl;
		float maxChangeOfDegree = MaxChangeOfDegreePerSecond * dt;

		if (degreeDifference < maxChangeOfDegree)
		{
			vx = newVx;
			vy = newVy;
			return;
		}
		
		//int signOfDegree = sign(desiredDegree - currentDegree);
		float newDegree = currentDegree + signOfDegree * maxChangeOfDegree;
		float newVx2 = cos(newDegree * M_PI / 180) * Speed;
		float newVy2 = sin(newDegree * M_PI / 180) * Speed;

		//std::cout << "Current Degree: " << currentDegree << " Desired Degree: " << desiredDegree << " New Degree: " << newDegree << std::endl;

		//std::cout << "Current Degree: " << currentDegree << " Desired Degree: " << desiredDegree << " New Degree: " << newDegree << std::endl;
		//std::cout << "Current Velocity: " << vx << " " << vy << " New Velocity: " << newVx2 << " " << newVy2 << " Wanted Velocity" << newVx << " " << newVy << std::endl<< std::endl;
		//std::cout << newVx2 << " " << newVy2 << std::endl;
		vx = newVx2;
		vy = newVy2;
	}

	void UpdatePositionKernel(Fish* fishes, int n, float dt, int mouseX, int mouseY) {

		float newVx, newVy;
		CalculateDesiredVelocity(newVx, newVy, mouseX, mouseY);
		
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

