#pragma once
#define _USE_MATH_DEFINES


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constants.h"
#include "parameters.h"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>




using namespace std;

class Fish
{

private:
	float x;
	float y;

	float vx;
	float vy;

	
	float speed=100;
	float maxChangeOfDegreePerSecond;

	Parameters* parameters;

public:
	float colorId = 0;
	int id = 1;

	__host__ __device__  Fish() : x(0), y(0) {
		vx = rand() % 100 - 50;
		vy = rand() % 100 - 50;
		normalize(vx, vy);
		vx = vx * speed;
		vy = vy * speed;
		colorId = (id == 0) ? 0 : 1;
	}

	__host__ __device__  Fish(float x, float y): Fish() {
		this->x = x;
		this->y = y;
	}

	__host__ __device__ void SetParameters(Parameters* parameters) {
		this->parameters = parameters;
		speed = parameters->speed;
		maxChangeOfDegreePerSecond = parameters->maxChangeOfDegreePerSecond;
	}

	__host__ __device__ float GetX() const {
		return x;
	}

	__host__ __device__ float GetY() const {
		return y;
	}

	__host__ __device__  void SetCordinates(float x, float y) {
		this->x = x;
		this->y = y;
	}

	__host__ __device__  char CheckPointSide(float x1, float y1, float x2, float y2, float px, float py) const {
		float value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
		return (value > 0) ? 'L' : 'R';
	}

	__host__ __device__ float Distance(const Fish& fish) const {
		return sqrt((x - fish.x) * (x - fish.x) + (y - fish.y) * (y - fish.y));
	}

	__host__ __device__ bool Angle(float angle, const Fish& fish) const {
		float direction = atan2(vy, vx) * 180 / M_PI;
		float fishAngle = atan2(fish.y - y, fish.x - x) * 180 / M_PI;

		float degreeDifference = fabs(direction - fishAngle);

		if (degreeDifference > 180)
		{
			degreeDifference = 360 - degreeDifference;
		}
		if (degreeDifference < angle / 2.0f)
			return true;

		return false;
	}

	__host__ __device__ static int calculateIndexOfMesh(float x, float y) {
		int row = y / MESH_SIZE;
		int col = x / MESH_SIZE;
		if (x >= SCR_WIDTH)
			col = NUM_COLUMNS - 1;
		if (y >= SCR_HEIGHT)
			row = NUM_ROWS - 1;
		if (x < 0)
			col = 0;
		if (y < 0)
			row = 0;
		return  row * NUM_COLUMNS + col;
	}


	__host__ __device__ int GetNeighbors(Fish* fishes, float distance, float angle, Fish** neighbors,  int* dev_indexes, int* dev_headsIndex) {
		int count = 0;

		int index = calculateIndexOfMesh(x, y);

		int list_index[9];
		for (int i = 0; i < 9; i++)
			list_index[i] = -1;
		list_index[0] = index;
		if (index % NUM_COLUMNS != 0)
			list_index[1] = index - 1;
		if (index % NUM_COLUMNS != NUM_COLUMNS -1)
			list_index[2] = index + 1;
		if (index - NUM_COLUMNS >0 )
			list_index[3] = index - NUM_COLUMNS;
		if (index + NUM_COLUMNS <NUM_SQUARES)
			list_index[4] = index + NUM_COLUMNS;
		if (index % NUM_COLUMNS != 0 && index - NUM_COLUMNS > 0)
			list_index[5] = index - 1 - NUM_COLUMNS;
		if (index % NUM_COLUMNS != NUM_COLUMNS - 1 && index - NUM_COLUMNS > 0)
			list_index[6] = index + 1 - NUM_COLUMNS;
		if (index % NUM_COLUMNS != 0 && index + NUM_COLUMNS < NUM_SQUARES)
			list_index[7] = index - 1 + NUM_COLUMNS;
		if (index % NUM_COLUMNS != NUM_COLUMNS - 1 && index + NUM_COLUMNS < NUM_SQUARES)
			list_index[8] = index + 1 + NUM_COLUMNS;

		

		for (int i = 0; i < 9; i++)
		{
			if (list_index[i] == -1)
				continue;
			int indexStart = dev_headsIndex[list_index[i]];
			int indexEnd = (list_index[i] == NUM_SQUARES - 1) ? NUM_FISH : dev_headsIndex[list_index[i] + 1];
			

			for (int j = indexStart; j < indexEnd; j++)
			{
				Fish* fish = &fishes[dev_indexes[j]];
				if (this == fish)
					continue;
				if (Distance(*fish) < distance && Angle(angle, *fish)) {
					neighbors[count++] = fish;
				}
			}
			
		}
		
		return count;
	}

	__host__ __device__ void CalculateAvoidVelocity(Fish** neighbors, int n, float& newVx, float& newVy) const
	{
		if (n <= 0)
		{
			newVx = 0;
			newVy = 0;
			return;
		}

		float closeDx = 0;
		float closeDy = 0;
		for (int i = 0; i < n; i++)
		{
			float dist=Distance(*neighbors[i]);
			closeDx += (x - neighbors[i]->x)/ (dist*dist);
			closeDy += (y - neighbors[i]->y) / (dist*dist);
			
		}

		newVx = closeDx;
		newVy = closeDy;
	}

	__host__ __device__ void CalculateAligmentVelocity(Fish** neighbors, int n, float& newVx, float& newVy) const
	{
		if (n <= 0)
		{
			newVx = 0;
			newVy = 0;
			return;
		}
		float avgVx = 0;
		float avgVy = 0;
		for (int i = 0; i < n; i++)
		{
			avgVx += neighbors[i]->vx;
			avgVy += neighbors[i]->vy;
		}

		newVx = avgVx / n;
		newVy = avgVy / n;

	}

	__host__ __device__ void CalculateCohesionVelocity(Fish** neighbors, int n, float& newVx, float& newVy) const
	{
		if (n <= 0)
		{
			newVx = 0;
			newVy = 0;
			return;
		}
		float avgX = 0;
		float avgY = 0;
		for (int i = 0; i < n; i++)
		{
			avgX += neighbors[i]->x;
			avgY += neighbors[i]->y;
		}

		newVx = avgX / n - x;
		newVy = avgY / n - y;

	}

	__host__ __device__ void CalculateObsticleAvoidance(float& newVx, float& newVy)
	{
		
		if (x < 0 || x > WIDTH || y < 0 || y > HEIGHT)
		{
			newVx = WIDTH / 2 - x;
			newVy = HEIGHT / 2 - y;
		}
		/*if (x < 0)
			x = SCR_WIDTH;
		if (x > SCR_WIDTH)
			x = 0;
		if (y < 0)
			y = SCR_HEIGHT;
		if (y > SCR_HEIGHT)
			y = 0;*/
		/*if (sqrt((x - WIDTH / 2) * (x - WIDTH / 2) + (y - HEIGHT / 2) * (y - HEIGHT / 2)) > HEIGHT / 2)
		{
			newVx = WIDTH / 2 - x;
			newVy = HEIGHT / 2 - y;
		}*/
	}
	
	__host__ __device__ void CalculateDesiredVelocity(Fish* fishes, int* dev_indexes, int* dev_headsIndex, float& newVx, float& newVy) {
		

		Fish* neighbors[NUM_FISH];

		float avoidVelocityX = 0;
		float avoidVelocityY = 0;
		int count = GetNeighbors(fishes,parameters->avoidDistance, parameters->avoidAngle, neighbors, dev_indexes, dev_headsIndex);
		CalculateAvoidVelocity(neighbors, count, avoidVelocityX, avoidVelocityY);
		normalize(avoidVelocityX, avoidVelocityY);

		float aligmentVelocityX = 0;
		float aligmentVelocityY = 0;
		count = GetNeighbors(fishes,parameters->alignDistance, parameters->alignAngle, neighbors, dev_indexes, dev_headsIndex);
		CalculateAligmentVelocity(neighbors, count, aligmentVelocityX, aligmentVelocityY);
		normalize(aligmentVelocityX, aligmentVelocityY);

		float cohesionVelocityX = 0;
		float cohesionVelocityY = 0;
		count = GetNeighbors(fishes, parameters->cohesionDistance, parameters->cohesionAngle, neighbors, dev_indexes, dev_headsIndex);
		CalculateCohesionVelocity(neighbors, count, cohesionVelocityX, cohesionVelocityY);
		normalize(cohesionVelocityX, cohesionVelocityY);


		newVx = parameters->alignWeight * aligmentVelocityX + parameters->cohesionWeight * cohesionVelocityX + parameters->avoidWeight * avoidVelocityX;
		newVy = parameters->alignWeight * aligmentVelocityY + parameters->cohesionWeight * cohesionVelocityY + parameters->avoidWeight * avoidVelocityY;

		CalculateObsticleAvoidance(newVx, newVy);


		normalize(newVx, newVy);
		newVx *= speed;
		newVy *= speed;
		if (newVx == 0 && newVy == 0)
		{
			newVx = vx;
			newVy = vy;
		}

	}

	__host__ __device__ void ChangeVelocity(float newVx, float newVy, float dt)
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

		float maxChangeOfDegree = parameters->maxChangeOfDegreePerSecond * dt;

		if (degreeDifference < maxChangeOfDegree)
		{
			vx = newVx;
			vy = newVy;
			return;
		}

		float newDegree = currentDegree + signOfDegree * maxChangeOfDegree;
		float newVx2 = cos(newDegree * M_PI / 180) * speed;
		float newVy2 = sin(newDegree * M_PI / 180) * speed;

		vx = newVx2;
		vy = newVy2;
	}

	__host__ __device__ void UpdatePositionKernel(Fish* fishes, int* dev_indexes, int* dev_headsIndex, float dt, Parameters* parameters) {

		SetParameters(parameters);
		if (parameters->stop_simulation) return;


		float newVx=30;
		float newVy=30;
		CalculateDesiredVelocity(fishes,dev_indexes, dev_headsIndex, newVx, newVy);

		ChangeVelocity(newVx, newVy, dt);

		x += vx * dt;
		y += vy * dt;
	}

	__host__ __device__ int sign(float x) {
		return (x > 0) - (x < 0);
	}

	__host__ __device__ void normalize(float& x, float& y) {
		float len = sqrtf(x * x + y * y);
		if (len > 0) {
			x = x / len;
			y = y / len;
		}
	}

	__host__ __device__ void SetVertexes(float* arr)
	{
		arr[0] = x + 5;
		arr[1] = y;

		arr[3] = colorId;
		arr[7] = colorId;
		arr[11] = colorId;

		arr[4] = x;
		arr[5] = y - 1.5;

		arr[8] = x;
		arr[9] = y + 1.5;

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
		x = (x - SCR_WIDTH /2) / (SCR_WIDTH /2);
		y = (y - SCR_HEIGHT /2) / (SCR_HEIGHT /2);
	}

};

