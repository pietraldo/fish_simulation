#pragma once
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	float vx = 1;
	float vy = 0.4;

	/*__host__ __device__  static float Speed;
	__host__ __device__  static float MaxChangeOfDegreePerSecond;

	__host__ __device__  static int FishId;*/
	float Speed = 10.0f;
	float MaxChangeOfDegreePerSecond = 200.0f;
	int id=1;

public:
	float colorId = 0;

	__host__ __device__  Fish() : x(0), y(0) {
		vx = rand() % 100 - 50;
		vy = rand() % 100 - 50;
		normalize(vx, vy);
		vx = vx * Speed;
		vy = vy * Speed;
		//id = FishId++;
		colorId = (id == 0) ? 0 : 1;
	}

	__host__ __device__  Fish(float x, float y) :x(x), y(y) {

		vx = rand() % 100 - 50;
		vy = rand() % 100 - 50;
		normalize(vx, vy);
		vx = vx * Speed;
		vy = vy * Speed;
		//id = FishId++;
		colorId = (id == 0) ? 0 : 1;
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

	__host__ __device__ int GetNeighbors(Fish* fishes, int n, float distance, float angle, Fish** neighbors) {
		int count = 0;

		for (int i = 0; i < n; i++) {
			if (this == &fishes[i])
				continue;
			if (Distance(fishes[i]) < distance && Angle(angle, fishes[i])) {
				neighbors[count++]=&fishes[i];
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
		for (int i=0; i<n;i++)
		{
			closeDx += x - neighbors[i]->x;
			closeDy += y - neighbors[i]->y;
		}
		
		newVx = closeDx / n;
		newVy = closeDy / n;
	}

	__host__ __device__ void CalculateAligmentVelocity(vector<Fish*> const& neighbors, float& newVx, float& newVy) const
	{
		float avgVx = 0;
		float avgVy = 0;
		for (Fish const* neighbor : neighbors)
		{
			avgVx += neighbor->vx;
			avgVy += neighbor->vy;
		}
		if (neighbors.size() > 0)
		{
			newVx = avgVx / (float)neighbors.size();
			newVy = avgVy / (float)neighbors.size();
		}
	}

	__host__ __device__ void CalculateCohesionVelocity(vector<Fish*> const& neighbors, float& newVx, float& newVy) const
	{
		float avgX = 0;
		float avgY = 0;
		for (Fish const* neighbor : neighbors)
		{
			avgX += neighbor->x;
			avgY += neighbor->y;
		}
		if (neighbors.size() > 0)
		{
			newVx = avgX / (float)neighbors.size();
			newVy = avgY / (float)neighbors.size();
		}
	}

	__host__ __device__ void CalculateObsticleAvoidance(float& newVx, float& newVy)
	{
		//float avoidanceRange = 50;

		//// check if it is good direction
		//float dirX = newVx;
		//float dirY = newVy;

		//normalize(dirX, dirY);
		//dirX *= avoidanceRange;
		//dirY *= avoidanceRange;

		//// no colicion good to return
		//if (x + dirX > 0 && x + dirX < 800 && y + dirY>0 && y + dirY < 600) return;

		//cout<<"colision "<<dirX<<" "<<dirY << endl;
		//newVx = 0;
		//newVy = 0;

		//while()

		/*float dirX = vx;
		float dirY = vy;

		normalize(dirX, dirY);
		dirX *= avoidanceRange;
		dirY *= avoidanceRange;

		while(x + dirX > 700)
		{
			
		}*/
		/*if (sqrt((x - 400) * (x - 400) + (300 - y) * (300 - y)) > 300)
		{
			newVx = 400 - x;
			newVy = 300 - y;
		}*/
		if(x<0)
			x = 800;
		if (x > 800)
			x = 0;
		if (y < 0)
			y = 600;
		if (y > 600)
			y = 0;
		
	}

	__host__ __device__ void CalculateDesiredVelocity(Fish* fishes, int n, float& newVx, float& newVy, int mouseX, int mouseY , float aligmentWeight, float cohesionWeight, float avoidWeight) {


		float avoidDistance = 100;
		float avoidAngle = 359;

		float aligmentDistance = 30;
		float aligmentAngle = 120;

		float cohesionDistance = 20;
		float cohesionAngle = 120;

		//float aligmentWeight = 0;
		//float cohesionWeight = 0;
		//float avoidWeight = 2.9;

		// coloring fishes
		/*if (id == 0)
		{
			for (int i = 0; i < n; i++)
				fishes[i].colorId = 1;
			colorId = 0;

			vector<Fish*> avoidNeighbors = GetNeighbors(fishes, n, avoidDistance, avoidAngle);
			for (int i = 0; i < avoidNeighbors.size(); i++)
				avoidNeighbors[i]->colorId = 2;
		}*/



		float avoidVelocityX = 0;
		float avoidVelocityY = 0;

		Fish* avoidNeighbors[100];

		int count = GetNeighbors(fishes, n, avoidDistance, avoidAngle,avoidNeighbors);
		CalculateAvoidVelocity(avoidNeighbors, count, avoidVelocityX, avoidVelocityY);



		float aligmentVelocityX = 0;
		float aligmentVelocityY = 0;
		/*vector<Fish*> aligmentNeighbors = GetNeighbors(fishes, n, aligmentDistance, aligmentAngle);
		CalculateAligmentVelocity(aligmentNeighbors, aligmentVelocityX, aligmentVelocityY);*/

		float cohesionVelocityX = 0;
		float cohesionVelocityY = 0;
		/*vector<Fish*> cohesionNeighbors = GetNeighbors(fishes, n, cohesionDistance, cohesionAngle);
		CalculateCohesionVelocity(cohesionNeighbors, cohesionVelocityX, cohesionVelocityY);

		*/

		newVx = aligmentWeight * aligmentVelocityX + cohesionWeight * cohesionVelocityX + avoidWeight * avoidVelocityX;
		newVy = aligmentWeight * aligmentVelocityY + cohesionWeight * cohesionVelocityY + avoidWeight * avoidVelocityY;

		CalculateObsticleAvoidance(newVx, newVy);
		

		normalize(newVx, newVy);
		newVx *= Speed;
		newVy *= Speed; 
		if (newVx == 0 && newVy == 0)
		{
			newVx = vx;
			newVy = vy;
			
		}
		colorId = (count > 0) ? 2 : 1;
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

	__host__ __device__ void UpdatePositionKernel(Fish* fishes, int n, float dt, int mouseX, int mouseY, float alignWeight, float cohesionWeight, float avoidWeight) {

		float newVx;
		float newVy;
		CalculateDesiredVelocity(fishes, n, newVx, newVy, mouseX, mouseY, alignWeight, cohesionWeight, avoidWeight);


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
		arr[0] = x + 10;
		arr[1] = y;

		arr[3] = colorId;
		arr[7] = colorId;
		arr[11] = colorId;

		arr[4] = x;
		arr[5] = y - 2;

		arr[8] = x;
		arr[9] = y + 2;

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

