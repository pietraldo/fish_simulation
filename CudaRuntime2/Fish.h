#pragma once
#define _USE_MATH_DEFINES

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

	static int maxLeft;
	static int maxRight;
	static int maxTop;
	static int maxBottom;

	Fish(float x, float y) {
		this->x = x;
		this->y = y;
	}
	void CalculateNewPosition(float dt) {
		x += vx * dt * speed;
		y += vy * dt * speed;

		if (x < maxLeft) {
			x = maxLeft;
			vx = -vx;
		}
		if (x > maxRight) {
			x = maxRight;
			vx = -vx;
		}
		if (y < maxTop) {
			y = maxTop;
			vy = -vy;
		}
		if (y > maxBottom) {
			y = maxBottom;
			vy = -vy;
		}
	}

	void SetVertexes(float* arr)
	{
		arr[0] = x + 3;
		arr[1] = y;

		arr[3] = x;
		arr[4] = y - 1;

		arr[6] = x;
		arr[7] = y + 1;

		ChangeCordinates(arr[0], arr[1]);
		ChangeCordinates(arr[3], arr[4]);
		ChangeCordinates(arr[6], arr[7]);

		float cx = 0, cy = 0;
		cx = (arr[0] + arr[3] + arr[6]) / 3;
		cy = (arr[1] + arr[4] + arr[7]) / 3;

		float degree = tan(abs(vy) / abs(vx)) * 180 / M_PI;
		if (vx < 0) {
			degree = 180 - degree;
		}
		if (vy > 0) {
			degree = -degree;
		}
		

		//std::cout << vy << " " << vx << " " << degree << std::endl;

		rotatePointAroundCenter(arr[0], arr[1], cx, cy, degree);
		rotatePointAroundCenter(arr[3], arr[4], cx, cy, degree);
		rotatePointAroundCenter(arr[6], arr[7], cx, cy, degree);


	}

	void rotatePointAroundCenter(float& x, float& y, float cx, float cy, float angleDegrees) {
		float radians = angleDegrees * M_PI / 180.0f;
		float cosTheta = cos(radians);
		float sinTheta = sin(radians);

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

	void ChangeCordinates(float& x, float& y) {
		x = (x - 400) / 400;
		y = (y - 300) / 300;
	}

};

