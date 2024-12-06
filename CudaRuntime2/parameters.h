#pragma once

struct Parameters {
	float avoidWeight;
	float alignWeight;
	float cohesionWeight;
	bool stop_simulation;
	float speed1;
	float speed2;
	float maxChangeOfDegreePerSecond1;
	float maxChangeOfDegreePerSecond2;
	float avoidDistance;
	float alignDistance;
	float cohesionDistance;
	float avoidAngle;
	float alignAngle;
	float cohesionAngle;
	int fish_number;
};

