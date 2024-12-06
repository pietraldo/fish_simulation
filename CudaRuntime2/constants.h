#pragma once

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;
const unsigned int MESH_SIZE = 10;

const unsigned int NUM_FISH = 1000;
const unsigned int BLOCK_SIZE = 100;

const int NUM_ROWS = SCR_HEIGHT / MESH_SIZE;
const int NUM_COLUMNS = SCR_WIDTH / MESH_SIZE;
const int NUM_SQUARES = NUM_ROWS * NUM_COLUMNS;

const float AVOID_WEIGHT = 4.1f;
const float ALIGN_WEIGHT = 4.3f;
const float COHESION_WEIGHT = 1.0f;
const bool STOP_SIMULATION = false;
const float SPEED1 = 25.0f;
const float SPEED2 = 100.0f;
const float MAX_CHANGE_OF_DEGREE_PER_SECOND1 = 240.0f;
const float MAX_CHANGE_OF_DEGREE_PER_SECOND2 = 1200.0f;
const float AVOID_DISTANCE = 6.0f;
const float ALIGN_DISTANCE = 15.0f;
const float COHESION_DISTANCE = 20.0f;
const float AVOID_ANGLE = 359.0f;
const float ALIGN_ANGLE = 120.0f;
const float COHESION_ANGLE = 350.0f;

const int FISH_WIDTH1 = 15;
const int FISH_HEIGHT1 = 50;
const int FISH_WIDTH2 = 10;
const int FISH_HEIGHT2 = 40;